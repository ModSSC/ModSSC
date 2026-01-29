from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    concat_data,
    cycle_batch_indices,
    cycle_batches,
    ensure_float_tensor,
    ensure_model_bundle,
    ensure_model_device,
    extract_logits,
    get_torch_device,
    get_torch_len,
    num_batches,
    slice_data,
)
from modssc.inductive.methods.utils import (
    detect_backend,
    ensure_1d_labels_torch,
    ensure_torch_data,
)
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec

logger = logging.getLogger(__name__)


def _sharpen(probs: Any, *, temperature: float) -> Any:
    if temperature <= 0:
        raise InductiveValidationError("temperature must be > 0.")
    if temperature == 1.0:
        return probs
    power = 1.0 / float(temperature)
    sharpened = probs.pow(power)
    denom = sharpened.sum(dim=1, keepdim=True)
    denom = denom.clamp_min(1e-12)
    return sharpened / denom


@dataclass(frozen=True)
class FlexMatchSpec:
    """Specification for FlexMatch (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    p_cutoff: float = 0.95
    temperature: float = 0.5
    hard_label: bool = True
    thresh_warmup: bool = True
    use_cat: bool = False
    batch_size: int = 64
    max_epochs: int = 1
    detach_target: bool = True


class FlexMatchMethod(InductiveMethod):
    """FlexMatch with classwise adaptive thresholds (torch-only)."""

    info = MethodInfo(
        method_id="flexmatch",
        name="FlexMatch",
        year=2021,
        family="pseudo-label",
        supports_gpu=True,
        paper_title="FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling",
        paper_pdf="https://arxiv.org/abs/2110.08263",
        official_code="",
    )

    def __init__(self, spec: FlexMatchSpec | None = None) -> None:
        self.spec = spec or FlexMatchSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None
        self._selected_label: Any | None = None
        self._classwise_acc: Any | None = None
        self._ulb_size: int | None = None

    def _init_state(self, *, n_classes: int, device: Any) -> None:
        torch = optional_import("torch", extra="inductive-torch")
        ulb_size = self._ulb_size
        if ulb_size is None:
            raise InductiveValidationError("Unlabeled pool size is missing.")
        self._selected_label = torch.full((int(ulb_size),), -1, dtype=torch.long, device=device)
        self._classwise_acc = torch.zeros((int(n_classes),), device=device)

    def _update_classwise_acc(self) -> None:
        torch = optional_import("torch", extra="inductive-torch")
        if self._selected_label is None or self._classwise_acc is None:
            raise InductiveValidationError("FlexMatch state not initialized.")
        sel_cpu = (self._selected_label + 1).detach().cpu()
        counts = torch.bincount(sel_cpu, minlength=int(self._classwise_acc.numel()) + 1)
        counts = counts.to(self._classwise_acc.device)
        if bool(self.spec.thresh_warmup):
            denom = counts.max().clamp_min(1.0)
            self._classwise_acc = counts[1:].to(self._classwise_acc.dtype) / denom
        else:
            counts_pos = counts[1:]
            denom = counts_pos.max()
            if denom <= 0:
                self._classwise_acc = torch.zeros_like(self._classwise_acc)
            else:
                self._classwise_acc = counts_pos.to(self._classwise_acc.dtype) / denom

    def _get_idx_u(self, data: Any, *, device: Any, n_u: int) -> Any:
        torch = optional_import("torch", extra="inductive-torch")
        if data.meta is None:
            raise InductiveValidationError("FlexMatch requires data.meta with idx_u.")
        if not isinstance(data.meta, Mapping):
            raise InductiveValidationError("FlexMatch requires data.meta to be a mapping.")
        idx_u = data.meta.get("idx_u")
        if idx_u is None:
            idx_u = data.meta.get("unlabeled_idx")
        if idx_u is None:
            idx_u = data.meta.get("unlabeled_indices")
        if idx_u is None:
            raise InductiveValidationError(
                "FlexMatch requires meta.idx_u (global unlabeled indices)."
            )
        if not isinstance(idx_u, torch.Tensor):
            raise InductiveValidationError("meta.idx_u must be a torch.Tensor.")
        if idx_u.dtype != torch.int64:
            raise InductiveValidationError("meta.idx_u must be int64.")
        if idx_u.ndim != 1:
            raise InductiveValidationError("meta.idx_u must be 1D.")
        if int(idx_u.shape[0]) != int(n_u):
            raise InductiveValidationError("meta.idx_u must match X_u size.")
        if idx_u.device != device:
            raise InductiveValidationError("meta.idx_u must be on the same device as X_u.")

        ulb_size = data.meta.get("ulb_size") or data.meta.get("unlabeled_size")
        if ulb_size is None:
            # Require contiguous 0..n_u-1 if size not provided
            if int(idx_u.min().item()) != 0 or int(idx_u.max().item()) != int(n_u) - 1:
                raise InductiveValidationError(
                    "meta.idx_u must be contiguous 0..n_u-1 or provide meta.ulb_size."
                )
            uniq = torch.unique(idx_u)
            if int(uniq.numel()) != int(n_u):
                raise InductiveValidationError("meta.idx_u must contain unique indices.")
            ulb_size = int(n_u)
        else:
            if not isinstance(ulb_size, int):
                raise InductiveValidationError("meta.ulb_size must be an int.")
            if ulb_size < int(n_u):
                raise InductiveValidationError("meta.ulb_size must be >= len(idx_u).")
            if int(idx_u.max().item()) >= int(ulb_size):
                raise InductiveValidationError("meta.idx_u entries must be < meta.ulb_size.")

        self._ulb_size = int(ulb_size)
        return idx_u

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> FlexMatchMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s p_cutoff=%s temperature=%s hard_label=%s thresh_warmup=%s "
            "use_cat=%s batch_size=%s max_epochs=%s detach_target=%s has_model_bundle=%s "
            "device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.p_cutoff,
            self.spec.temperature,
            self.spec.hard_label,
            self.spec.thresh_warmup,
            self.spec.use_cat,
            self.spec.batch_size,
            self.spec.max_epochs,
            self.spec.detach_target,
            bool(self.spec.model_bundle),
            device,
            seed,
        )
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        if backend != "torch":
            raise InductiveValidationError("FlexMatch requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u_w is None or ds.X_u_s is None:
            raise InductiveValidationError("FlexMatch requires X_u_w and X_u_s.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s = ds.X_u_s
        logger.info(
            "FlexMatch sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_u_w)),
        )

        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        if int(get_torch_len(X_u_w)) == 0 or int(get_torch_len(X_u_s)) == 0:
            raise InductiveValidationError("X_u_w and X_u_s must be non-empty.")
        if int(get_torch_len(X_u_w)) != int(get_torch_len(X_u_s)):
            raise InductiveValidationError("X_u_w and X_u_s must have the same number of rows.")

        ensure_float_tensor(X_l, name="X_l")
        ensure_float_tensor(X_u_w, name="X_u_w")
        ensure_float_tensor(X_u_s, name="X_u_s")

        if y_l.dtype != torch.int64:
            raise InductiveValidationError("y_l must be int64 for torch cross entropy.")

        if self.spec.model_bundle is None:
            raise InductiveValidationError("model_bundle must be provided for FlexMatch.")
        bundle = ensure_model_bundle(self.spec.model_bundle)
        model = bundle.model
        optimizer = bundle.optimizer
        ensure_model_device(model, device=get_torch_device(X_l))

        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.max_epochs) <= 0:
            raise InductiveValidationError("max_epochs must be >= 1.")
        if float(self.spec.lambda_u) < 0:
            raise InductiveValidationError("lambda_u must be >= 0.")
        if not (0.0 <= float(self.spec.p_cutoff) <= 1.0):
            raise InductiveValidationError("p_cutoff must be in [0, 1].")
        if float(self.spec.temperature) <= 0:
            raise InductiveValidationError("temperature must be > 0.")

        idx_u_all = self._get_idx_u(
            data, device=get_torch_device(X_u_w), n_u=int(get_torch_len(X_u_w))
        )

        steps_l = num_batches(int(get_torch_len(X_l)), int(self.spec.batch_size))
        steps_u = num_batches(int(get_torch_len(X_u_w)), int(self.spec.batch_size))
        steps_per_epoch = max(int(steps_l), int(steps_u))

        gen_l = torch.Generator().manual_seed(int(seed))
        gen_u = torch.Generator().manual_seed(int(seed) + 1)

        model.train()
        for epoch in range(int(self.spec.max_epochs)):
            iter_l = cycle_batches(
                X_l,
                y_l,
                batch_size=int(self.spec.batch_size),
                generator=gen_l,
                steps=steps_per_epoch,
            )
            iter_u_idx = cycle_batch_indices(
                int(get_torch_len(X_u_w)),
                batch_size=int(self.spec.batch_size),
                generator=gen_u,
                device=get_torch_device(X_u_w),
                steps=steps_per_epoch,
            )
            for step, ((x_lb, y_lb), idx_u) in enumerate(zip(iter_l, iter_u_idx, strict=False)):
                x_uw = slice_data(X_u_w, idx_u)
                x_us = slice_data(X_u_s, idx_u)
                idx_global = idx_u_all[idx_u]

                if bool(self.spec.use_cat):
                    inputs = concat_data([x_lb, x_uw, x_us])
                    logits = extract_logits(model(inputs))
                    if int(logits.ndim) != 2:
                        raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                    num_lb = int(get_torch_len(x_lb))
                    num_u = int(get_torch_len(x_uw))
                    expected = num_lb + num_u + int(get_torch_len(x_us))
                    if int(logits.shape[0]) != expected:
                        raise InductiveValidationError(
                            "Concatenated logits batch size does not match inputs."
                        )
                    logits_l = logits[:num_lb]
                    logits_uw = logits[num_lb : num_lb + num_u]
                    logits_us = logits[num_lb + num_u :]
                else:
                    logits_l = extract_logits(model(x_lb))
                    logits_us = extract_logits(model(x_us))
                    with torch.no_grad():
                        logits_uw = extract_logits(model(x_uw))

                if int(logits_l.ndim) != 2 or int(logits_uw.ndim) != 2 or int(logits_us.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                if logits_uw.shape != logits_us.shape:
                    raise InductiveValidationError("Unlabeled logits shape mismatch.")
                if logits_uw.shape[1] != logits_l.shape[1]:
                    raise InductiveValidationError("Logits must agree on class dimension.")
                if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits_l.shape[1]):
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")

                if self._classwise_acc is None or self._selected_label is None:
                    self._init_state(
                        n_classes=int(logits_l.shape[1]),
                        device=get_torch_device(X_u_w),
                    )

                sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)
                logits_uw_target = (
                    logits_uw.detach() if bool(self.spec.detach_target) else logits_uw
                )
                probs_uw = torch.softmax(logits_uw_target, dim=1)
                max_probs, max_idx = probs_uw.max(dim=1)

                class_acc = self._classwise_acc[max_idx]
                thresh = float(self.spec.p_cutoff) * (class_acc / (2.0 - class_acc))
                mask = (max_probs >= thresh).to(logits_us.dtype)

                pseudo_soft = _sharpen(probs_uw, temperature=float(self.spec.temperature))
                if bool(self.spec.hard_label):
                    pseudo = pseudo_soft.argmax(dim=1)
                    loss_u = torch.nn.functional.cross_entropy(logits_us, pseudo, reduction="none")
                else:
                    log_probs = torch.nn.functional.log_softmax(logits_us, dim=1)
                    loss_u = -(pseudo_soft * log_probs).sum(dim=1)

                denom = mask.sum().clamp_min(1.0)
                unsup_loss = (loss_u * mask).sum() / denom

                if step == 0:
                    mask_mean = float(mask.mean().item()) if int(mask.numel()) else 0.0
                    logger.debug(
                        "FlexMatch epoch=%s p_cutoff=%s thresh_warmup=%s class_acc_mean=%.3f "
                        "thresh_mean=%.3f mask_mean=%.3f",
                        epoch,
                        self.spec.p_cutoff,
                        self.spec.thresh_warmup,
                        float(class_acc.mean().item()),
                        float(thresh.mean().item()),
                        mask_mean,
                    )

                select = max_probs >= float(self.spec.p_cutoff)
                if int(select.sum().item()) > 0:
                    self._selected_label[idx_global[select]] = max_idx[select]
                    self._update_classwise_acc()

                loss = sup_loss + float(self.spec.lambda_u) * unsup_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._bundle = bundle
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> Any:
        if self._bundle is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if backend != "torch":
            raise InductiveValidationError("predict_proba requires torch tensors.")
        torch = optional_import("torch", extra="inductive-torch")

        # Support Dict or Tensor
        if not isinstance(X, torch.Tensor) and not isinstance(X, dict):
            raise InductiveValidationError("predict_proba requires torch.Tensor or dict inputs.")

        model = self._bundle.model
        was_training = model.training
        model.eval()

        # Batched inference
        batch_size = int(self.spec.batch_size)
        from .deep_utils import extract_logits, slice_data

        n_samples = int(X["x"].shape[0]) if isinstance(X, dict) else int(X.shape[0])

        all_logits = []
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                if isinstance(X, dict):
                    idx = torch.arange(start, end, device=X["x"].device)
                    batch_X = slice_data(X, idx)
                else:
                    batch_X = X[start:end]

                logits_batch = extract_logits(model(batch_X))
                if int(logits_batch.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                all_logits.append(logits_batch)

            if not all_logits:
                # Handle empty case
                logits = torch.empty(
                    (0, 0), device=X["x"].device if isinstance(X, dict) else X.device
                )
            else:
                logits = torch.cat(all_logits, dim=0)

            probs = torch.softmax(logits, dim=1)

        if was_training:
            model.train()
        return probs

    def predict(self, X: Any) -> Any:
        proba = self.predict_proba(X)
        return proba.argmax(dim=1)
