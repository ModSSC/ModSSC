from __future__ import annotations

import builtins
import logging
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

round = builtins.round


def _update_scores(
    X_u_w: Any,
    model: Any,
    *,
    score: Any,
    batch_size: int,
    p_cutoff: float,
    majority_class: int,
) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if score.ndim != 1:
        raise InductiveValidationError("ADSH score must be a 1D tensor.")
    if score.device != get_torch_device(X_u_w):
        raise InductiveValidationError("ADSH score must be on the same device as X_u_w.")

    n_classes = int(score.shape[0])
    class_lists: list[list[Any]] = [[] for _ in range(n_classes)]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, int(get_torch_len(X_u_w)), int(batch_size)):
            batch = slice_data(X_u_w, slice(start, start + int(batch_size)))
            logits = extract_logits(model(batch))
            if int(logits.ndim) != 2:
                raise InductiveValidationError("Model logits must be 2D (batch, classes).")
            if int(logits.shape[1]) != n_classes:
                raise InductiveValidationError("Model logits class dimension changed during fit().")
            probs = torch.softmax(logits, dim=1)
            max_probs, preds = probs.max(dim=1)
            for cls in range(n_classes):
                cls_mask = preds == cls
                if bool(cls_mask.any()):
                    class_lists[cls].append(max_probs[cls_mask].detach())
    if was_training:
        model.train()

    sorted_lists: list[Any | None] = []
    for cls in range(n_classes):
        if class_lists[cls]:
            vals = torch.cat(class_lists[cls])
            vals, _ = torch.sort(vals, descending=True)
            sorted_lists.append(vals)
        else:
            sorted_lists.append(None)

    rho = 1.0
    majority_vals = sorted_lists[int(majority_class)]
    if majority_vals is not None and int(majority_vals.numel()) > 0:
        count = int((majority_vals >= float(p_cutoff)).sum().item())
        if count > 0:
            rho = count / int(majority_vals.numel())

    for cls in range(n_classes):
        if cls == int(majority_class):
            continue
        vals = sorted_lists[cls]
        if vals is None or int(vals.numel()) == 0:
            continue
        idx = int(round(float(vals.numel()) * float(rho) - 1.0))
        if idx < 0:
            idx = 0
        if idx >= int(vals.numel()):
            idx = int(vals.numel()) - 1
        candidate = torch.minimum(vals[idx], vals.new_tensor(float(p_cutoff)))
        score[cls] = candidate

    score[int(majority_class)] = score.new_tensor(float(p_cutoff))
    return score


@dataclass(frozen=True)
class ADSHSpec:
    """Specification for ADSH (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    p_cutoff: float = 0.95
    use_cat: bool = False
    batch_size: int = 64
    max_epochs: int = 1
    detach_target: bool = True
    score_warmup_epochs: int = 2
    majority_class: int | None = None


class ADSHMethod(InductiveMethod):
    """ADSH adaptive thresholding for class-imbalanced SSL (torch-only)."""

    info = MethodInfo(
        method_id="adsh",
        name="ADSH",
        year=2022,
        family="pseudo-label",
        supports_gpu=True,
        paper_title="Class-Imbalanced Semi-Supervised Learning with Adaptive Thresholding",
        paper_pdf="",
        official_code="http://www.lamda.nju.edu.cn/code_ADSH.ashx",
    )

    def __init__(self, spec: ADSHSpec | None = None) -> None:
        self.spec = spec or ADSHSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None

    def _resolve_majority_class(self, y_l: Any, *, n_classes: int) -> int:
        torch = optional_import("torch", extra="inductive-torch")
        if self.spec.majority_class is not None:
            if not isinstance(self.spec.majority_class, int):
                raise InductiveValidationError("majority_class must be an int or None.")
            if not (0 <= int(self.spec.majority_class) < int(n_classes)):
                raise InductiveValidationError("majority_class must be within [0, n_classes).")
            return int(self.spec.majority_class)
        counts = torch.bincount(y_l, minlength=int(n_classes))
        return int(counts.argmax().item())

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> ADSHMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s p_cutoff=%s use_cat=%s batch_size=%s max_epochs=%s "
            "detach_target=%s score_warmup_epochs=%s majority_class=%s has_model_bundle=%s "
            "device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.p_cutoff,
            self.spec.use_cat,
            self.spec.batch_size,
            self.spec.max_epochs,
            self.spec.detach_target,
            self.spec.score_warmup_epochs,
            self.spec.majority_class,
            bool(self.spec.model_bundle),
            device,
            seed,
        )
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        if backend != "torch":
            raise InductiveValidationError("ADSH requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u_w is None or ds.X_u_s is None:
            raise InductiveValidationError("ADSH requires X_u_w and X_u_s.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s = ds.X_u_s
        logger.info(
            "ADSH sizes: n_labeled=%s n_unlabeled=%s",
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
            raise InductiveValidationError("model_bundle must be provided for ADSH.")
        bundle = ensure_model_bundle(self.spec.model_bundle)
        model = bundle.model
        optimizer = bundle.optimizer
        ensure_model_device(model, device=get_torch_device(X_l))

        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.max_epochs) <= 0:
            raise InductiveValidationError("max_epochs must be >= 1.")
        if int(self.spec.score_warmup_epochs) < 0:
            raise InductiveValidationError("score_warmup_epochs must be >= 0.")
        if float(self.spec.lambda_u) < 0:
            raise InductiveValidationError("lambda_u must be >= 0.")
        if not (0.0 < float(self.spec.p_cutoff) <= 1.0):
            raise InductiveValidationError("p_cutoff must be in (0, 1].")

        was_training = model.training
        model.eval()
        with torch.no_grad():
            init_logits = extract_logits(model(slice_data(X_l, slice(0, 1))))
        if was_training:
            model.train()
        if int(init_logits.ndim) != 2:
            raise InductiveValidationError("Model logits must be 2D (batch, classes).")
        n_classes = int(init_logits.shape[1])
        if y_l.min().item() < 0 or y_l.max().item() >= int(n_classes):
            raise InductiveValidationError("y_l labels must be within [0, n_classes).")

        majority_class = self._resolve_majority_class(y_l, n_classes=n_classes)
        score = torch.full(
            (n_classes,),
            float(self.spec.p_cutoff),
            device=get_torch_device(X_l),
            dtype=init_logits.dtype,
        )

        steps_l = num_batches(int(get_torch_len(X_l)), int(self.spec.batch_size))
        steps_u = num_batches(int(get_torch_len(X_u_w)), int(self.spec.batch_size))
        steps_per_epoch = max(int(steps_l), int(steps_u))

        gen_l = torch.Generator().manual_seed(int(seed))
        gen_u = torch.Generator().manual_seed(int(seed) + 1)

        model.train()
        for epoch in range(int(self.spec.max_epochs)):
            if int(epoch) >= int(self.spec.score_warmup_epochs):
                score = _update_scores(
                    X_u_w,
                    model,
                    score=score,
                    batch_size=int(self.spec.batch_size),
                    p_cutoff=float(self.spec.p_cutoff),
                    majority_class=int(majority_class),
                )

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
                if int(logits_uw.shape[1]) != int(score.shape[0]):
                    raise InductiveValidationError("ADSH score length does not match logits.")
                if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits_l.shape[1]):
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")

                sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)

                probs_uw = torch.softmax(logits_uw, dim=1)
                if bool(self.spec.detach_target):
                    probs_uw = probs_uw.detach()
                score_t = score.to(dtype=probs_uw.dtype, device=probs_uw.device)
                rectify = probs_uw / score_t
                max_rectify, rp_hat = rectify.max(dim=1)
                mask = (max_rectify >= 1.0).to(logits_us.dtype)
                loss_u = torch.nn.functional.cross_entropy(logits_us, rp_hat, reduction="none")
                unsup_loss = (loss_u * mask).mean()

                if step == 0:
                    mask_mean = float(mask.mean().item()) if int(mask.numel()) else 0.0
                    logger.debug(
                        "ADSH epoch=%s p_cutoff=%s mask_mean=%.3f sup_loss=%.4f unsup_loss=%.4f",
                        epoch,
                        self.spec.p_cutoff,
                        mask_mean,
                        float(sup_loss.item()),
                        float(unsup_loss.item()),
                    )

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
            raise RuntimeError("ADSHMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if backend != "torch":
            raise InductiveValidationError("ADSH predict_proba requires torch tensors.")
        torch = optional_import("torch", extra="inductive-torch")
        if not isinstance(X, torch.Tensor) and not isinstance(X, dict):
            raise InductiveValidationError("predict_proba requires torch.Tensor or dict inputs.")

        model = self._bundle.model
        was_training = model.training
        model.eval()

        batch_size = int(self.spec.batch_size)
        n_samples = int(get_torch_len(X))
        all_logits = []

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                if isinstance(X, dict):
                    idx = torch.arange(start, end, device=get_torch_device(X))
                    batch_X = slice_data(X, idx)
                else:
                    batch_X = X[start:end]
                logits_batch = extract_logits(model(batch_X))

                if int(logits_batch.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                all_logits.append(logits_batch)

            if not all_logits:
                logits = torch.empty((0, 0), device=get_torch_device(X))
            else:
                logits = torch.cat(all_logits, dim=0)
            proba = torch.softmax(logits, dim=1)

        if was_training:
            model.train()
        return proba

    def predict(self, X: Any) -> Any:
        proba = self.predict_proba(X)
        return proba.argmax(dim=1)
