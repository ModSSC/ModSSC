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
    extract_features,
    extract_logits,
    freeze_batchnorm,
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


def _mixup(
    X: Any,
    y: Any,
    *,
    alpha: float,
    generator: Any,
):
    torch = optional_import("torch", extra="inductive-torch")
    if not isinstance(y, torch.Tensor):
        raise InductiveValidationError("mixup expects torch.Tensor labels.")
    is_dict = isinstance(X, dict)
    base = X["x"] if is_dict and "x" in X else X
    if not isinstance(base, torch.Tensor):
        raise InductiveValidationError("mixup expects torch.Tensor inputs.")
    if int(base.shape[0]) != int(y.shape[0]):
        raise InductiveValidationError("mixup X and y must have the same first dimension.")
    batch = int(base.shape[0])
    if batch == 0:
        raise InductiveValidationError("mixup requires non-empty batch.")
    if alpha <= 0:
        lam = torch.ones((batch,), device=base.device, dtype=base.dtype)
    else:
        dist = torch.distributions.Beta(float(alpha), float(alpha))
        lam = dist.sample((batch,)).to(device=base.device, dtype=base.dtype)
        lam = torch.max(lam, 1.0 - lam)

    perm = torch.randperm(batch, generator=generator, device="cpu")
    if base.device.type != "cpu":
        perm = perm.to(device=base.device)
    X2 = base[perm]
    y2 = y[perm]

    view = [batch] + [1] * (int(base.dim()) - 1)
    lam_x = lam.view(*view)
    mixed_x = lam_x * base + (1.0 - lam_x) * X2
    mixed_y = lam.view(batch, 1) * y + (1.0 - lam.view(batch, 1)) * y2
    if is_dict:
        out = dict(X)
        out["x"] = mixed_x
        return out, mixed_y
    return mixed_x, mixed_y


def _forward_head(bundle: TorchModelBundle, *, features: Any) -> Any:
    meta = bundle.meta or {}
    head = None
    if isinstance(meta, Mapping):
        head = meta.get("forward_head") or meta.get("head")
    if callable(head):
        return head(features)
    model = bundle.model
    try:
        return model(features, only_fc=True)
    except TypeError as exc:
        raise InductiveValidationError(
            "mixup_manifold requires bundle.meta['forward_head'] (callable) or "
            "a model that accepts only_fc=True."
        ) from exc


@dataclass(frozen=True)
class MixMatchSpec:
    """Specification for MixMatch (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    temperature: float = 0.5
    mixup_alpha: float = 0.5
    unsup_warm_up: float = 0.4
    mixup_manifold: bool = False
    freeze_bn: bool = False
    batch_size: int = 64
    max_epochs: int = 1


class MixMatchMethod(InductiveMethod):
    """MixMatch consistency with MixUp (torch-only, uses two augmentations)."""

    info = MethodInfo(
        method_id="mixmatch",
        name="MixMatch",
        year=2019,
        family="mixup",
        supports_gpu=True,
        paper_title="MixMatch: A Holistic Approach to Semi-Supervised Learning",
        paper_pdf="https://arxiv.org/abs/1905.02249",
        official_code="",
    )

    def __init__(self, spec: MixMatchSpec | None = None) -> None:
        self.spec = spec or MixMatchSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> MixMatchMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s temperature=%s mixup_alpha=%s unsup_warm_up=%s mixup_manifold=%s "
            "freeze_bn=%s batch_size=%s max_epochs=%s has_model_bundle=%s device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.temperature,
            self.spec.mixup_alpha,
            self.spec.unsup_warm_up,
            self.spec.mixup_manifold,
            self.spec.freeze_bn,
            self.spec.batch_size,
            self.spec.max_epochs,
            bool(self.spec.model_bundle),
            device,
            seed,
        )
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        if backend != "torch":
            raise InductiveValidationError("MixMatch requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u_w is None or ds.X_u_s is None:
            raise InductiveValidationError("MixMatch requires X_u_w and X_u_s.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s = ds.X_u_s
        logger.info(
            "MixMatch sizes: n_labeled=%s n_unlabeled=%s",
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
            raise InductiveValidationError("model_bundle must be provided for MixMatch.")
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
        if float(self.spec.temperature) <= 0:
            raise InductiveValidationError("temperature must be > 0.")
        if float(self.spec.mixup_alpha) < 0:
            raise InductiveValidationError("mixup_alpha must be >= 0.")
        if float(self.spec.unsup_warm_up) < 0:
            raise InductiveValidationError("unsup_warm_up must be >= 0.")

        steps_l = num_batches(int(get_torch_len(X_l)), int(self.spec.batch_size))
        steps_u = num_batches(int(get_torch_len(X_u_w)), int(self.spec.batch_size))
        steps_per_epoch = max(int(steps_l), int(steps_u))
        total_steps = int(self.spec.max_epochs) * steps_per_epoch
        if float(self.spec.unsup_warm_up) <= 0:
            warmup_steps = 0
        else:
            warmup_steps = int(max(1, round(float(self.spec.unsup_warm_up) * total_steps)))

        gen_l = torch.Generator().manual_seed(int(seed))
        gen_u = torch.Generator().manual_seed(int(seed) + 1)

        step_idx = 0
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

                with torch.no_grad(), freeze_batchnorm(model, enabled=bool(self.spec.freeze_bn)):
                    logits_uw = extract_logits(model(x_uw))
                    logits_us = extract_logits(model(x_us))

                if logits_uw.shape != logits_us.shape:
                    raise InductiveValidationError("Unlabeled logits shape mismatch.")
                if int(logits_uw.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")

                probs_u = (torch.softmax(logits_uw, dim=1) + torch.softmax(logits_us, dim=1)) / 2.0
                pseudo_u = _sharpen(probs_u, temperature=float(self.spec.temperature)).detach()

                n_classes = int(pseudo_u.shape[1])
                if y_lb.min().item() < 0 or y_lb.max().item() >= n_classes:
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")
                y_lb_onehot = torch.nn.functional.one_hot(y_lb, num_classes=n_classes).to(
                    pseudo_u.dtype
                )
                targets = torch.cat([y_lb_onehot, pseudo_u, pseudo_u], dim=0)

                if bool(self.spec.mixup_manifold):
                    out_lb = model(x_lb)
                    out_uw = model(x_uw)
                    out_us = model(x_us)
                    feat_lb = extract_features(out_lb)
                    feat_uw = extract_features(out_uw)
                    feat_us = extract_features(out_us)
                    inputs = torch.cat([feat_lb, feat_uw, feat_us], dim=0)
                    mixed_x, mixed_y = _mixup(
                        inputs, targets, alpha=float(self.spec.mixup_alpha), generator=gen_l
                    )
                    logits_all = extract_logits(_forward_head(bundle, features=mixed_x))
                else:
                    inputs = concat_data([x_lb, x_uw, x_us])
                    mixed_x, mixed_y = _mixup(
                        inputs, targets, alpha=float(self.spec.mixup_alpha), generator=gen_l
                    )
                    logits_all = extract_logits(model(mixed_x))

                if int(logits_all.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")

                num_lb = int(get_torch_len(x_lb))
                logits_l = logits_all[:num_lb]
                logits_u = logits_all[num_lb:]

                log_probs_l = torch.nn.functional.log_softmax(logits_l, dim=1)
                sup_loss = -(mixed_y[:num_lb] * log_probs_l).sum(dim=1).mean()

                probs_u_mixed = torch.softmax(logits_u, dim=1)
                unsup_loss = ((probs_u_mixed - mixed_y[num_lb:]) ** 2).mean()

                warm = 1.0 if warmup_steps <= 0 else min(float(step_idx) / float(warmup_steps), 1.0)
                loss = sup_loss + float(self.spec.lambda_u) * unsup_loss * float(warm)

                if step == 0:
                    logger.debug(
                        "MixMatch epoch=%s warm=%.3f sup_loss=%.4f unsup_loss=%.4f",
                        epoch,
                        float(warm),
                        float(sup_loss.item()),
                        float(unsup_loss.item()),
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_idx += 1

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
