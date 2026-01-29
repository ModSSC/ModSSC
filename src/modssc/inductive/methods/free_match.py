from __future__ import annotations

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
class FreeMatchSpec:
    """Specification for FreeMatch (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    lambda_e: float = 0.01
    temperature: float = 0.5
    ema_p: float = 0.999
    use_quantile: bool = True
    clip_thresh: bool = False
    hard_label: bool = True
    use_cat: bool = False
    batch_size: int = 64
    max_epochs: int = 1
    detach_target: bool = True


class FreeMatchMethod(InductiveMethod):
    """FreeMatch with self-adaptive thresholding (torch-only)."""

    info = MethodInfo(
        method_id="free_match",
        name="FreeMatch",
        year=2022,
        family="pseudo-label",
        supports_gpu=True,
        paper_title="FreeMatch: Self-Adaptive Thresholding for Semi-Supervised Learning",
        paper_pdf="https://arxiv.org/abs/2205.07246",
        official_code="",
    )

    def __init__(self, spec: FreeMatchSpec | None = None) -> None:
        self.spec = spec or FreeMatchSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None
        self._p_model: Any | None = None
        self._label_hist: Any | None = None
        self._time_p: Any | None = None

    def _init_state(self, *, n_classes: int, device: Any) -> None:
        torch = optional_import("torch", extra="inductive-torch")
        uniform = torch.full((int(n_classes),), 1.0 / float(n_classes), device=device)
        self._p_model = uniform.clone()
        self._label_hist = uniform.clone()
        self._time_p = self._p_model.mean()

    def _update_state(self, probs_u: Any, max_probs: Any, max_idx: Any) -> None:
        torch = optional_import("torch", extra="inductive-torch")
        if self._p_model is None or self._label_hist is None or self._time_p is None:
            self._init_state(n_classes=int(probs_u.shape[1]), device=probs_u.device)
        assert (
            self._p_model is not None and self._label_hist is not None and self._time_p is not None
        )

        m = float(self.spec.ema_p)
        value = max_probs.quantile(0.8) if bool(self.spec.use_quantile) else max_probs.mean()
        self._time_p = self._time_p * m + (1.0 - m) * value
        if bool(self.spec.clip_thresh):
            self._time_p = self._time_p.clamp(0.0, 0.95)

        self._p_model = self._p_model * m + (1.0 - m) * probs_u.mean(dim=0)
        hist = (
            probs_u.new_zeros((int(self._p_model.shape[0]),))
            if int(max_idx.numel()) == 0
            else torch.bincount(max_idx, minlength=int(self._p_model.shape[0])).to(
                self._p_model.dtype
            )
        )
        if hist.sum() > 0:
            hist = hist / hist.sum()
        self._label_hist = self._label_hist * m + (1.0 - m) * hist

    def _entropy_loss(self, logits_s: Any, label_hist: Any, p_model: Any) -> Any:
        torch = optional_import("torch", extra="inductive-torch")
        prob_s = torch.softmax(logits_s, dim=1)
        pred_s = prob_s.argmax(dim=1)
        hist_s = pred_s.bincount(minlength=int(prob_s.shape[1])).to(prob_s.dtype)
        if hist_s.sum() > 0:
            hist_s = hist_s / hist_s.sum()

        prob_model_scaler = torch.where(
            label_hist > 0, 1.0 / label_hist, torch.zeros_like(label_hist)
        )
        mod_prob_model = p_model * prob_model_scaler
        mod_prob_model = mod_prob_model / mod_prob_model.sum().clamp_min(1e-12)

        mean_prob_s = prob_s.mean(dim=0)
        mean_prob_scaler_s = torch.where(hist_s > 0, 1.0 / hist_s, torch.zeros_like(hist_s))
        mod_mean_prob_s = mean_prob_s * mean_prob_scaler_s
        mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum().clamp_min(1e-12)

        ent = (mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)).sum()
        return ent

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> FreeMatchMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s lambda_e=%s temperature=%s ema_p=%s use_quantile=%s "
            "clip_thresh=%s hard_label=%s use_cat=%s batch_size=%s max_epochs=%s "
            "detach_target=%s has_model_bundle=%s device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.lambda_e,
            self.spec.temperature,
            self.spec.ema_p,
            self.spec.use_quantile,
            self.spec.clip_thresh,
            self.spec.hard_label,
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
            raise InductiveValidationError("FreeMatch requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u_w is None or ds.X_u_s is None:
            raise InductiveValidationError("FreeMatch requires X_u_w and X_u_s.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s = ds.X_u_s
        logger.info(
            "FreeMatch sizes: n_labeled=%s n_unlabeled=%s",
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
            raise InductiveValidationError("model_bundle must be provided for FreeMatch.")
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
        if float(self.spec.lambda_e) < 0:
            raise InductiveValidationError("lambda_e must be >= 0.")
        if float(self.spec.temperature) <= 0:
            raise InductiveValidationError("temperature must be > 0.")
        if not (0.0 <= float(self.spec.ema_p) < 1.0):
            raise InductiveValidationError("ema_p must be in [0, 1).")

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

                sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)

                logits_uw_target = (
                    logits_uw.detach() if bool(self.spec.detach_target) else logits_uw
                )
                probs_u = torch.softmax(logits_uw_target, dim=1)
                max_probs, max_idx = probs_u.max(dim=1)

                self._update_state(probs_u, max_probs, max_idx)
                assert (
                    self._p_model is not None
                    and self._label_hist is not None
                    and self._time_p is not None
                )

                mod = self._p_model / self._p_model.max().clamp_min(1e-12)
                thresh = self._time_p * mod[max_idx]
                mask = (max_probs >= thresh).to(logits_us.dtype)

                pseudo_soft = _sharpen(probs_u, temperature=float(self.spec.temperature))
                if bool(self.spec.hard_label):
                    pseudo = pseudo_soft.argmax(dim=1)
                    loss_u = torch.nn.functional.cross_entropy(logits_us, pseudo, reduction="none")
                else:
                    log_probs = torch.nn.functional.log_softmax(logits_us, dim=1)
                    loss_u = -(pseudo_soft * log_probs).sum(dim=1)

                denom = mask.sum().clamp_min(1.0)
                unsup_loss = (loss_u * mask).sum() / denom

                if float(mask.sum().item()) > 0:
                    logits_s = logits_us[mask.bool()]
                    ent_loss = self._entropy_loss(logits_s, self._label_hist, self._p_model)
                else:
                    ent_loss = torch.zeros((), device=logits_us.device)

                if step == 0 and self._p_model is not None and self._label_hist is not None:
                    mask_mean = float(mask.mean().item()) if int(mask.numel()) else 0.0
                    logger.debug(
                        "FreeMatch epoch=%s time_p=%.3f mask_mean=%.3f "
                        "p_model[min=%.3f max=%.3f] label_hist[min=%.3f max=%.3f]",
                        epoch,
                        float(self._time_p.item()) if self._time_p is not None else 0.0,
                        mask_mean,
                        float(self._p_model.min().item()),
                        float(self._p_model.max().item()),
                        float(self._label_hist.min().item()),
                        float(self._label_hist.max().item()),
                    )

                loss = (
                    sup_loss
                    + float(self.spec.lambda_u) * unsup_loss
                    + float(self.spec.lambda_e) * ent_loss
                )

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
