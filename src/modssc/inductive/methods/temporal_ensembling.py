from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    cycle_batch_indices,
    cycle_batches,
    ensure_float_tensor,
    ensure_model_bundle,
    ensure_model_device,
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


@dataclass(frozen=True)
class TemporalEnsemblingSpec:
    """Specification for Temporal Ensembling (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    unsup_warm_up: float = 0.4
    batch_size: int = 64
    max_epochs: int = 1
    alpha: float = 0.6
    freeze_bn: bool = False
    detach_target: bool = True


class TemporalEnsemblingMethod(InductiveMethod):
    """Temporal ensembling consistency training (torch-only)."""

    info = MethodInfo(
        method_id="temporal_ensembling",
        name="Temporal Ensembling",
        year=2017,
        family="consistency",
        supports_gpu=True,
        paper_title="Temporal Ensembling for Semi-Supervised Learning",
        paper_pdf="https://arxiv.org/abs/1610.02242",
        official_code="",
    )

    def __init__(self, spec: TemporalEnsemblingSpec | None = None) -> None:
        self.spec = spec or TemporalEnsemblingSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> TemporalEnsemblingMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s unsup_warm_up=%s batch_size=%s max_epochs=%s alpha=%s "
            "freeze_bn=%s detach_target=%s has_model_bundle=%s device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.unsup_warm_up,
            self.spec.batch_size,
            self.spec.max_epochs,
            self.spec.alpha,
            self.spec.freeze_bn,
            self.spec.detach_target,
            bool(self.spec.model_bundle),
            device,
            seed,
        )
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        if backend != "torch":
            raise InductiveValidationError(
                "Temporal Ensembling requires torch tensors (torch backend)."
            )

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u_w is None or ds.X_u_s is None:
            raise InductiveValidationError("Temporal Ensembling requires X_u_w and X_u_s.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s = ds.X_u_s
        logger.info(
            "Temporal Ensembling sizes: n_labeled=%s n_unlabeled=%s",
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
            raise InductiveValidationError("model_bundle must be provided for Temporal Ensembling.")

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
        if float(self.spec.unsup_warm_up) < 0:
            raise InductiveValidationError("unsup_warm_up must be >= 0.")
        if not (0.0 <= float(self.spec.alpha) < 1.0):
            raise InductiveValidationError("alpha must be in [0, 1).")

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

        ensemble_predictions = None
        step_idx = 0
        model.train()
        for epoch in range(int(self.spec.max_epochs)):
            if ensemble_predictions is not None:
                epoch_predictions = torch.zeros_like(ensemble_predictions)
                bias = 1.0 - float(self.spec.alpha) ** float(epoch + 1)
                bias = max(bias, 1e-12)
                targets = ensemble_predictions / bias
            else:
                epoch_predictions = None
                targets = None

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
                logits_l = extract_logits(model(x_lb))
                if int(logits_l.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")

                with freeze_batchnorm(model, enabled=bool(self.spec.freeze_bn)):
                    logits_uw = extract_logits(model(x_uw))
                    logits_us = extract_logits(model(x_us))

                if int(logits_uw.ndim) != 2 or int(logits_us.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                if logits_uw.shape != logits_us.shape:
                    raise InductiveValidationError("Unlabeled logits shape mismatch.")
                if logits_uw.shape[1] != logits_l.shape[1]:
                    raise InductiveValidationError("Logits must agree on class dimension.")
                if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits_l.shape[1]):
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")

                prob_uw = torch.softmax(logits_uw, dim=1)
                prob_us = torch.softmax(logits_us, dim=1)
                prob_u = 0.5 * (prob_uw + prob_us)

                if ensemble_predictions is None:
                    ensemble_predictions = torch.zeros(
                        (int(get_torch_len(X_u_w)), int(prob_u.shape[1])),
                        device=get_torch_device(X_u_w),
                        dtype=prob_u.dtype,
                    )
                    epoch_predictions = torch.zeros_like(ensemble_predictions)
                    bias = 1.0 - float(self.spec.alpha) ** float(epoch + 1)
                    bias = max(bias, 1e-12)
                    targets = ensemble_predictions / bias

                if targets is None or epoch_predictions is None:
                    raise InductiveValidationError("Temporal Ensembling targets not initialized.")

                target_u = targets[idx_u]
                if bool(self.spec.detach_target):
                    target_u = target_u.detach()

                sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)
                unsup_loss = torch.mean((prob_u - target_u) ** 2)

                warm = 1.0 if warmup_steps <= 0 else min(float(step_idx) / float(warmup_steps), 1.0)
                loss = sup_loss + float(self.spec.lambda_u) * unsup_loss * float(warm)

                if step == 0:
                    logger.debug(
                        "TemporalEnsembling epoch=%s warm=%.3f sup_loss=%.4f unsup_loss=%.4f "
                        "alpha=%s",
                        epoch,
                        float(warm),
                        float(sup_loss.item()),
                        float(unsup_loss.item()),
                        self.spec.alpha,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_predictions[idx_u] = prob_u.detach()
                step_idx += 1

            if ensemble_predictions is None or epoch_predictions is None:
                raise InductiveValidationError("Temporal Ensembling predictions not initialized.")
            ensemble_predictions = (
                float(self.spec.alpha) * ensemble_predictions
                + (1.0 - float(self.spec.alpha)) * epoch_predictions
            )

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
