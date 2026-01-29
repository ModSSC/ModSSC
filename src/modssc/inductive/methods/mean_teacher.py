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
class MeanTeacherSpec:
    """Specification for Mean Teacher (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    unsup_warm_up: float = 0.4
    batch_size: int = 64
    max_epochs: int = 1
    ema_decay: float = 0.999
    freeze_bn: bool = True
    detach_target: bool = True


class MeanTeacherMethod(InductiveMethod):
    """Mean Teacher consistency training (torch-only)."""

    info = MethodInfo(
        method_id="mean_teacher",
        name="Mean Teacher",
        year=2017,
        family="consistency",
        supports_gpu=True,
        paper_title="Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results",
        paper_pdf="https://arxiv.org/abs/1703.01780",
        official_code="",
    )

    def __init__(self, spec: MeanTeacherSpec | None = None) -> None:
        self.spec = spec or MeanTeacherSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None

    def _check_teacher(self, student: Any, teacher: Any) -> None:
        if teacher is student:
            raise InductiveValidationError("ema_model must be distinct from model.")
        s_params = list(student.parameters())
        t_params = list(teacher.parameters())
        if len(s_params) != len(t_params):
            raise InductiveValidationError("ema_model must match model parameter count.")
        for sp, tp in zip(s_params, t_params, strict=True):
            if sp.shape != tp.shape:
                raise InductiveValidationError("ema_model parameter shapes must match model.")
            if sp.device != tp.device:
                raise InductiveValidationError("ema_model must be on the same device as model.")

    def _init_teacher(self, student: Any, teacher: Any) -> None:
        try:
            teacher.load_state_dict(student.state_dict(), strict=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise InductiveValidationError(
                "ema_model must be initialized with the same architecture as model."
            ) from exc

    def _update_teacher(self, student: Any, teacher: Any, *, decay: float) -> None:
        for t_param, s_param in zip(teacher.parameters(), student.parameters(), strict=True):
            t_param.data.mul_(float(decay)).add_(s_param.data, alpha=1.0 - float(decay))

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> MeanTeacherMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s unsup_warm_up=%s batch_size=%s max_epochs=%s ema_decay=%s "
            "freeze_bn=%s detach_target=%s has_model_bundle=%s device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.unsup_warm_up,
            self.spec.batch_size,
            self.spec.max_epochs,
            self.spec.ema_decay,
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
            raise InductiveValidationError("Mean Teacher requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u_w is None or ds.X_u_s is None:
            raise InductiveValidationError("Mean Teacher requires X_u_w and X_u_s.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s = ds.X_u_s
        logger.info(
            "Mean Teacher sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_u_w)),
        )

        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        if int(get_torch_len(X_u_w)) == 0 or int(get_torch_len(X_u_s)) == 0:
            raise InductiveValidationError("MeanTeacher requires unlabeled data. X_u is empty.")

        ensure_float_tensor(X_l, name="X_l")
        ensure_float_tensor(X_u_w, name="X_u_w")
        ensure_float_tensor(X_u_s, name="X_u_s")

        if y_l.dtype != torch.int64:
            raise InductiveValidationError("y_l must be int64 for torch cross entropy.")

        if self.spec.model_bundle is None:
            raise InductiveValidationError("model_bundle must be provided for Mean Teacher.")
        bundle = ensure_model_bundle(self.spec.model_bundle)
        if bundle.ema_model is None:
            raise InductiveValidationError(
                "model_bundle.ema_model must be provided for Mean Teacher."
            )

        student = bundle.model
        teacher = bundle.ema_model
        optimizer = bundle.optimizer

        ensure_model_device(student, device=get_torch_device(X_l))
        ensure_model_device(teacher, device=get_torch_device(X_l))
        self._check_teacher(student, teacher)

        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.max_epochs) <= 0:
            raise InductiveValidationError("max_epochs must be >= 1.")
        if float(self.spec.lambda_u) < 0:
            raise InductiveValidationError("lambda_u must be >= 0.")
        if float(self.spec.unsup_warm_up) < 0:
            raise InductiveValidationError("unsup_warm_up must be >= 0.")
        if not (0.0 <= float(self.spec.ema_decay) <= 1.0):
            raise InductiveValidationError("ema_decay must be in [0, 1].")

        self._init_teacher(student, teacher)

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
        student.train()
        teacher.train()
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
                logits_l = extract_logits(student(x_lb))
                if int(logits_l.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")

                with torch.no_grad(), freeze_batchnorm(teacher, enabled=bool(self.spec.freeze_bn)):
                    logits_uw = extract_logits(teacher(slice_data(X_u_w, idx_u)))

                with freeze_batchnorm(student, enabled=bool(self.spec.freeze_bn)):
                    logits_us = extract_logits(student(slice_data(X_u_s, idx_u)))

                if logits_uw.shape != logits_us.shape:
                    raise InductiveValidationError("Unlabeled logits shape mismatch.")
                if logits_uw.shape[1] != logits_l.shape[1]:
                    raise InductiveValidationError("Logits must agree on class dimension.")
                if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits_l.shape[1]):
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")

                sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)
                prob_uw = torch.softmax(logits_uw, dim=1)
                prob_us = torch.softmax(logits_us, dim=1)
                if bool(self.spec.detach_target):
                    prob_uw = prob_uw.detach()
                unsup_loss = torch.mean((prob_us - prob_uw) ** 2)

                warm = 1.0 if warmup_steps <= 0 else min(float(step_idx) / float(warmup_steps), 1.0)
                loss = sup_loss + float(self.spec.lambda_u) * unsup_loss * float(warm)

                if step == 0:
                    logger.debug(
                        "MeanTeacher epoch=%s warm=%.3f sup_loss=%.4f unsup_loss=%.4f ema_decay=%s",
                        epoch,
                        float(warm),
                        float(sup_loss.item()),
                        float(unsup_loss.item()),
                        self.spec.ema_decay,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self._update_teacher(student, teacher, decay=float(self.spec.ema_decay))

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
