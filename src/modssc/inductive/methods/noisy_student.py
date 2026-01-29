from __future__ import annotations

import copy
import logging
from contextlib import nullcontext
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


def _kl_divergence(probs: Any, log_probs: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    return (probs * (torch.log(probs + 1e-12) - log_probs)).sum(dim=1)


@dataclass(frozen=True)
class NoisyStudentSpec:
    """Specification for Noisy Student (torch-only)."""

    model_bundle: TorchModelBundle | None = None
    lambda_u: float = 1.0
    p_cutoff: float = 0.95
    temperature: float = 0.5
    hard_label: bool = True
    use_cat: bool = False
    batch_size: int = 64
    max_epochs: int = 1
    teacher_epochs: int = 1
    unsup_warm_up: float = 0.0
    freeze_bn: bool = True
    detach_target: bool = True


class NoisyStudentMethod(InductiveMethod):
    """Noisy Student training (torch-only)."""

    info = MethodInfo(
        method_id="noisy_student",
        name="Noisy Student",
        year=2020,
        family="pseudo-label",
        supports_gpu=True,
        paper_title="Noisy Student improves ImageNet classification",
        paper_pdf="https://arxiv.org/abs/1911.04252",
        official_code="",
    )

    def __init__(self, spec: NoisyStudentSpec | None = None) -> None:
        self.spec = spec or NoisyStudentSpec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None

    def _check_teacher(self, student: Any, teacher: Any) -> None:
        if teacher is student:
            raise InductiveValidationError("teacher model must be distinct from student model.")
        s_params = list(student.parameters())
        t_params = list(teacher.parameters())
        if len(s_params) != len(t_params):
            raise InductiveValidationError("teacher model must match student parameter count.")
        for sp, tp in zip(s_params, t_params, strict=True):
            if sp.shape != tp.shape:
                raise InductiveValidationError("teacher model parameter shapes must match.")
            if sp.device != tp.device:
                raise InductiveValidationError("teacher model must be on the same device.")

    def _init_teacher(self, student: Any, teacher: Any) -> None:
        try:
            teacher.load_state_dict(student.state_dict(), strict=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise InductiveValidationError(
                "teacher model must share the same architecture as the student."
            ) from exc

    def _train_teacher(
        self,
        student: Any,
        optimizer: Any,
        X_l: Any,
        y_l: Any,
        *,
        batch_size: int,
        epochs: int,
        seed: int,
    ) -> None:
        if int(epochs) <= 0:
            return
        torch = optional_import("torch", extra="inductive-torch")
        steps_l = num_batches(int(get_torch_len(X_l)), int(batch_size))
        gen_l = torch.Generator().manual_seed(int(seed))
        student.train()
        for epoch in range(int(epochs)):
            iter_l = cycle_batches(
                X_l,
                y_l,
                batch_size=int(batch_size),
                generator=gen_l,
                steps=steps_l,
            )
            for step, (x_lb, y_lb) in enumerate(iter_l):
                logits_l = extract_logits(student(x_lb))
                if int(logits_l.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits_l.shape[1]):
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")
                sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)
                if step == 0:
                    logger.debug(
                        "NoisyStudent teacher epoch=%s sup_loss=%.4f",
                        epoch,
                        float(sup_loss.item()),
                    )
                optimizer.zero_grad()
                sup_loss.backward()
                optimizer.step()

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> NoisyStudentMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params lambda_u=%s p_cutoff=%s temperature=%s hard_label=%s use_cat=%s "
            "batch_size=%s max_epochs=%s teacher_epochs=%s unsup_warm_up=%s freeze_bn=%s "
            "detach_target=%s has_model_bundle=%s device=%s seed=%s",
            self.spec.lambda_u,
            self.spec.p_cutoff,
            self.spec.temperature,
            self.spec.hard_label,
            self.spec.use_cat,
            self.spec.batch_size,
            self.spec.max_epochs,
            self.spec.teacher_epochs,
            self.spec.unsup_warm_up,
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
            raise InductiveValidationError("Noisy Student requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        X_uw = ds.X_u_w if ds.X_u_w is not None else ds.X_u
        X_us = ds.X_u_s if ds.X_u_s is not None else X_uw
        if X_uw is None or X_us is None:
            raise InductiveValidationError("Noisy Student requires unlabeled data (X_u or X_u_*).")
        if int(get_torch_len(X_uw)) != int(get_torch_len(X_us)):
            raise InductiveValidationError("X_u_w and X_u_s must have the same number of rows.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        logger.info(
            "Noisy Student sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_uw)),
        )

        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        if int(get_torch_len(X_uw)) == 0:
            raise InductiveValidationError("X_u must be non-empty.")

        ensure_float_tensor(X_l, name="X_l")
        ensure_float_tensor(X_uw, name="X_u_w")
        ensure_float_tensor(X_us, name="X_u_s")

        if y_l.dtype != torch.int64:
            raise InductiveValidationError("y_l must be int64 for torch cross entropy.")

        if self.spec.model_bundle is None:
            raise InductiveValidationError("model_bundle must be provided for Noisy Student.")

        bundle = ensure_model_bundle(self.spec.model_bundle)
        student = bundle.model
        optimizer = bundle.optimizer
        ensure_model_device(student, device=get_torch_device(X_l))

        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.max_epochs) <= 0:
            raise InductiveValidationError("max_epochs must be >= 1.")
        if int(self.spec.teacher_epochs) < 0:
            raise InductiveValidationError("teacher_epochs must be >= 0.")
        if float(self.spec.lambda_u) < 0:
            raise InductiveValidationError("lambda_u must be >= 0.")
        if not (0.0 <= float(self.spec.p_cutoff) <= 1.0):
            raise InductiveValidationError("p_cutoff must be in [0, 1].")
        if float(self.spec.temperature) <= 0:
            raise InductiveValidationError("temperature must be > 0.")
        if float(self.spec.unsup_warm_up) < 0:
            raise InductiveValidationError("unsup_warm_up must be >= 0.")

        self._train_teacher(
            student,
            optimizer,
            X_l,
            y_l,
            batch_size=int(self.spec.batch_size),
            epochs=int(self.spec.teacher_epochs),
            seed=int(seed),
        )

        teacher = bundle.ema_model or copy.deepcopy(student)
        self._check_teacher(student, teacher)
        self._init_teacher(student, teacher)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()

        steps_l = num_batches(int(get_torch_len(X_l)), int(self.spec.batch_size))
        steps_u = num_batches(int(get_torch_len(X_uw)), int(self.spec.batch_size))
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
        for epoch in range(int(self.spec.max_epochs)):
            iter_l = cycle_batches(
                X_l,
                y_l,
                batch_size=int(self.spec.batch_size),
                generator=gen_l,
                steps=steps_per_epoch,
            )
            iter_u_idx = cycle_batch_indices(
                int(get_torch_len(X_uw)),
                batch_size=int(self.spec.batch_size),
                generator=gen_u,
                device=get_torch_device(X_uw),
                steps=steps_per_epoch,
            )
            for step, ((x_lb, y_lb), idx_u) in enumerate(zip(iter_l, iter_u_idx, strict=False)):
                x_uw = slice_data(X_uw, idx_u)
                x_us = slice_data(X_us, idx_u)

                if bool(self.spec.use_cat) and not bool(self.spec.freeze_bn):
                    inputs = concat_data([x_lb, x_us])
                    logits = extract_logits(student(inputs))
                    if int(logits.ndim) != 2:
                        raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                    num_lb = int(get_torch_len(x_lb))
                    expected = num_lb + int(get_torch_len(x_us))
                    if int(logits.shape[0]) != expected:
                        raise InductiveValidationError(
                            "Concatenated logits batch size does not match inputs."
                        )
                    logits_l = logits[:num_lb]
                    logits_us = logits[num_lb:]
                else:
                    logits_l = extract_logits(student(x_lb))
                    with freeze_batchnorm(student, enabled=bool(self.spec.freeze_bn)):
                        logits_us = extract_logits(student(x_us))
                if int(logits_l.ndim) != 2 or int(logits_us.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                if int(logits_l.shape[0]) != int(get_torch_len(x_lb)):
                    raise InductiveValidationError(
                        "Labeled logits batch size does not match inputs."
                    )
                if int(logits_us.shape[0]) != int(get_torch_len(x_us)):
                    raise InductiveValidationError(
                        "Unlabeled logits batch size does not match inputs."
                    )

                if logits_l.shape[1] != logits_us.shape[1]:
                    raise InductiveValidationError("Logits must agree on class dimension.")
                if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits_l.shape[1]):
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")

                sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)

                teacher_ctx = torch.no_grad() if bool(self.spec.detach_target) else nullcontext()
                with teacher_ctx, freeze_batchnorm(teacher, enabled=bool(self.spec.freeze_bn)):
                    logits_uw = extract_logits(teacher(x_uw))
                if int(logits_uw.ndim) != 2:
                    raise InductiveValidationError("Teacher logits must be 2D (batch, classes).")
                if logits_uw.shape[1] != logits_us.shape[1]:
                    raise InductiveValidationError("Teacher logits must match class dimension.")

                probs_uw = torch.softmax(logits_uw / float(self.spec.temperature), dim=1)
                conf = probs_uw.max(dim=1).values
                mask = (conf >= float(self.spec.p_cutoff)).to(logits_us.dtype)

                if bool(self.spec.hard_label):
                    pseudo = probs_uw.argmax(dim=1)
                    ce = torch.nn.functional.cross_entropy(logits_us, pseudo, reduction="none")
                    if int(mask.numel()) == 0 or float(mask.sum().item()) == 0.0:
                        unsup_loss = ce.mean() * 0.0
                    else:
                        unsup_loss = (ce * mask).sum() / mask.sum()
                else:
                    log_probs_us = torch.nn.functional.log_softmax(logits_us, dim=1)
                    kl = _kl_divergence(probs_uw, log_probs_us)
                    if int(mask.numel()) == 0 or float(mask.sum().item()) == 0.0:
                        unsup_loss = kl.mean() * 0.0
                    else:
                        unsup_loss = (kl * mask).sum() / mask.sum()

                warm = 1.0 if warmup_steps <= 0 else min(float(step_idx) / float(warmup_steps), 1.0)
                loss = sup_loss + float(self.spec.lambda_u) * unsup_loss * float(warm)

                if step == 0:
                    mask_mean = float(mask.mean().item()) if int(mask.numel()) else 0.0
                    logger.debug(
                        "NoisyStudent epoch=%s warm=%.3f mask_mean=%.3f sup_loss=%.4f "
                        "unsup_loss=%.4f",
                        epoch,
                        float(warm),
                        mask_mean,
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
