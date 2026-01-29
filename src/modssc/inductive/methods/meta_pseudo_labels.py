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
    ensure_float_tensor,
    ensure_model_bundle,
    ensure_model_device,
    extract_logits,
    get_torch_device,
    get_torch_feature_dim,
    get_torch_len,
    get_torch_ndim,
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


def _soft_cross_entropy(target_probs: Any, logits: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    if target_probs.shape != log_probs.shape:
        raise InductiveValidationError("Target distribution shape mismatch.")
    return -(target_probs * log_probs).sum(dim=1)


def _label_smoothed_ce(logits: Any, y: Any, *, smoothing: float) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if float(smoothing) <= 0.0:
        return torch.nn.functional.cross_entropy(logits, y)
    if float(smoothing) >= 1.0:
        raise InductiveValidationError("label_smoothing must be in [0, 1).")
    num_classes = int(logits.shape[1])
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(float(smoothing) / float(num_classes))
        true_dist.scatter_(
            1, y.view(-1, 1), 1.0 - float(smoothing) + float(smoothing) / num_classes
        )
    return -(true_dist * log_probs).sum(dim=1).mean()


def _entropy_loss(logits: Any) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1).mean()


@dataclass(frozen=True)
class MetaPseudoLabelsSpec:
    """Specification for Meta Pseudo Labels (torch-only)."""

    teacher_bundle: TorchModelBundle | None = None
    student_bundle: TorchModelBundle | None = None
    uda_weight: float = 1.0
    uda_threshold: float = 0.8
    uda_temperature: float = 0.4
    label_smoothing: float = 0.1
    mpl_ema: float = 0.99
    mpl_weight: float = 1.0
    batch_size: int = 64
    max_epochs: int = 1
    detach_target: bool = True
    init_teacher_from_student: bool = True


class MetaPseudoLabelsMethod(InductiveMethod):
    """Meta Pseudo Labels with UDA teacher (torch-only)."""

    info = MethodInfo(
        method_id="meta_pseudo_labels",
        name="Meta Pseudo Labels",
        year=2021,
        family="teacher",
        supports_gpu=True,
        paper_title="Meta Pseudo Labels",
        paper_pdf="https://arxiv.org/abs/2003.10580",
        official_code="https://github.com/google-research/google-research/tree/master/meta_pseudo_labels",
    )

    def __init__(self, spec: MetaPseudoLabelsSpec | None = None) -> None:
        self.spec = spec or MetaPseudoLabelsSpec()
        self._teacher_bundle: TorchModelBundle | None = None
        self._student_bundle: TorchModelBundle | None = None
        self._backend: str | None = None

    def _check_models(self, teacher: Any, student: Any) -> None:
        if teacher is student:
            raise InductiveValidationError(
                "teacher_bundle and student_bundle must wrap distinct models."
            )
        t_params = list(teacher.parameters())
        s_params = list(student.parameters())
        ids_t = {id(p) for p in t_params}
        for p in s_params:
            if id(p) in ids_t:
                raise InductiveValidationError(
                    "teacher and student models must not share parameters."
                )

    def _init_teacher(self, teacher: Any, student: Any) -> None:
        try:
            teacher.load_state_dict(student.state_dict(), strict=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise InductiveValidationError(
                "teacher model must share the same architecture as student."
            ) from exc

    def _uda_loss(self, logits_uw: Any, logits_us: Any) -> Any:
        torch = optional_import("torch", extra="inductive-torch")
        if float(self.spec.uda_temperature) <= 0:
            raise InductiveValidationError("uda_temperature must be > 0.")
        probs_uw = torch.softmax(logits_uw / float(self.spec.uda_temperature), dim=1)
        if bool(self.spec.detach_target):
            probs_uw = probs_uw.detach()
        if float(self.spec.uda_threshold) <= 0.0:
            mask = torch.ones_like(probs_uw[:, 0], dtype=logits_us.dtype)
        else:
            mask = (probs_uw.max(dim=1).values >= float(self.spec.uda_threshold)).to(
                logits_us.dtype
            )
        loss_u = _soft_cross_entropy(probs_uw, logits_us)
        if int(mask.numel()) == 0 or float(mask.sum().item()) == 0.0:
            return loss_u.mean() * 0.0, mask
        return (loss_u * mask).sum() / mask.sum(), mask

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> MetaPseudoLabelsMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params uda_weight=%s uda_threshold=%s uda_temperature=%s label_smoothing=%s "
            "mpl_ema=%s mpl_weight=%s batch_size=%s max_epochs=%s detach_target=%s "
            "init_teacher_from_student=%s has_teacher_bundle=%s has_student_bundle=%s "
            "device=%s seed=%s",
            self.spec.uda_weight,
            self.spec.uda_threshold,
            self.spec.uda_temperature,
            self.spec.label_smoothing,
            self.spec.mpl_ema,
            self.spec.mpl_weight,
            self.spec.batch_size,
            self.spec.max_epochs,
            self.spec.detach_target,
            self.spec.init_teacher_from_student,
            bool(self.spec.teacher_bundle),
            bool(self.spec.student_bundle),
            device,
            seed,
        )
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        if backend != "torch":
            raise InductiveValidationError("Meta Pseudo Labels requires torch tensors.")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        if ds.X_u_w is None or ds.X_u_s is None:
            raise InductiveValidationError("Meta Pseudo Labels requires X_u_w and X_u_s.")

        X_l = ds.X_l
        y_l = ensure_1d_labels_torch(ds.y_l, name="y_l")
        X_u_w = ds.X_u_w
        X_u_s = ds.X_u_s

        X_l_s = X_l
        if ds.views:
            for key in ("X_l_s", "X_l_strong", "labeled_strong"):
                cand = ds.views.get(key)
                if cand is not None:
                    X_l_s = cand
                    break

        logger.info(
            "Meta Pseudo Labels sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_u_w)),
        )

        if int(get_torch_len(X_l)) == 0:
            raise InductiveValidationError("X_l must be non-empty.")
        if int(get_torch_len(X_u_w)) == 0 or int(get_torch_len(X_u_s)) == 0:
            raise InductiveValidationError(
                "MetaPseudoLabels requires unlabeled data (X_u). Provided X_u is empty. "
                "Check your data loader or splits configuration."
            )

        if int(get_torch_len(X_u_w)) != int(get_torch_len(X_u_s)):
            raise InductiveValidationError("X_u_w and X_u_s must have the same number of rows.")

        ensure_float_tensor(X_l, name="X_l")
        ensure_float_tensor(X_u_w, name="X_u_w")
        ensure_float_tensor(X_u_s, name="X_u_s")
        ensure_float_tensor(X_l_s, name="X_l_s")

        if int(get_torch_ndim(X_l_s)) != 2:
            raise InductiveValidationError("X_l_s must be 2D.")
        if int(get_torch_len(X_l_s)) != int(get_torch_len(X_l)):
            raise InductiveValidationError("X_l_s must have the same number of rows as X_l.")
        if int(get_torch_feature_dim(X_l_s)) != int(get_torch_feature_dim(X_l)):
            raise InductiveValidationError("X_l_s must have the same feature dimension as X_l.")

        if y_l.dtype != torch.int64:
            raise InductiveValidationError("y_l must be int64 for torch cross entropy.")

        if self.spec.teacher_bundle is None or self.spec.student_bundle is None:
            raise InductiveValidationError(
                "teacher_bundle and student_bundle must be provided for Meta Pseudo Labels."
            )
        teacher_bundle = ensure_model_bundle(self.spec.teacher_bundle)
        student_bundle = ensure_model_bundle(self.spec.student_bundle)
        teacher = teacher_bundle.model
        student = student_bundle.model
        teacher_opt = teacher_bundle.optimizer
        student_opt = student_bundle.optimizer

        ensure_model_device(teacher, device=get_torch_device(X_l))
        ensure_model_device(student, device=get_torch_device(X_l))
        self._check_models(teacher, student)

        if bool(self.spec.init_teacher_from_student):
            self._init_teacher(teacher, student)

        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.max_epochs) <= 0:
            raise InductiveValidationError("max_epochs must be >= 1.")
        if float(self.spec.uda_weight) < 0:
            raise InductiveValidationError("uda_weight must be >= 0.")
        if not (0.0 <= float(self.spec.uda_threshold) <= 1.0):
            raise InductiveValidationError("uda_threshold must be in [0, 1].")
        if float(self.spec.label_smoothing) < 0 or float(self.spec.label_smoothing) >= 1:
            raise InductiveValidationError("label_smoothing must be in [0, 1).")
        if not (0.0 <= float(self.spec.mpl_ema) < 1.0):
            raise InductiveValidationError("mpl_ema must be in [0, 1).")
        if float(self.spec.mpl_weight) < 0:
            raise InductiveValidationError("mpl_weight must be >= 0.")

        steps_l = num_batches(int(get_torch_len(X_l)), int(self.spec.batch_size))
        steps_u = num_batches(int(get_torch_len(X_u_w)), int(self.spec.batch_size))
        steps_per_epoch = max(int(steps_l), int(steps_u))

        gen_l = torch.Generator().manual_seed(int(seed))
        gen_u = torch.Generator().manual_seed(int(seed) + 1)

        dot_product_ma = 0.0

        teacher.train()
        student.train()
        for epoch in range(int(self.spec.max_epochs)):
            iter_l_idx = cycle_batch_indices(
                int(get_torch_len(X_l)),
                batch_size=int(self.spec.batch_size),
                generator=gen_l,
                device=get_torch_device(X_l),
                steps=steps_per_epoch,
            )
            iter_u_idx = cycle_batch_indices(
                int(get_torch_len(X_u_w)),
                batch_size=int(self.spec.batch_size),
                generator=gen_u,
                device=get_torch_device(X_u_w),
                steps=steps_per_epoch,
            )
            for step, (idx_l, idx_u) in enumerate(zip(iter_l_idx, iter_u_idx, strict=False)):
                x_lb = slice_data(X_l, idx_l)
                y_lb = y_l[idx_l]
                x_lb_s = slice_data(X_l_s, idx_l)
                x_uw = slice_data(X_u_w, idx_u)
                x_us = slice_data(X_u_s, idx_u)

                logits_l = extract_logits(teacher(x_lb_s))
                logits_uw = extract_logits(teacher(x_uw))
                logits_us = extract_logits(teacher(x_us))

                if int(logits_l.ndim) != 2 or int(logits_uw.ndim) != 2 or int(logits_us.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                if logits_uw.shape != logits_us.shape:
                    raise InductiveValidationError("Unlabeled logits shape mismatch.")
                if logits_uw.shape[1] != logits_l.shape[1]:
                    raise InductiveValidationError("Logits must agree on class dimension.")
                if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits_l.shape[1]):
                    raise InductiveValidationError("y_l labels must be within [0, n_classes).")

                sup_loss = _label_smoothed_ce(
                    logits_l, y_lb, smoothing=float(self.spec.label_smoothing)
                )
                uda_loss, mask = self._uda_loss(logits_uw, logits_us)
                mpl_loss = _entropy_loss(logits_us)

                with torch.no_grad():
                    logits_s_l_old = extract_logits(student(x_lb))
                    if int(logits_s_l_old.ndim) != 2:
                        raise InductiveValidationError(
                            "Student logits must be 2D (batch, classes)."
                        )
                    if logits_s_l_old.shape[1] != logits_l.shape[1]:
                        raise InductiveValidationError(
                            "Teacher and student must share the same class count."
                        )
                    student_sup_old = torch.nn.functional.cross_entropy(logits_s_l_old, y_lb)

                teacher_probs = torch.softmax(logits_us, dim=1).detach()
                logits_s_u = extract_logits(student(x_us))
                if int(logits_s_u.ndim) != 2:
                    raise InductiveValidationError("Student logits must be 2D (batch, classes).")
                if logits_s_u.shape[1] != logits_l.shape[1]:
                    raise InductiveValidationError(
                        "Teacher and student must share the same class count."
                    )
                student_unsup = _soft_cross_entropy(teacher_probs, logits_s_u).mean()

                student_opt.zero_grad()
                student_unsup.backward()
                student_opt.step()

                with torch.no_grad():
                    logits_s_l_new = extract_logits(student(x_lb))
                    if int(logits_s_l_new.ndim) != 2:
                        raise InductiveValidationError(
                            "Student logits must be 2D (batch, classes)."
                        )
                    if logits_s_l_new.shape[1] != logits_l.shape[1]:
                        raise InductiveValidationError(
                            "Teacher and student must share the same class count."
                        )
                    student_sup_new = torch.nn.functional.cross_entropy(logits_s_l_new, y_lb)
                    dot_product = student_sup_new - student_sup_old

                dot_product_ma = float(self.spec.mpl_ema) * dot_product_ma + (
                    1.0 - float(self.spec.mpl_ema)
                ) * float(dot_product.item())
                dot_product_adj = dot_product - dot_product_ma

                # Meta feedback scales the teacher's unsupervised loss signal.
                teacher_loss = sup_loss
                if float(self.spec.uda_weight) != 0.0:
                    teacher_loss = teacher_loss + float(self.spec.uda_weight) * uda_loss
                if float(self.spec.mpl_weight) != 0.0:
                    teacher_loss = (
                        teacher_loss + float(self.spec.mpl_weight) * mpl_loss * dot_product_adj
                    )

                teacher_opt.zero_grad()
                teacher_loss.backward()
                teacher_opt.step()

                if step == 0:
                    mask_mean = float(mask.mean().item()) if int(mask.numel()) else 0.0
                    logger.debug(
                        "MPL epoch=%s dot=%.4f dot_adj=%.4f sup=%.4f uda=%.4f mpl=%.4f "
                        "student_u=%.4f mask_mean=%.3f",
                        epoch,
                        float(dot_product.item()),
                        float(dot_product_adj.item()),
                        float(sup_loss.item()),
                        float(uda_loss.item()),
                        float(mpl_loss.item()),
                        float(student_unsup.item()),
                        mask_mean,
                    )

        self._teacher_bundle = teacher_bundle
        self._student_bundle = student_bundle
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> Any:
        if self._student_bundle is None:
            raise RuntimeError("MetaPseudoLabelsMethod is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if backend != "torch":
            raise InductiveValidationError(
                "Meta Pseudo Labels predict_proba requires torch tensors."
            )
        torch = optional_import("torch", extra="inductive-torch")
        if not isinstance(X, torch.Tensor) and not (isinstance(X, dict) and "x" in X):
            raise InductiveValidationError("predict_proba requires torch.Tensor or dict inputs.")

        student = self._student_bundle.model
        was_training = student.training
        student.eval()
        with torch.no_grad():
            logits = extract_logits(student(X))
            if int(logits.ndim) != 2:
                raise InductiveValidationError("Model logits must be 2D (batch, classes).")
            proba = torch.softmax(logits, dim=1)
        if was_training:
            student.train()
        return proba

    def predict(self, X: Any) -> Any:
        proba = self.predict_proba(X)
        return proba.argmax(dim=1)
