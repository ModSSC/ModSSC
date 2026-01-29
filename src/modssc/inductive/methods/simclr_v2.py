from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
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


def _as_tensor(value: Any, *, name: str) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if not isinstance(value, torch.Tensor):
        raise InductiveValidationError(f"{name} must be a torch.Tensor.")
    return value


def _tensor_from_output(out: Any, *, keys: tuple[str, ...], name: str) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, Mapping):
        for key in keys:
            if key in out:
                return _as_tensor(out[key], name=f"{name}[{key}]")
    if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise InductiveValidationError(
        f"{name} must be a torch.Tensor, tuple[0], or mapping with keys {keys}."
    )


def _forward_features(model: Any, meta: Mapping[str, Any] | None, X: Any) -> Any:
    if isinstance(meta, Mapping):
        forward = (
            meta.get("forward_features") or meta.get("feature_extractor") or meta.get("encoder")
        )
        if callable(forward):
            return _as_tensor(forward(X), name="forward_features output")
    out = model(X)
    return _tensor_from_output(
        out,
        keys=("feat", "features", "embedding", "proj", "projection", "z", "logits"),
        name="model output",
    )


def _forward_projection(model: Any, meta: Mapping[str, Any] | None, X: Any) -> Any:
    if isinstance(meta, Mapping):
        forward = meta.get("forward_projection")
        if callable(forward):
            return _as_tensor(forward(X), name="forward_projection output")
        projector = meta.get("projection_head") or meta.get("projector")
        if callable(projector):
            feats = _forward_features(model, meta, X)
            return _as_tensor(projector(feats), name="projection_head output")
    out = model(X)
    return _tensor_from_output(
        out,
        keys=("proj", "projection", "z", "embedding", "feat", "features", "logits"),
        name="model output",
    )


def _forward_logits(model: Any, meta: Mapping[str, Any] | None, X: Any) -> Any:
    if isinstance(meta, Mapping):
        forward = (
            meta.get("forward_logits") or meta.get("forward_classifier") or meta.get("classifier")
        )
        if callable(forward):
            return extract_logits(forward(X))
        head = meta.get("forward_head") or meta.get("head")
        if callable(head):
            has_features = (
                meta.get("forward_features") or meta.get("feature_extractor") or meta.get("encoder")
            )
            if callable(has_features):
                feats = _forward_features(model, meta, X)
                return extract_logits(head(feats))
            return extract_logits(head(X))
    return extract_logits(model(X))


def _rebind_meta(
    meta: Mapping[str, Any] | None, *, source: Any, target: Any
) -> Mapping[str, Any] | None:
    if meta is None or not isinstance(meta, Mapping):
        return meta
    rebound: dict[str, Any] = {}
    for key, value in meta.items():
        if callable(value):
            bound_self = getattr(value, "__self__", None)
            name = getattr(value, "__name__", None)
            if bound_self is source and name and hasattr(target, name):
                candidate = getattr(target, name)
                if callable(candidate):
                    rebound[key] = candidate
                    continue
        rebound[key] = value
    return rebound


def _nt_xent_loss(z: Any, *, temperature: float) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if float(temperature) <= 0:
        raise InductiveValidationError("temperature must be > 0.")
    if int(z.ndim) != 2:
        raise InductiveValidationError("Projection outputs must be 2D (batch, dim).")
    n = int(z.shape[0])
    if n < 2 or n % 2 != 0:
        raise InductiveValidationError("Contrastive batch must be even and >= 2.")
    z = torch.nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / float(temperature)
    mask = torch.eye(n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    n_pairs = n // 2
    pos = torch.arange(n_pairs, device=z.device)
    pos_idx = torch.cat([pos + n_pairs, pos], dim=0)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -log_prob[torch.arange(n, device=z.device), pos_idx].mean()


def _distill_loss(
    logits_s: Any,
    logits_t: Any,
    *,
    temperature: float,
    detach_target: bool,
) -> Any:
    torch = optional_import("torch", extra="inductive-torch")
    if float(temperature) <= 0:
        raise InductiveValidationError("distill_temperature must be > 0.")
    log_probs_s = torch.nn.functional.log_softmax(logits_s / float(temperature), dim=1)
    probs_t = torch.softmax(logits_t / float(temperature), dim=1)
    if detach_target:
        probs_t = probs_t.detach()
    return -(probs_t * log_probs_s).sum(dim=1).mean()


def _check_distill_models(student: Any, teacher: Any) -> None:
    if teacher is student:
        raise InductiveValidationError("teacher and student models must be distinct.")
    params_s = list(student.parameters())
    params_t = list(teacher.parameters())
    ids_s = {id(p) for p in params_s}
    for p in params_t:
        if id(p) in ids_s:
            raise InductiveValidationError("teacher and student must not share parameters.")


@dataclass(frozen=True)
class SimCLRv2Spec:
    """Specification for SimCLRv2 (torch-only)."""

    pretrain_bundle: TorchModelBundle | None = None
    finetune_bundle: TorchModelBundle | None = None
    student_bundle: TorchModelBundle | None = None
    temperature: float = 0.5
    distill_temperature: float = 1.0
    alpha: float = 0.5
    batch_size: int = 64
    pretrain_epochs: int = 1
    finetune_epochs: int = 1
    distill_epochs: int = 1
    transfer_pretrain: bool = True
    use_labeled_in_distill: bool = True
    freeze_bn: bool = True
    detach_target: bool = True


class SimCLRv2Method(InductiveMethod):
    """SimCLRv2 pretrain -> fine-tune -> distill (torch-only)."""

    info = MethodInfo(
        method_id="simclr_v2",
        name="SimCLRv2",
        year=2020,
        family="contrastive",
        supports_gpu=True,
        paper_title="Big Self-Supervised Models are Strong Semi-Supervised Learners",
        paper_pdf="https://arxiv.org/abs/2006.10029",
        official_code="https://github.com/google-research/simclr",
    )

    def __init__(self, spec: SimCLRv2Spec | None = None) -> None:
        self.spec = spec or SimCLRv2Spec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None

    def fit(self, data: Any, *, device: DeviceSpec, seed: int = 0) -> SimCLRv2Method:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "params temperature=%s distill_temperature=%s alpha=%s batch_size=%s "
            "pretrain_epochs=%s finetune_epochs=%s distill_epochs=%s transfer_pretrain=%s "
            "use_labeled_in_distill=%s freeze_bn=%s detach_target=%s "
            "has_pretrain_bundle=%s has_finetune_bundle=%s has_student_bundle=%s "
            "device=%s seed=%s",
            self.spec.temperature,
            self.spec.distill_temperature,
            self.spec.alpha,
            self.spec.batch_size,
            self.spec.pretrain_epochs,
            self.spec.finetune_epochs,
            self.spec.distill_epochs,
            self.spec.transfer_pretrain,
            self.spec.use_labeled_in_distill,
            self.spec.freeze_bn,
            self.spec.detach_target,
            bool(self.spec.pretrain_bundle),
            bool(self.spec.finetune_bundle),
            bool(self.spec.student_bundle),
            device,
            seed,
        )
        if data is None:
            raise InductiveValidationError("data must not be None.")

        backend = detect_backend(data.X_l)
        if backend != "torch":
            raise InductiveValidationError("SimCLRv2 requires torch tensors (torch backend).")

        ds = ensure_torch_data(data, device=device)
        torch = optional_import("torch", extra="inductive-torch")

        X_l = ds.X_l
        y_l = ds.y_l
        X_uw = ds.X_u_w if ds.X_u_w is not None else ds.X_u
        X_us = ds.X_u_s

        logger.info(
            "SimCLRv2 sizes: n_labeled=%s n_unlabeled=%s",
            int(get_torch_len(X_l)),
            int(get_torch_len(X_uw)) if X_uw is not None else 0,
        )

        if int(self.spec.batch_size) <= 0:
            raise InductiveValidationError("batch_size must be >= 1.")
        if int(self.spec.pretrain_epochs) < 0:
            raise InductiveValidationError("pretrain_epochs must be >= 0.")
        if int(self.spec.finetune_epochs) < 0:
            raise InductiveValidationError("finetune_epochs must be >= 0.")
        if int(self.spec.distill_epochs) < 0:
            raise InductiveValidationError("distill_epochs must be >= 0.")
        if (
            int(self.spec.pretrain_epochs) == 0
            and int(self.spec.finetune_epochs) == 0
            and int(self.spec.distill_epochs) == 0
        ):
            raise InductiveValidationError(
                "At least one of pretrain_epochs, finetune_epochs, or distill_epochs must be > 0."
            )
        if not (0.0 <= float(self.spec.alpha) <= 1.0):
            raise InductiveValidationError("alpha must be in [0, 1].")
        if float(self.spec.temperature) <= 0:
            raise InductiveValidationError("temperature must be > 0.")
        if float(self.spec.distill_temperature) <= 0:
            raise InductiveValidationError("distill_temperature must be > 0.")

        if int(self.spec.pretrain_epochs) > 0 or int(self.spec.distill_epochs) > 0:
            if X_uw is None:
                raise InductiveValidationError(
                    "SimCLRv2 requires unlabeled data for pretrain/distill."
                )
            if int(get_torch_len(X_uw)) == 0:
                raise InductiveValidationError("X_u must be non-empty for pretrain/distill.")
            if X_us is None:
                X_us = X_uw
            if int(get_torch_len(X_uw)) != int(get_torch_len(X_us)):
                raise InductiveValidationError("X_u_w and X_u_s must have the same number of rows.")
            ensure_float_tensor(X_uw, name="X_u_w")
            ensure_float_tensor(X_us, name="X_u_s")

        use_labeled = int(self.spec.finetune_epochs) > 0 or bool(self.spec.use_labeled_in_distill)
        if use_labeled:
            if int(get_torch_len(X_l)) == 0:
                raise InductiveValidationError("X_l must be non-empty for supervised stages.")
            ensure_float_tensor(X_l, name="X_l")
            y_l = ensure_1d_labels_torch(y_l, name="y_l")
            if y_l.dtype != torch.int64:
                raise InductiveValidationError("y_l must be int64 for torch cross entropy.")

        pretrain_bundle = None
        finetune_bundle = None
        if int(self.spec.pretrain_epochs) > 0:
            if self.spec.pretrain_bundle is None and self.spec.finetune_bundle is None:
                raise InductiveValidationError(
                    "pretrain_bundle or finetune_bundle must be provided."
                )
            pretrain_bundle = ensure_model_bundle(
                self.spec.pretrain_bundle or self.spec.finetune_bundle
            )
            ensure_model_device(
                pretrain_bundle.model,
                device=get_torch_device(X_uw) if X_uw is not None else get_torch_device(X_l),
            )

        if int(self.spec.finetune_epochs) > 0 or int(self.spec.distill_epochs) > 0:
            if self.spec.finetune_bundle is None and pretrain_bundle is None:
                raise InductiveValidationError(
                    "finetune_bundle or pretrain_bundle must be provided."
                )
            finetune_bundle = ensure_model_bundle(self.spec.finetune_bundle or pretrain_bundle)
            ensure_model_device(finetune_bundle.model, device=get_torch_device(X_l))

        if int(self.spec.pretrain_epochs) > 0 and pretrain_bundle is not None:
            steps_u = num_batches(int(get_torch_len(X_uw)), int(self.spec.batch_size))
            gen_u = torch.Generator().manual_seed(int(seed))
            model = pretrain_bundle.model
            optimizer = pretrain_bundle.optimizer
            model.train()
            for epoch in range(int(self.spec.pretrain_epochs)):
                iter_u = cycle_batch_indices(
                    int(get_torch_len(X_uw)),
                    batch_size=int(self.spec.batch_size),
                    generator=gen_u,
                    device=get_torch_device(X_uw),
                    steps=steps_u,
                )
                for step, idx_u in enumerate(iter_u):
                    x_uw = slice_data(X_uw, idx_u)
                    x_us = slice_data(X_us, idx_u)
                    z1 = _forward_projection(model, pretrain_bundle.meta, x_uw)
                    z2 = _forward_projection(model, pretrain_bundle.meta, x_us)
                    if int(z1.ndim) != 2 or int(z2.ndim) != 2:
                        raise InductiveValidationError(
                            "Projection outputs must be 2D (batch, dim)."
                        )
                    if z1.shape != z2.shape:
                        raise InductiveValidationError(
                            "Projection outputs must have the same shape."
                        )
                    loss = _nt_xent_loss(
                        torch.cat([z1, z2], dim=0), temperature=float(self.spec.temperature)
                    )
                    if step == 0:
                        logger.debug(
                            "SimCLRv2 pretrain epoch=%s loss=%.4f",
                            epoch,
                            float(loss.item()),
                        )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        if (
            bool(self.spec.transfer_pretrain)
            and pretrain_bundle is not None
            and finetune_bundle is not None
            and pretrain_bundle is not finetune_bundle
            and int(self.spec.pretrain_epochs) > 0
        ):
            try:
                finetune_bundle.model.load_state_dict(
                    pretrain_bundle.model.state_dict(), strict=False
                )
            except Exception as exc:  # pragma: no cover - defensive
                raise InductiveValidationError(
                    "finetune_bundle.model must be compatible with pretrain_bundle.model."
                ) from exc

        if int(self.spec.finetune_epochs) > 0 and finetune_bundle is not None:
            steps_l = num_batches(int(get_torch_len(X_l)), int(self.spec.batch_size))
            gen_l = torch.Generator().manual_seed(int(seed) + 1)
            model = finetune_bundle.model
            optimizer = finetune_bundle.optimizer
            model.train()
            for epoch in range(int(self.spec.finetune_epochs)):
                iter_l = cycle_batches(
                    X_l,
                    y_l,
                    batch_size=int(self.spec.batch_size),
                    generator=gen_l,
                    steps=steps_l,
                )
                for step, (x_lb, y_lb) in enumerate(iter_l):
                    logits = _forward_logits(model, finetune_bundle.meta, x_lb)
                    if int(logits.ndim) != 2:
                        raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                    if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits.shape[1]):
                        raise InductiveValidationError("y_l labels must be within [0, n_classes).")
                    sup_loss = torch.nn.functional.cross_entropy(logits, y_lb)
                    if step == 0:
                        logger.debug(
                            "SimCLRv2 finetune epoch=%s sup_loss=%.4f",
                            epoch,
                            float(sup_loss.item()),
                        )
                    optimizer.zero_grad()
                    sup_loss.backward()
                    optimizer.step()

        if int(self.spec.distill_epochs) > 0 and finetune_bundle is not None:
            if X_uw is None:
                raise InductiveValidationError("SimCLRv2 distill requires unlabeled data.")
            student_bundle = self.spec.student_bundle
            if student_bundle is None:
                student_model = finetune_bundle.model
                teacher_model = copy.deepcopy(student_model)
                teacher_meta = _rebind_meta(
                    finetune_bundle.meta, source=student_model, target=teacher_model
                )
                optimizer = finetune_bundle.optimizer
                student_meta = finetune_bundle.meta
            else:
                student_bundle = ensure_model_bundle(student_bundle)
                student_model = student_bundle.model
                optimizer = student_bundle.optimizer
                teacher_model = finetune_bundle.model
                teacher_meta = finetune_bundle.meta
                student_meta = student_bundle.meta
                ensure_model_device(student_model, device=get_torch_device(X_uw))
                ensure_model_device(teacher_model, device=get_torch_device(X_uw))
                _check_distill_models(student_model, teacher_model)

            for p in teacher_model.parameters():
                p.requires_grad_(False)
            teacher_model.eval()
            student_model.train()

            steps_u = num_batches(int(get_torch_len(X_uw)), int(self.spec.batch_size))
            steps_l = (
                num_batches(int(get_torch_len(X_l)), int(self.spec.batch_size))
                if bool(self.spec.use_labeled_in_distill)
                else 0
            )
            steps_per_epoch = max(int(steps_u), int(steps_l) or 1)
            gen_u = torch.Generator().manual_seed(int(seed) + 2)
            gen_l = torch.Generator().manual_seed(int(seed) + 3)

            for epoch in range(int(self.spec.distill_epochs)):
                iter_u = cycle_batch_indices(
                    int(get_torch_len(X_uw)),
                    batch_size=int(self.spec.batch_size),
                    generator=gen_u,
                    device=get_torch_device(X_uw),
                    steps=steps_per_epoch,
                )
                iter_l = (
                    cycle_batches(
                        X_l,
                        y_l,
                        batch_size=int(self.spec.batch_size),
                        generator=gen_l,
                        steps=steps_per_epoch,
                    )
                    if bool(self.spec.use_labeled_in_distill)
                    else None
                )
                for step in range(int(steps_per_epoch)):
                    idx_u = next(iter_u)
                    x_uw = slice_data(X_uw, idx_u)
                    x_us = slice_data(X_us, idx_u)

                    with (
                        torch.no_grad(),
                        freeze_batchnorm(teacher_model, enabled=bool(self.spec.freeze_bn)),
                    ):
                        logits_t = _forward_logits(teacher_model, teacher_meta, x_uw)
                    logits_s = _forward_logits(student_model, student_meta, x_us)
                    if int(logits_t.ndim) != 2 or int(logits_s.ndim) != 2:
                        raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                    if logits_t.shape != logits_s.shape:
                        raise InductiveValidationError("Teacher and student logits shape mismatch.")

                    distill_loss = _distill_loss(
                        logits_s,
                        logits_t,
                        temperature=float(self.spec.distill_temperature),
                        detach_target=bool(self.spec.detach_target),
                    )

                    if iter_l is not None:
                        x_lb, y_lb = next(iter_l)
                        logits_l = _forward_logits(student_model, student_meta, x_lb)
                        if int(logits_l.ndim) != 2:
                            raise InductiveValidationError(
                                "Model logits must be 2D (batch, classes)."
                            )
                        if y_lb.min().item() < 0 or y_lb.max().item() >= int(logits_l.shape[1]):
                            raise InductiveValidationError(
                                "y_l labels must be within [0, n_classes)."
                            )
                        sup_loss = torch.nn.functional.cross_entropy(logits_l, y_lb)
                        loss = (1.0 - float(self.spec.alpha)) * sup_loss + float(
                            self.spec.alpha
                        ) * distill_loss
                    else:
                        sup_loss = None
                        loss = distill_loss

                    if step == 0:
                        logger.debug(
                            "SimCLRv2 distill epoch=%s sup_loss=%s distill_loss=%.4f",
                            epoch,
                            f"{float(sup_loss.item()):.4f}" if sup_loss is not None else "n/a",
                            float(distill_loss.item()),
                        )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if student_bundle is not None:
                finetune_bundle = student_bundle

        if int(self.spec.distill_epochs) > 0 and self.spec.student_bundle is not None:
            final_bundle = self.spec.student_bundle
        elif int(self.spec.finetune_epochs) > 0:
            final_bundle = finetune_bundle
        else:
            final_bundle = pretrain_bundle

        self._bundle = final_bundle
        self._backend = backend
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, X: Any) -> Any:
        if self._bundle is None:
            raise RuntimeError("SimCLRv2Method is not fitted yet. Call fit() first.")
        backend = self._backend or detect_backend(X)
        if backend != "torch":
            raise InductiveValidationError("SimCLRv2 predict_proba requires torch tensors.")
        torch = optional_import("torch", extra="inductive-torch")
        if not isinstance(X, torch.Tensor) and not (isinstance(X, dict) and "x" in X):
            raise InductiveValidationError("predict_proba requires torch.Tensor or dict inputs.")

        model = self._bundle.model
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = _forward_logits(model, self._bundle.meta, X)
            if int(logits.ndim) != 2:
                raise InductiveValidationError("Model logits must be 2D (batch, classes).")
            proba = torch.softmax(logits, dim=1)
        if was_training:
            model.train()
        return proba

    def predict(self, X: Any) -> Any:
        proba = self.predict_proba(X)
        return proba.argmax(dim=1)
