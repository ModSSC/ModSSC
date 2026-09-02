from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from modssc.capabilities import TORCH_INDUCTIVE_CAPABILITIES, MethodCapabilities
from modssc.inductive.base import InductiveMethod, MethodInfo
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.deep_utils import (
    ArgmaxPredictMixin,
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
from modssc.inductive.model_binding import ModelBindingSpec
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec
from modssc.runtime.contracts import (
    ComponentRelation,
    ComponentRequirement,
    MethodExecutionContract,
)
from modssc.runtime.method_contracts import (
    fallback_method_execution_contract,
    with_inductive_input_roles,
)

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
    if isinstance(out, Mapping):
        for key in ("feat", "features", "embedding"):
            if key in out:
                return _as_tensor(out[key], name=f"model feature output[{key}]")
    raise InductiveValidationError(
        "SimCLRv2 requires an explicit encoder feature via meta['forward_features'] "
        "or a model output keyed as 'feat', 'features', or 'embedding'; logits and "
        "projection outputs cannot serve as encoder features."
    )


def _forward_projection(
    model: Any,
    meta: Mapping[str, Any] | None,
    X: Any,
    *,
    contract: Any | None = None,
) -> Any:
    if isinstance(meta, Mapping):
        forward = meta.get("forward_projection")
        if callable(forward):
            return _as_tensor(forward(X), name="forward_projection output")
        projector = meta.get("projection_head") or meta.get("projector")
        if callable(projector):
            declared_outputs = frozenset(getattr(contract, "outputs", ()))
            if contract is not None and not declared_outputs & {
                "forward_features",
                "feature_extractor",
                "encoder",
                "feat",
                "features",
                "embedding",
            }:
                raise InductiveValidationError(
                    "SimCLRv2 projection_head requires a declared encoder feature path; "
                    "classifier logits cannot be projected as features."
                )
            feats = _forward_features(model, meta, X)
            return _as_tensor(projector(feats), name="projection_head output")
    out = model(X)
    if isinstance(out, Mapping):
        for key in ("proj", "projection", "z"):
            if key in out:
                return _as_tensor(out[key], name=f"model projection output[{key}]")
    declared_outputs = frozenset(getattr(contract, "outputs", ()))
    if declared_outputs & {"proj", "projection", "z"} and not isinstance(out, Mapping):
        return _tensor_from_output(
            out,
            keys=("proj", "projection", "z"),
            name="declared model projection output",
        )
    raise InductiveValidationError(
        "SimCLRv2 pretraining requires an explicit projection head or a declared "
        "projection output; classifier logits and encoder features are not projections."
    )


def _validate_projection_optimization(bundle: TorchModelBundle) -> None:
    """Require projection state to be model-owned, optimized, and serializable."""

    model_parameters = tuple(
        parameter for parameter in bundle.model.parameters() if parameter.requires_grad
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in bundle.optimizer.param_groups
        for parameter in group.get("params", ())
    }
    missing_model_parameters = [
        parameter for parameter in model_parameters if id(parameter) not in optimizer_parameter_ids
    ]
    if missing_model_parameters:
        raise InductiveValidationError(
            "SimCLRv2 pretraining optimizer must own every trainable encoder/projection parameter."
        )

    meta = bundle.meta
    if not isinstance(meta, Mapping):
        return
    projection_callable = None
    for key in ("forward_projection", "projection_head", "projector"):
        candidate = meta.get(key)
        if callable(candidate):
            projection_callable = candidate
            break
    if projection_callable is None:
        return
    owner = (
        projection_callable
        if hasattr(projection_callable, "parameters")
        else getattr(projection_callable, "__self__", None)
    )
    if owner is None or not hasattr(owner, "parameters"):
        raise InductiveValidationError(
            "SimCLRv2 projection callable must be a module or a bound module method so its "
            "optimization state is auditable."
        )
    model_module_ids = {id(module) for module in bundle.model.modules()}
    if id(owner) not in model_module_ids:
        raise InductiveValidationError(
            "SimCLRv2 projection head must be registered inside bundle.model so its state "
            "is included in model.state_dict()."
        )
    projection_parameters = tuple(
        parameter for parameter in owner.parameters() if parameter.requires_grad
    )
    if not projection_parameters:
        raise InductiveValidationError("SimCLRv2 projection head must expose trainable parameters.")


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


class SimCLRv2Method(ArgmaxPredictMixin, InductiveMethod):
    """SimCLRv2 pretrain -> fine-tune -> distill (torch-only)."""

    info = MethodInfo(
        method_id="simclr_v2",
        name="SimCLRv2",
        year=2020,
        family="contrastive",
        supports_gpu=True,
        paper_title="Big Self-Supervised Models are Strong Semi-Supervised Learners",
        paper_pdf="https://arxiv.org/pdf/2006.10029",
        official_code="https://github.com/google-research/simclr",
        capabilities=TORCH_INDUCTIVE_CAPABILITIES,
        model_binding=ModelBindingSpec.pretrain_finetune(),
    )

    @classmethod
    def execution_contract(
        cls,
        spec: SimCLRv2Spec,
        capabilities: MethodCapabilities,
        model_binding: Any | None = None,
    ) -> MethodExecutionContract:
        pretrain_active = int(spec.pretrain_epochs) > 0
        finetune_active = int(spec.finetune_epochs) > 0
        distill_active = int(spec.distill_epochs) > 0
        needs_unlabeled = pretrain_active or distill_active
        effective_capabilities = replace(
            capabilities,
            requires_unlabeled=needs_unlabeled,
            requires_weak_augmentation=False,
            min_strong_augmentations=0,
        )
        unlabeled_roles = ("fit.X_u", "fit.X_u_w", "fit.X_u_s.0") if needs_unlabeled else ()
        feature_roles = ("fit.X_l", *unlabeled_roles)
        contract = with_inductive_input_roles(
            fallback_method_execution_contract(
                cls,
                effective_capabilities,
                model_binding,
            ),
            feature_roles=feature_roles,
            optional_feature_roles=(unlabeled_roles if needs_unlabeled else ()),
            row_groups=(
                (("fit.X_l", "fit.y_l"),)
                + ((("fit.X_u_w", "fit.X_u_s.0"),) if needs_unlabeled else ())
            ),
        )

        if len(contract.components) != 2:
            return contract
        pretrain_template, finetune_template = contract.components
        templates = {
            pretrain_template.slot: pretrain_template,
            finetune_template.slot: finetune_template,
        }
        selected: dict[str, ComponentRequirement] = {}

        def merge_requirement(
            slot: str,
            *,
            outputs: frozenset[str] = frozenset(),
            output_alternatives: tuple[frozenset[str], ...] = (),
            input_roles: tuple[str, ...] = (),
        ) -> None:
            previous = selected.get(slot)
            template = (
                replace(
                    templates[slot],
                    outputs=frozenset(),
                    output_alternatives=(),
                    input_roles=(),
                )
                if previous is None
                else previous
            )
            selected[slot] = replace(
                template,
                outputs=template.outputs | outputs,
                output_alternatives=(
                    *template.output_alternatives,
                    *output_alternatives,
                ),
                input_roles=(*template.input_roles, *input_roles),
            )

        pretrain_slot: str | None = None
        if pretrain_active:
            pretrain_slot = (
                pretrain_template.slot
                if spec.pretrain_bundle is not None or spec.finetune_bundle is None
                else finetune_template.slot
            )
            merge_requirement(
                pretrain_slot,
                output_alternatives=(
                    frozenset({"forward_projection"}),
                    frozenset({"projection"}),
                    frozenset({"proj"}),
                    frozenset({"z"}),
                    *tuple(
                        frozenset({projector, feature})
                        for projector in ("projection_head", "projector")
                        for feature in (
                            "forward_features",
                            "feature_extractor",
                            "encoder",
                            "feat",
                            "features",
                            "embedding",
                        )
                    ),
                ),
                input_roles=unlabeled_roles,
            )

        finetune_slot: str | None = None
        if finetune_active or distill_active:
            finetune_slot = (
                finetune_template.slot
                if spec.finetune_bundle is not None or pretrain_slot is None
                else pretrain_slot
            )
            finetune_inputs: list[str] = []
            if finetune_active:
                finetune_inputs.append("fit.X_l")
            if distill_active:
                finetune_inputs.extend(unlabeled_roles)
                if spec.student_bundle is None and bool(spec.use_labeled_in_distill):
                    finetune_inputs.append("fit.X_l")
            merge_requirement(
                finetune_slot,
                outputs=frozenset({"logits"}),
                input_roles=tuple(finetune_inputs),
            )

        component_relations: tuple[ComponentRelation, ...] = ()
        if distill_active and spec.student_bundle is not None:
            student_inputs = list(unlabeled_roles)
            if bool(spec.use_labeled_in_distill):
                student_inputs.append("fit.X_l")
            selected["student_bundle"] = ComponentRequirement(
                slot="student_bundle",
                kind="torch_model",
                outputs=frozenset({"logits"}),
                input_roles=tuple(student_inputs),
                requires_optimizer=True,
            )
            assert finetune_slot is not None
            relation_slots = (finetune_slot, "student_bundle")
            component_relations = (
                ComponentRelation("distinct_objects", relation_slots),
                ComponentRelation("disjoint_parameters", relation_slots),
            )

        return replace(
            contract,
            components=tuple(selected[slot] for slot in sorted(selected)),
            component_relations=component_relations,
        )

    def __init__(self, spec: SimCLRv2Spec | None = None) -> None:
        self.spec = spec or SimCLRv2Spec()
        self._bundle: TorchModelBundle | None = None
        self._backend: str | None = None
        self.device: str | None = None

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
            _validate_projection_optimization(pretrain_bundle)
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
                    z1 = _forward_projection(
                        model,
                        pretrain_bundle.meta,
                        x_uw,
                        contract=pretrain_bundle.contract,
                    )
                    z2 = _forward_projection(
                        model,
                        pretrain_bundle.meta,
                        x_us,
                        contract=pretrain_bundle.contract,
                    )
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
            if X_uw is None:  # pragma: no cover - guarded by the earlier unlabeled-data validation
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
        self.device = str(get_torch_device(X_l))
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
        batch_size = int(self.spec.batch_size)
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
                logits = _forward_logits(model, self._bundle.meta, batch_X)
                if int(logits.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
                all_logits.append(logits)
            if not all_logits:
                if isinstance(X, dict):
                    empty_idx = torch.arange(0, 0, device=X["x"].device)
                    empty_X = slice_data(X, empty_idx)
                else:
                    empty_X = X[:0]
                logits = _forward_logits(model, self._bundle.meta, empty_X)
                if int(logits.ndim) != 2:
                    raise InductiveValidationError("Model logits must be 2D (batch, classes).")
            else:
                logits = torch.cat(all_logits, dim=0)
            proba = torch.softmax(logits, dim=1)
        if was_training:
            model.train()
        return proba
