"""Native model-to-method binding for inductive methods.

The benchmark runner may describe a model through configuration, but the
method owns how many independent bundles it needs and where they are injected
in its specification.  This module materializes that method-owned contract
without depending on ``bench`` types.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Literal

import numpy as np

from modssc.data_augmentation.utils import is_torch_tensor
from modssc.utils.imports import load_object

from .deep import build_torch_bundle_from_classifier
from .errors import InductiveValidationError

ModelBindingKind = Literal[
    "none",
    "single",
    "teacher_student",
    "pair",
    "pretrain_finetune",
    "shared_heads",
]
ModelBindingErrorKind = Literal[
    "method_spec",
    "model_config",
    "torch_required",
    "dtype",
]


class ModelBindingError(InductiveValidationError):
    """Raised when a native method model-binding contract cannot be fulfilled."""

    def __init__(self, kind: ModelBindingErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class ModelBuildConfig:
    """Runner-independent description of a model bundle factory."""

    factory: str | Callable[..., Any] | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    classifier_id: str | None = None
    classifier_backend: str | None = None
    classifier_params: Mapping[str, Any] = field(default_factory=dict)
    ema: bool | None = None


@dataclass(frozen=True)
class ModelBindingSpec:
    """Declare how a method specification receives constructed model bundles."""

    kind: ModelBindingKind = "none"
    bundle_fields: tuple[str, ...] = ()
    shared_bundle_field: str | None = None
    head_bundles_field: str | None = None
    head_count: int = 0
    head_classifier_ids: tuple[str, ...] = ()
    head_classifier_fallback: str | None = None

    @classmethod
    def single(cls, field_name: str = "model_bundle") -> ModelBindingSpec:
        return cls(kind="single", bundle_fields=(field_name,))

    @classmethod
    def teacher_student(
        cls,
        *,
        student_field: str = "student_bundle",
        teacher_field: str = "teacher_bundle",
    ) -> ModelBindingSpec:
        # Student keeps the base seed and teacher gets the independent offset.
        return cls(kind="teacher_student", bundle_fields=(student_field, teacher_field))

    @classmethod
    def pair(
        cls,
        *,
        first_field: str = "model_bundle_1",
        second_field: str = "model_bundle_2",
    ) -> ModelBindingSpec:
        return cls(kind="pair", bundle_fields=(first_field, second_field))

    @classmethod
    def pretrain_finetune(
        cls,
        *,
        pretrain_field: str = "pretrain_bundle",
        finetune_field: str = "finetune_bundle",
    ) -> ModelBindingSpec:
        return cls(kind="pretrain_finetune", bundle_fields=(pretrain_field, finetune_field))

    @classmethod
    def shared_heads(
        cls,
        *,
        shared_field: str = "shared_bundle",
        heads_field: str = "head_bundles",
        head_count: int,
        head_classifier_ids: tuple[str, ...],
        head_classifier_fallback: str,
    ) -> ModelBindingSpec:
        return cls(
            kind="shared_heads",
            shared_bundle_field=shared_field,
            head_bundles_field=heads_field,
            head_count=int(head_count),
            head_classifier_ids=tuple(head_classifier_ids),
            head_classifier_fallback=head_classifier_fallback,
        )


NO_MODEL_BINDING = ModelBindingSpec()


def _torch_module() -> Any:
    return importlib.import_module("torch")


def _infer_num_classes(y: Any) -> int:
    if is_torch_tensor(y):
        torch = _torch_module()
        return int(torch.unique(y).numel())
    return int(np.unique(np.asarray(y)).size)


def _as_torch_sample(sample: Any, *, strict: bool) -> Any:
    if is_torch_tensor(sample):
        return sample
    if strict:
        raise ModelBindingError(
            "torch_required",
            "model bundle requires a torch sample; declare conversion in preprocessing",
        )
    torch = _torch_module()
    sample_np = np.asarray(sample)
    if sample_np.dtype == np.uint8:
        return torch.tensor(sample_np, dtype=torch.float32).div_(255.0)
    dtype = torch.float32 if sample_np.dtype == np.float64 else None
    return torch.as_tensor(sample_np, dtype=dtype)


def _validate_float_sample(sample: Any, *, strict: bool) -> None:
    if not strict:
        return
    torch = _torch_module()
    if not torch.is_floating_point(sample):
        raise ModelBindingError(
            "dtype",
            f"model bundle sample must be a floating tensor; got {sample.dtype}",
        )


def _first_graph_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Take node zero from a graph mapping for shared-feature shape probing."""

    x = sample.get("x")
    shape = getattr(x, "shape", None)
    if shape is None or len(shape) == 0 or int(shape[0]) <= 1:
        return dict(sample)
    node_count = int(shape[0])
    out: dict[str, Any] = {}
    for key, value in sample.items():
        if key == "edge_index":
            edge_index = value
            if is_torch_tensor(edge_index) or isinstance(edge_index, np.ndarray):
                mask = (edge_index[0] == 0) & (edge_index[1] == 0)
                out[key] = edge_index[:, mask]
            else:
                out[key] = edge_index
            continue
        value_shape = getattr(value, "shape", None)
        slices_by_shape = (
            value_shape is not None and len(value_shape) > 0 and int(value_shape[0]) == node_count
        )
        slices_as_list = isinstance(value, list) and len(value) == node_count
        if slices_by_shape or slices_as_list:
            out[key] = value[:1]
        elif key == "num_nodes":
            out[key] = 1
        else:
            out[key] = value
    return out


def _shared_probe_sample(sample: Any) -> Any:
    if is_torch_tensor(sample):
        if int(sample.ndim) > 0 and int(sample.shape[0]) > 1:
            return sample[:1]
        return sample
    if isinstance(sample, Mapping) and "x" in sample:
        return _first_graph_sample(sample)
    raise ModelBindingError(
        "torch_required",
        "shared/head bundle construction requires torch inputs",
    )


def _extract_head_sample(output: Any) -> Any:
    if is_torch_tensor(output):
        return output.detach()
    if isinstance(output, Mapping):
        for key in ("feat", "features", "embedding", "proj", "projection", "z", "logits"):
            candidate = output.get(key)
            if is_torch_tensor(candidate):
                return candidate.detach()
    if isinstance(output, tuple) and output and is_torch_tensor(output[0]):
        return output[0].detach()
    raise ModelBindingError(
        "model_config",
        "head bundle construction requires the shared model to return a torch.Tensor",
    )


def _shared_head_sample(bundle: Any, sample: Any) -> Any:
    """Resolve the same shared representation consumed by TriNet at runtime."""

    meta = getattr(bundle, "meta", None)
    forward = None
    if isinstance(meta, Mapping):
        forward = meta.get("forward_features") or meta.get("feature_extractor")
    output = forward(sample) if callable(forward) else bundle.model(sample)
    return _extract_head_sample(output)


def _validate_declared_fields(spec: Any, binding: ModelBindingSpec) -> None:
    fields = binding.bundle_fields
    if binding.kind == "shared_heads":
        fields = tuple(
            field_name
            for field_name in (binding.shared_bundle_field, binding.head_bundles_field)
            if field_name is not None
        )
    missing = [field_name for field_name in fields if not hasattr(spec, field_name)]
    if missing:
        raise ModelBindingError(
            "method_spec",
            f"method model-binding declaration references absent spec fields: {missing}",
        )


def _validate_prebound_model_spec(spec: Any, binding: ModelBindingSpec) -> None:
    """Require programmatic callers to provide the bundles a binding consumes."""

    if binding.kind == "none":
        return
    if spec is None or not is_dataclass(spec):
        raise ModelBindingError(
            "model_config",
            "method requires a native model bundle configuration",
        )
    _validate_declared_fields(spec, binding)
    if binding.kind == "pretrain_finetune":
        if not any(getattr(spec, field_name) is not None for field_name in binding.bundle_fields):
            raise ModelBindingError(
                "model_config",
                "pretrain/finetune method requires at least one bound model bundle",
            )
        return
    if binding.kind == "shared_heads":
        fields = tuple(
            field_name
            for field_name in (binding.shared_bundle_field, binding.head_bundles_field)
            if field_name is not None
        )
    else:
        fields = binding.bundle_fields
    missing = [field_name for field_name in fields if getattr(spec, field_name) is None]
    if missing:
        raise ModelBindingError(
            "model_config",
            f"method requires bound model fields: {missing}",
        )


def bind_model_to_spec(
    spec: Any,
    config: ModelBuildConfig | None,
    *,
    binding: ModelBindingSpec,
    X_l: Any,
    y_l: Any,
    default_ema: bool,
    seed: int,
    strict: bool = False,
) -> Any:
    """Construct and inject model bundles according to a method declaration."""

    if config is None:
        _validate_prebound_model_spec(spec, binding)
        return spec
    if spec is None:
        raise ModelBindingError(
            "method_spec",
            "model configuration is set but no dataclass method spec is available",
        )
    if not is_dataclass(spec):
        raise ModelBindingError("method_spec", "model binding requires a dataclass method spec")
    if binding.kind == "none":
        raise ModelBindingError(
            "model_config",
            "method does not declare a native model-binding contract",
        )
    _validate_declared_fields(spec, binding)

    def make_bundle(
        *,
        seed_offset: int,
        sample_override: Any | None = None,
        classifier_id: str | None = None,
        classifier_backend: str | None = None,
        classifier_params: Mapping[str, Any] | None = None,
        ema: bool | None = None,
    ) -> Any:
        if config.factory is not None:
            factory = (
                load_object(config.factory, error_prefix="Invalid import path")
                if isinstance(config.factory, str)
                else config.factory
            )
            return factory(**dict(config.params))

        sample = sample_override if sample_override is not None else X_l
        if isinstance(sample, Mapping) and "x" in sample:
            sample = sample["x"]
        sample = _as_torch_sample(sample, strict=strict)
        _validate_float_sample(sample, strict=strict)

        local_classifier_id = classifier_id or config.classifier_id
        if local_classifier_id is None:
            raise ModelBindingError(
                "model_config",
                "classifier_id must be provided for torch model bundles",
            )
        local_ema = config.ema if ema is None else ema
        if local_ema is None:
            local_ema = bool(default_ema)
        return build_torch_bundle_from_classifier(
            classifier_id=local_classifier_id,
            classifier_backend=classifier_backend or config.classifier_backend,
            classifier_params=(
                config.classifier_params if classifier_params is None else classifier_params
            ),
            sample=sample,
            num_classes=_infer_num_classes(y_l),
            seed=int(seed) + int(seed_offset),
            ema=bool(local_ema),
        )

    if binding.kind in {"single", "teacher_student", "pair", "pretrain_finetune"}:
        replacements = {
            field_name: make_bundle(seed_offset=offset)
            for offset, field_name in enumerate(binding.bundle_fields)
        }
        return replace(spec, **replacements)

    if binding.kind != "shared_heads":  # pragma: no cover - guarded by the Literal contract
        raise ModelBindingError("model_config", f"unknown model-binding kind: {binding.kind!r}")
    if config.factory is not None:
        raise ModelBindingError(
            "model_config",
            "shared/head bundle specs do not support model.factory; use classifier config",
        )
    if binding.shared_bundle_field is None or binding.head_bundles_field is None:
        raise ModelBindingError("method_spec", "shared/head binding fields must be declared")
    if binding.head_count <= 0:
        raise ModelBindingError("method_spec", "shared/head binding head_count must be positive")
    if binding.head_classifier_fallback not in binding.head_classifier_ids:
        raise ModelBindingError(
            "method_spec",
            "shared/head classifier fallback must be one of the declared classifier ids",
        )

    shared_bundle = make_bundle(seed_offset=0)
    probe_sample = _shared_probe_sample(X_l)
    head_sample = _shared_head_sample(shared_bundle, probe_sample)
    head_classifier_id = config.classifier_id
    if head_classifier_id not in binding.head_classifier_ids:
        head_classifier_id = binding.head_classifier_fallback
    head_bundles = tuple(
        make_bundle(
            seed_offset=1 + index,
            sample_override=head_sample,
            classifier_id=head_classifier_id,
            classifier_backend="torch",
            classifier_params=config.classifier_params,
            ema=False,
        )
        for index in range(binding.head_count)
    )
    return replace(
        spec,
        **{
            binding.shared_bundle_field: shared_bundle,
            binding.head_bundles_field: head_bundles,
        },
    )


__all__ = [
    "ModelBindingError",
    "ModelBindingKind",
    "ModelBindingSpec",
    "ModelBuildConfig",
    "NO_MODEL_BINDING",
    "bind_model_to_spec",
]
