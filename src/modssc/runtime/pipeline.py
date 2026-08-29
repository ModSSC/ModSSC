"""Native resolution of method and materialized-pipeline runtime contracts.

This module is deliberately independent from YAML runners.  It owns registry
selection, backend and device policy, the explicit torch-preprocessing contract,
and method-to-pipeline capability validation for both learning regimes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal, cast

from modssc.capabilities import (
    CompatibilityReport,
    IncompatiblePipelineError,
    LearningRegime,
    MethodCapabilities,
    PipelineCapabilities,
    materialize_pipeline_capabilities,
    validate_pipeline_compatibility,
)
from modssc.inductive.registry import get_method_class as get_inductive_method_class
from modssc.inductive.registry import get_method_info as get_inductive_method_info
from modssc.runtime.device import resolve_device_name
from modssc.runtime.method_spec import MethodSpecError, build_method_spec, method_spec_has_field
from modssc.supervised.api import resolve_classifier_backend_spec
from modssc.supervised.errors import SupervisedError, UnknownClassifierError
from modssc.transductive.registry import get_method_class as get_transductive_method_class
from modssc.transductive.registry import get_method_info as get_transductive_method_info

PipelineResolutionErrorKind = Literal[
    "method_lookup",
    "method_introspection",
    "method_spec",
    "auto_device",
    "auto_backend",
    "backend_required",
    "torch_preprocess",
    "capability",
]

_ERROR_CODES: dict[PipelineResolutionErrorKind, str] = {
    "method_lookup": "E_PIPELINE_METHOD_LOOKUP",
    "method_introspection": "E_PIPELINE_METHOD_INTROSPECTION",
    "method_spec": "E_PIPELINE_METHOD_SPEC",
    "auto_device": "E_PIPELINE_AUTO_DEVICE_FORBIDDEN",
    "auto_backend": "E_PIPELINE_AUTO_BACKEND_FORBIDDEN",
    "backend_required": "E_PIPELINE_BACKEND_REQUIRED",
    "torch_preprocess": "E_PIPELINE_TORCH_PREPROCESS_REQUIRED",
    "capability": "E_PIPELINE_CAPABILITY",
}

_TORCH_PREPROCESS_STEP_ID = "core.to_torch"
_REGIMES = frozenset({"inductive", "transductive"})


class PipelineResolutionError(ValueError):
    """Raised when a native runtime pipeline cannot be resolved safely."""

    def __init__(
        self,
        kind: PipelineResolutionErrorKind,
        message: str,
        *,
        report: CompatibilityReport | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.code = _ERROR_CODES[kind]
        self.report = report


def _non_empty_name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_backend(value: Any) -> str | None:
    if value is None:
        return None
    return _non_empty_name(value, field_name="backend")


@dataclass(frozen=True)
class MethodResolutionRequest:
    """Runner-independent inputs needed to resolve one registered method."""

    regime: LearningRegime
    method_id: str
    params: Mapping[str, Any] = field(default_factory=dict)
    requested_device: str = "cpu"
    dtype: str = "float32"
    strict: bool = False
    preprocess_step_ids: Sequence[str] = ()
    model_classifier_id: str | None = None
    model_classifier_backend: str | None = None
    model_configured: bool = False

    def __post_init__(self) -> None:
        if self.regime not in _REGIMES:
            raise ValueError(f"regime must be one of {sorted(_REGIMES)!r}")
        object.__setattr__(self, "regime", cast(LearningRegime, self.regime))
        object.__setattr__(
            self, "method_id", _non_empty_name(self.method_id, field_name="method_id")
        )
        if not isinstance(self.params, Mapping):
            raise TypeError("params must be a mapping")
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(
            self,
            "requested_device",
            _non_empty_name(self.requested_device, field_name="requested_device"),
        )
        object.__setattr__(self, "dtype", _non_empty_name(self.dtype, field_name="dtype"))
        if isinstance(self.preprocess_step_ids, str):
            raise TypeError("preprocess_step_ids must be a sequence of step identifiers")
        object.__setattr__(
            self,
            "preprocess_step_ids",
            tuple(
                _non_empty_name(step_id, field_name="preprocess_step_ids item")
                for step_id in self.preprocess_step_ids
            ),
        )
        object.__setattr__(
            self,
            "model_classifier_id",
            (
                None
                if self.model_classifier_id is None
                else _non_empty_name(
                    self.model_classifier_id,
                    field_name="model_classifier_id",
                )
            ),
        )
        object.__setattr__(
            self,
            "model_classifier_backend",
            _optional_backend(self.model_classifier_backend),
        )
        object.__setattr__(self, "strict", bool(self.strict))
        object.__setattr__(self, "model_configured", bool(self.model_configured))


@dataclass(frozen=True)
class MethodRuntimeResolution:
    """Portable public facts resolved from a registered method contract."""

    regime: LearningRegime
    method_id: str
    supports_gpu: bool
    required_extra: str | None
    capabilities: MethodCapabilities
    requested_device: str
    resolved_device: str
    requested_backend: str | None
    resolved_backend: str | None
    classifier_backend: str | None
    requires_torch: bool
    model_configured: bool
    dtype: str
    strict: bool
    preprocess_step_ids: tuple[str, ...]
    required_extras: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible method facts useful for run resolution reports."""

        return {
            "regime": self.regime,
            "method_id": self.method_id,
            "supports_gpu": self.supports_gpu,
            "required_extra": self.required_extra,
            "required_extras": list(self.required_extras),
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "classifier_backend": self.classifier_backend,
            "requires_torch": self.requires_torch,
            "model_configured": self.model_configured,
            "dtype": self.dtype,
            "strict": self.strict,
            "preprocess_step_ids": list(self.preprocess_step_ids),
        }


@dataclass(frozen=True)
class MaterializedPipeline:
    """Exact materialized inputs used to validate a resolved method."""

    modality: str
    primary_input: Any
    sampling: Any
    view_count: int = 0
    has_graph: bool = False
    has_weak_augmentation: bool = False
    strong_augmentation_count: int = 0
    checkpointing_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", _non_empty_name(self.modality, field_name="modality"))
        for field_name in ("view_count", "strong_augmentation_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in (
            "has_graph",
            "has_weak_augmentation",
            "checkpointing_required",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))


@dataclass(frozen=True)
class PipelineResolutionRequest:
    """Complete native request for resolving and validating one pipeline."""

    method: MethodResolutionRequest
    pipeline: MaterializedPipeline

    def __post_init__(self) -> None:
        if not isinstance(self.method, MethodResolutionRequest):
            raise TypeError("method must be a MethodResolutionRequest")
        if not isinstance(self.pipeline, MaterializedPipeline):
            raise TypeError("pipeline must be a MaterializedPipeline")


@dataclass(frozen=True)
class PipelineResolution:
    """Resolved method facts and validated materialized capabilities."""

    method: MethodRuntimeResolution
    pipeline_capabilities: PipelineCapabilities
    compatibility: CompatibilityReport

    @property
    def resolved_backend(self) -> str | None:
        return self.pipeline_capabilities.backend

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible facts for reproducibility reports."""

        return {
            "method": self.method.to_dict(),
            "pipeline_capabilities": self.pipeline_capabilities.to_dict(),
            "compatibility": {
                "compatible": self.compatibility.compatible,
                "issues": [
                    {"code": issue.code, "message": issue.message}
                    for issue in self.compatibility.issues
                ],
            },
        }


def _method_class_and_info(regime: LearningRegime, method_id: str) -> tuple[type[Any], Any]:
    try:
        if regime == "inductive":
            return get_inductive_method_class(method_id), get_inductive_method_info(method_id)
        return get_transductive_method_class(method_id), get_transductive_method_info(method_id)
    except (TypeError, ValueError, RuntimeError, ImportError, ModuleNotFoundError) as exc:
        raise PipelineResolutionError(
            "method_lookup",
            f"failed to resolve {regime} method {method_id!r}: {exc}",
        ) from exc


def _method_backend(
    request: MethodResolutionRequest,
    *,
    method_class: type[Any],
) -> tuple[str | None, str | None]:
    requested = _optional_backend(request.params.get("backend"))
    try:
        has_backend_field = method_spec_has_field(method_class, "backend", strict=request.strict)
    except MethodSpecError as exc:
        raise PipelineResolutionError(exc.kind, str(exc)) from exc

    if request.strict and has_backend_field and requested is None:
        raise PipelineResolutionError(
            "backend_required",
            f"method params must explicitly define backend for {request.method_id!r} "
            "in strict execution",
        )

    resolved = requested
    if resolved is None and has_backend_field:
        try:
            spec = build_method_spec(
                method_class,
                {},
                require_spec=True,
                strict=request.strict,
            )
        except MethodSpecError as exc:
            raise PipelineResolutionError(exc.kind, str(exc)) from exc
        resolved = _optional_backend(getattr(spec, "backend", None))

    if resolved is not None and resolved.lower() == "auto":
        if request.strict:
            raise PipelineResolutionError(
                "auto_backend",
                "method backend 'auto' is forbidden in strict execution",
            )
        resolved = None
    return requested, resolved


def _classifier_backend(request: MethodResolutionRequest) -> str | None:
    backend = _optional_backend(request.params.get("classifier_backend"))
    if backend is None:
        backend = request.model_classifier_backend
    if backend is not None and backend.lower() == "auto":
        if request.strict:
            raise PipelineResolutionError(
                "auto_backend",
                "classifier backend 'auto' is forbidden in strict execution",
            )
        return None
    return backend


_MISSING = object()


def _classifier_references(
    value: Any,
    *,
    seen: set[int] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return classifier identifiers from one recursively materialized method spec."""

    active_seen = seen if seen is not None else set()
    if isinstance(value, (str, bytes)) or value is None:
        return ()

    item_id = id(value)
    if item_id in active_seen:
        return ()

    values: Mapping[str, Any] | None = None
    if is_dataclass(value) and not isinstance(value, type):
        values = {spec_field.name: getattr(value, spec_field.name) for spec_field in fields(value)}
    elif isinstance(value, Mapping):
        values = value

    if values is not None:
        active_seen.add(item_id)
        references: list[tuple[str, str]] = []
        nested_specs = values.get("classifier_specs", _MISSING)
        if nested_specs is _MISSING or nested_specs is None:
            classifier_id = values.get("classifier_id", _MISSING)
            classifier_backend = values.get("classifier_backend", _MISSING)
            if classifier_id is not _MISSING or classifier_backend is not _MISSING:
                if classifier_id is _MISSING or classifier_backend is _MISSING:
                    raise ValueError(
                        "classifier dependency declarations require both classifier_id "
                        "and classifier_backend"
                    )
                references.append(
                    (
                        _non_empty_name(classifier_id, field_name="classifier_id"),
                        _non_empty_name(classifier_backend, field_name="classifier_backend"),
                    )
                )

        for key, child in values.items():
            if key in {"classifier_id", "classifier_backend"}:
                continue
            if key == "classifier_specs":
                if child is None:
                    continue
                if isinstance(child, (str, bytes)) or not isinstance(child, Sequence):
                    raise ValueError("classifier_specs must be a sequence when provided")
                for entry in child:
                    if isinstance(entry, Mapping):
                        entry = {
                            "classifier_id": "knn",
                            "classifier_backend": "numpy",
                            **dict(entry),
                        }
                    elif not (is_dataclass(entry) and not isinstance(entry, type)):
                        raise ValueError(
                            "classifier_specs entries must be mappings or dataclass instances"
                        )
                    references.extend(_classifier_references(entry, seen=active_seen))
                continue
            references.extend(_classifier_references(child, seen=active_seen))
        active_seen.remove(item_id)
        return tuple(references)

    if isinstance(value, Sequence):
        active_seen.add(item_id)
        references = tuple(
            reference
            for child in value
            for reference in _classifier_references(child, seen=active_seen)
        )
        active_seen.remove(item_id)
        return references
    return ()


def _materialized_method_spec(request: MethodResolutionRequest, method_class: type[Any]) -> Any:
    try:
        return build_method_spec(
            method_class,
            request.params,
            require_spec=True,
            strict=request.strict,
        )
    except MethodSpecError as exc:
        if not request.params and exc.kind == "method_spec":
            # Dependency discovery must remain compatible with registered methods
            # that have no dataclass spec.  Re-check the concrete class so other
            # method-spec failures are not accidentally swallowed.
            try:
                instance = method_class()
            except (TypeError, ValueError, RuntimeError, ImportError, ModuleNotFoundError):
                instance = None
            spec = getattr(instance, "spec", None) if instance is not None else None
            if spec is None or not is_dataclass(spec):
                return None
        raise PipelineResolutionError(exc.kind, str(exc)) from exc


def _classifier_dependency_facts(
    request: MethodResolutionRequest,
    *,
    method_spec: Any,
    method_required_extra: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    extras = {method_required_extra} if method_required_extra else set()
    resolved_backends: set[str] = set()

    try:
        references = [
            (classifier_id, backend, False)
            for classifier_id, backend in _classifier_references(method_spec)
        ]
    except (TypeError, ValueError) as exc:
        raise PipelineResolutionError(
            "method_spec",
            f"invalid classifier dependency declaration: {exc}",
        ) from exc
    if request.model_classifier_id is not None:
        references.append(
            (
                request.model_classifier_id,
                request.model_classifier_backend or "auto",
                True,
            )
        )

    for classifier_id, backend, allow_unregistered in references:
        try:
            backend_spec = resolve_classifier_backend_spec(classifier_id, backend=backend)
        except UnknownClassifierError as exc:
            if not allow_unregistered:
                raise PipelineResolutionError(
                    "method_spec",
                    f"failed to resolve classifier dependency {classifier_id!r}: {exc}",
                ) from exc
            # Native model-bundle factories may expose classifier identifiers
            # outside the supervised registry.  Their distributions cannot be
            # inferred here and remain an explicit run.software_dependencies
            # declaration owned by the caller.
            if backend != "auto":
                resolved_backends.add(backend)
            continue
        except SupervisedError as exc:
            raise PipelineResolutionError(
                "method_spec",
                "failed to resolve classifier dependency "
                f"{classifier_id!r} with backend {backend!r}: {exc}",
            ) from exc
        resolved_backends.add(backend_spec.backend)
        if backend_spec.required_extra:
            extras.add(str(backend_spec.required_extra))

    return tuple(sorted(extras)), tuple(sorted(resolved_backends))


def _resolve_device(request: MethodResolutionRequest, *, supports_gpu: bool) -> str:
    if request.strict and request.requested_device == "auto":
        raise PipelineResolutionError(
            "auto_device",
            "method device 'auto' is forbidden in strict execution",
        )
    if request.requested_device != "auto":
        return request.requested_device
    if not supports_gpu:
        return "cpu"
    return str(resolve_device_name("auto") or "cpu")


def resolve_method(request: MethodResolutionRequest) -> MethodRuntimeResolution:
    """Resolve one method without exposing registries or spec introspection to callers."""

    if not isinstance(request, MethodResolutionRequest):
        raise TypeError("request must be a MethodResolutionRequest")
    method_class, method_info = _method_class_and_info(request.regime, request.method_id)
    capabilities = getattr(method_info, "capabilities", None)
    if not isinstance(capabilities, MethodCapabilities):
        raise PipelineResolutionError(
            "method_lookup",
            f"method {request.method_id!r} exposes no valid capability contract",
        )

    method_spec = _materialized_method_spec(request, method_class)
    requested_backend, method_backend = _method_backend(request, method_class=method_class)
    configured_classifier_backend = _classifier_backend(request)
    method_required_extra_value = getattr(method_info, "required_extra", None)
    method_required_extra = (
        str(method_required_extra_value) if method_required_extra_value is not None else None
    )
    required_extras, classifier_backends = _classifier_dependency_facts(
        request,
        method_spec=method_spec,
        method_required_extra=method_required_extra,
    )
    classifier_backend = configured_classifier_backend
    if classifier_backend is None and len(classifier_backends) == 1:
        classifier_backend = classifier_backends[0]
    requires_torch = (
        capabilities.backends == frozenset({"torch"})
        or (method_backend is not None and method_backend.lower() == "torch")
        or (classifier_backend is not None and classifier_backend.lower() == "torch")
        or "torch" in classifier_backends
    )
    if (
        request.strict
        and request.regime == "inductive"
        and requires_torch
        and _TORCH_PREPROCESS_STEP_ID not in request.preprocess_step_ids
    ):
        raise PipelineResolutionError(
            "torch_preprocess",
            "strict torch inductive execution requires the native torch-conversion "
            "preprocessing step",
        )

    resolved_backend = method_backend or classifier_backend
    if resolved_backend is None and requires_torch:
        resolved_backend = "torch"
    supports_gpu = bool(getattr(method_info, "supports_gpu", False))
    return MethodRuntimeResolution(
        regime=request.regime,
        method_id=request.method_id,
        supports_gpu=supports_gpu,
        required_extra=method_required_extra,
        capabilities=capabilities,
        requested_device=request.requested_device,
        resolved_device=_resolve_device(request, supports_gpu=supports_gpu),
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        classifier_backend=classifier_backend,
        requires_torch=requires_torch,
        model_configured=request.model_configured,
        dtype=request.dtype,
        strict=request.strict,
        preprocess_step_ids=tuple(request.preprocess_step_ids),
        required_extras=required_extras,
    )


def validate_materialized_pipeline(
    method: MethodRuntimeResolution,
    pipeline: MaterializedPipeline,
) -> PipelineResolution:
    """Validate exact materialized inputs against a resolved method contract."""

    if not isinstance(method, MethodRuntimeResolution):
        raise TypeError("method must be a MethodRuntimeResolution")
    if not isinstance(pipeline, MaterializedPipeline):
        raise TypeError("pipeline must be a MaterializedPipeline")
    capabilities = materialize_pipeline_capabilities(
        regime=method.regime,
        modality=pipeline.modality,
        primary_input=pipeline.primary_input,
        sampling=pipeline.sampling,
        view_count=pipeline.view_count,
        has_graph=pipeline.has_graph,
        has_weak_augmentation=pipeline.has_weak_augmentation,
        strong_augmentation_count=pipeline.strong_augmentation_count,
        configured_backend=method.resolved_backend,
        model_configured=method.model_configured,
        requires_torch=method.requires_torch,
        device=method.resolved_device,
        dtype=method.dtype,
        checkpointing_required=pipeline.checkpointing_required,
    )
    try:
        compatibility = validate_pipeline_compatibility(
            method.method_id,
            method.capabilities,
            capabilities,
        )
    except IncompatiblePipelineError as exc:
        raise PipelineResolutionError(
            "capability",
            str(exc),
            report=exc.report,
        ) from exc
    return PipelineResolution(
        method=method,
        pipeline_capabilities=capabilities,
        compatibility=compatibility,
    )


def resolve_pipeline(request: PipelineResolutionRequest) -> PipelineResolution:
    """Resolve a registered method and validate its exact materialized pipeline."""

    if not isinstance(request, PipelineResolutionRequest):
        raise TypeError("request must be a PipelineResolutionRequest")
    return validate_materialized_pipeline(resolve_method(request.method), request.pipeline)


__all__ = [
    "MaterializedPipeline",
    "MethodResolutionRequest",
    "MethodRuntimeResolution",
    "PipelineResolution",
    "PipelineResolutionError",
    "PipelineResolutionErrorKind",
    "PipelineResolutionRequest",
    "resolve_method",
    "resolve_pipeline",
    "validate_materialized_pipeline",
]
