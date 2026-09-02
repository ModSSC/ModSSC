"""Native preparation and execution of transductive ModSSC methods.

This module owns the complete runner-independent execution boundary: method
lookup and specification, device/backend resolution, dependency checks,
method-facing data preparation, fitting, and public runtime metadata.
Configuration-file runners only adapt their schema to these native types.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from modssc.capabilities import (
    IncompatiblePipelineError,
    validate_consumed_input_capabilities,
)
from modssc.data_augmentation import validate_augmentation_regime
from modssc.data_augmentation.errors import DataAugmentationValidationError
from modssc.data_loader.types import LoadedDataset
from modssc.graph.artifacts import GraphArtifact
from modssc.runtime.composition import (
    build_execution_contract_report,
    enforce_execution_contract,
    execution_contract_sha256,
)
from modssc.runtime.contracts import ExecutionContractError, ExecutionContractReport
from modssc.runtime.device import resolve_device_name
from modssc.runtime.execution import ExecutionContext
from modssc.runtime.input_contracts import (
    materialize_input_contracts,
    validate_input_contracts,
)
from modssc.runtime.method_contracts import resolve_method_execution_contract
from modssc.runtime.method_spec import MethodSpecError, build_method_spec
from modssc.runtime.outcome import enforce_method_execution

from .data import PreparedNodeData, prepare_node_data, to_numpy
from .errors import TransductiveDataError, TransductiveValidationError
from .registry import get_method_class, get_method_info
from .types import DeviceSpec

TransductiveExecutionErrorKind = Literal[
    "auto_backend",
    "dependency_missing",
    "method_contract",
    "method_introspection",
    "method_spec",
    "data_contract",
    "augmentation_contract",
    "capability",
    "execution_contract",
]

_ERROR_CODES: dict[TransductiveExecutionErrorKind, str] = {
    "auto_backend": "E_TRANSDUCTIVE_AUTO_FORBIDDEN",
    "dependency_missing": "E_TRANSDUCTIVE_DEPENDENCY_MISSING",
    "method_contract": "E_TRANSDUCTIVE_METHOD_CONTRACT",
    "method_introspection": "E_TRANSDUCTIVE_METHOD_INTROSPECTION",
    "method_spec": "E_TRANSDUCTIVE_METHOD_SPEC",
    "data_contract": "E_TRANSDUCTIVE_DATA_CONTRACT",
    "augmentation_contract": "E_TRANSDUCTIVE_AUGMENTATION_UNSUPPORTED",
    "capability": "E_TRANSDUCTIVE_CAPABILITY",
    "execution_contract": "E_TRANSDUCTIVE_EXECUTION_CONTRACT",
}


class TransductiveExecutionError(TransductiveValidationError):
    """Raised when the native transductive execution contract cannot be fulfilled."""

    def __init__(self, kind: TransductiveExecutionErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.code = _ERROR_CODES[kind]


@dataclass(frozen=True)
class TransductiveExecutionConfig:
    """Runner-independent configuration for one transductive method fit."""

    method_id: str
    device: DeviceSpec = field(default_factory=DeviceSpec)
    params: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    strict: bool = False
    use_test_split: bool = False
    expected_labeled_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", str(self.method_id))
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "strict", bool(self.strict))
        object.__setattr__(self, "use_test_split", bool(self.use_test_split))


@dataclass(frozen=True)
class TransductiveExecutionInput:
    """Native upstream artifacts required for transductive execution."""

    dataset: LoadedDataset
    graph: GraphArtifact | None
    masks: Mapping[str, Any]
    augmentation_configured: bool = False
    augmentation: Any | None = None
    routing_events: tuple[Mapping[str, Any], ...] = ()
    execution_context: ExecutionContext | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "masks", dict(self.masks))
        object.__setattr__(self, "augmentation_configured", bool(self.augmentation_configured))
        object.__setattr__(
            self, "routing_events", tuple(dict(event) for event in self.routing_events)
        )


@dataclass(frozen=True)
class TransductiveExecutionResult:
    """Fitted method, isolated data bundle, and public runtime resolution."""

    method: Any
    data: PreparedNodeData
    resolution: Mapping[str, Any]

    @property
    def backend(self) -> str | None:
        value = self.resolution.get("backend")
        return str(value) if value is not None else None

    @property
    def resolved_device(self) -> str | None:
        value = self.resolution.get("resolved_device")
        return str(value) if value is not None else None


def _resolve_method_device(
    config: TransductiveExecutionConfig,
    *,
    supports_gpu: bool,
) -> str | None:
    requested = config.device.device
    if config.strict and requested == "auto":
        raise TransductiveExecutionError(
            "auto_backend",
            "method device 'auto' is forbidden in strict mode",
        )
    if requested != "auto":
        return requested
    if not supports_gpu:
        return "cpu"
    return resolve_device_name(requested)


def _requested_backend(
    config: TransductiveExecutionConfig,
    *,
    spec: Any,
) -> str | None:
    backend = config.params.get("backend")
    if backend is None and spec is not None and hasattr(spec, "backend"):
        backend = spec.backend
    if backend is None:
        return None
    normalized = str(backend)
    if config.strict and normalized.lower() == "auto":
        raise TransductiveExecutionError(
            "auto_backend",
            "method backend 'auto' is forbidden in strict mode",
        )
    return normalized


def _ensure_backend_dependencies(backend: str | None) -> None:
    if backend is None or backend.lower() != "torch":
        return
    try:
        importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise TransductiveExecutionError(
            "dependency_missing",
            "method backend 'torch' requires dependency 'torch'",
        ) from exc


def _dtype_descriptor(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    descriptor: dict[str, Any] = {}
    if dtype is not None:
        descriptor["dtype"] = str(dtype)
    if shape is not None:
        try:
            descriptor["shape"] = list(shape)
        except TypeError:
            descriptor["shape"] = None
    return descriptor or None


def _public_method_resolution(method: Any) -> dict[str, Any]:
    provider = getattr(method, "execution_resolution", None)
    if provider is None:
        return {}
    if not callable(provider):
        raise TransductiveExecutionError(
            "method_contract",
            "method execution_resolution must be callable",
        )
    resolution = provider()
    if not isinstance(resolution, Mapping):
        raise TransductiveExecutionError(
            "method_contract",
            "method execution_resolution() must return a mapping",
        )
    return dict(resolution)


def _resolve_execution_contract(
    *,
    config: TransductiveExecutionConfig,
    method_class: type[Any],
    spec: Any,
    method_info: Any,
    data: Any,
) -> tuple[ExecutionContractReport, str]:
    """Compose the exact method-facing input contract before fit."""

    contract = resolve_method_execution_contract(
        method_class,
        spec,
        method_info.capabilities,
    )
    input_provisions = materialize_input_contracts(
        regime="transductive",
        consumed_input=data,
    )
    input_issues, input_unverified = validate_input_contracts(
        contract.inputs,
        contract.relations,
        input_provisions,
    )
    report = build_execution_contract_report(
        method_id=config.method_id,
        contract=contract,
        input_provisions=input_provisions,
        issues=input_issues,
        unverified=input_unverified,
    )
    digest = execution_contract_sha256(report)
    enforce_execution_contract(report, strict=config.strict)
    return report, digest


def execute_transductive_method(
    inputs: TransductiveExecutionInput,
    config: TransductiveExecutionConfig,
) -> TransductiveExecutionResult:
    """Prepare, build, fit, and resolve one registered transductive method."""

    try:
        validate_augmentation_regime(
            regime="transductive",
            configured=inputs.augmentation_configured or inputs.augmentation is not None,
        )
    except DataAugmentationValidationError as exc:
        raise TransductiveExecutionError("augmentation_contract", str(exc)) from exc

    method_info = get_method_info(config.method_id)
    resolved_device = _resolve_method_device(
        config,
        supports_gpu=method_info.supports_gpu,
    )
    method_class = get_method_class(config.method_id)
    try:
        spec = build_method_spec(
            method_class,
            config.params,
            require_spec=True,
            strict=config.strict,
        )
    except MethodSpecError as exc:
        raise TransductiveExecutionError(exc.kind, str(exc)) from exc

    requested_backend = _requested_backend(config, spec=spec)
    _ensure_backend_dependencies(requested_backend)
    try:
        prepared = prepare_node_data(
            dataset=inputs.dataset,
            graph=inputs.graph,
            masks=inputs.masks,
            use_test_split=config.use_test_split,
            expected_labeled_count=config.expected_labeled_count,
        )
    except TransductiveDataError as exc:
        raise TransductiveExecutionError("data_contract", str(exc)) from exc

    method = method_class(spec) if spec is not None else method_class()
    modality = str(inputs.dataset.meta.get("modality") or "graph")
    checkpointing_required = bool(
        inputs.execution_context is not None and inputs.execution_context.resume_policy != "never"
    )
    advisory_capabilities = replace(
        method_info.capabilities,
        required_classifier_outputs=frozenset(),
    )
    try:
        consumed_capabilities = validate_consumed_input_capabilities(
            config.method_id,
            advisory_capabilities,
            regime="transductive",
            modality=modality,
            consumed_input=prepared.fit,
            runtime_backend=requested_backend,
            device=resolved_device,
            dtype=config.device.dtype,
            checkpointing_required=checkpointing_required,
        )
    except IncompatiblePipelineError as exc:
        raise TransductiveExecutionError("capability", str(exc)) from exc
    try:
        contract_report, contract_sha256 = _resolve_execution_contract(
            config=config,
            method_class=method_class,
            spec=spec,
            method_info=method_info,
            data=prepared.fit,
        )
    except ExecutionContractError as exc:
        raise TransductiveExecutionError("execution_contract", str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise TransductiveExecutionError(
            "execution_contract",
            f"invalid execution contract for method {config.method_id!r}: {exc}",
        ) from exc
    method.fit(prepared.fit, device=resolved_device, seed=config.seed)
    enforce_method_execution(method)

    public_resolution = _public_method_resolution(method)
    runtime_backend = public_resolution.pop("backend", requested_backend)
    resolution: dict[str, Any] = {
        "backend": str(runtime_backend) if runtime_backend is not None else None,
        "classifier_backend": None,
        "resolved_device": resolved_device,
        "dtypes": {
            "X": _dtype_descriptor(prepared.fit.X),
            "y_true": _dtype_descriptor(to_numpy(prepared.evaluation.y_true)),
            "y_obs": _dtype_descriptor(prepared.fit.y),
        },
        "normalization": {
            "implicit_method_conversion": False,
            "strict_contract_validated": config.strict,
        },
        "pipeline_capabilities": consumed_capabilities.to_dict(),
        "execution_contract": contract_report.to_dict(),
        "execution_contract_sha256": contract_sha256,
        "input_routing": [dict(event) for event in inputs.routing_events],
        **public_resolution,
    }
    diagnostics = getattr(method, "diagnostics_", None)
    if isinstance(diagnostics, Mapping):
        resolution["diagnostics"] = dict(diagnostics)
    return TransductiveExecutionResult(
        method=method,
        data=prepared,
        resolution=resolution,
    )


__all__ = [
    "TransductiveExecutionConfig",
    "TransductiveExecutionError",
    "TransductiveExecutionErrorKind",
    "TransductiveExecutionInput",
    "TransductiveExecutionResult",
    "execute_transductive_method",
]
