"""Native preparation and execution of an inductive ModSSC method.

This module is the pipeline boundary for inductive methods.  It consumes
native preprocessing, sampling, view, augmentation, and execution artifacts;
materializes the method-facing :class:`InductiveDataset`; binds the model; and
runs ``fit``.  Configuration-file runners are intentionally kept out of this
module so the same behavior is available to the CLI and programmatic users.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from modssc.capabilities import (
    IncompatiblePipelineError,
    validate_consumed_input_capabilities,
)
from modssc.data_augmentation.utils import is_torch_tensor
from modssc.data_loader.selection import select_rows
from modssc.evaluation.runtime import MethodEvaluationRuntime
from modssc.preprocess.types import PreprocessResult
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
from modssc.sampling.errors import SamplingValidationError
from modssc.sampling.result import SamplingResult
from modssc.sampling.routing import (
    InductiveGraphSamplingPolicy,
    SamplingRoutingEvent,
    route_sampling_for_regime,
)
from modssc.views.types import ViewsResult

from .errors import InductiveValidationError
from .model_binding import (
    ModelBindingError,
    ModelBuildConfig,
    bind_model_to_spec,
)
from .model_contracts import (
    resolve_bound_component_contracts,
    validate_component_contracts,
)
from .registry import get_method_class, get_method_info
from .types import DeviceSpec, InductiveDataset

InductiveExecutionErrorKind = Literal[
    "graph_sampling",
    "auto_backend",
    "dependency_missing",
    "labels_contract",
    "evaluation_split",
    "torch_required",
    "shape",
    "dtype",
    "method_contract",
    "method_introspection",
    "method_spec",
    "model_config",
    "capability",
    "execution_contract",
    "graph_contract",
]

_ERROR_CODES: dict[InductiveExecutionErrorKind, str] = {
    "graph_sampling": "E_INDUCTIVE_GRAPH_SAMPLING_INVALID",
    "auto_backend": "E_INDUCTIVE_AUTO_FORBIDDEN",
    "dependency_missing": "E_INDUCTIVE_DEPENDENCY_MISSING",
    "labels_contract": "E_INDUCTIVE_LABELS_CONTRACT",
    "evaluation_split": "E_INDUCTIVE_EVAL_SPLIT_INVALID",
    "torch_required": "E_INDUCTIVE_PREPROCESS_TO_TORCH_REQUIRED",
    "shape": "E_INDUCTIVE_SHAPE_CONTRACT",
    "dtype": "E_INDUCTIVE_DTYPE_CONTRACT",
    "method_contract": "E_INDUCTIVE_METHOD_CONTRACT",
    "method_introspection": "E_INDUCTIVE_METHOD_INTROSPECTION",
    "method_spec": "E_INDUCTIVE_METHOD_SPEC",
    "model_config": "E_INDUCTIVE_MODEL_CONFIG",
    "capability": "E_INDUCTIVE_CAPABILITY",
    "execution_contract": "E_INDUCTIVE_EXECUTION_CONTRACT",
    "graph_contract": "E_INDUCTIVE_GRAPH_CONTRACT",
}


class InductiveExecutionError(InductiveValidationError):
    """Raised when the native inductive execution contract cannot be fulfilled."""

    def __init__(self, kind: InductiveExecutionErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.code = _ERROR_CODES[kind]


@dataclass(frozen=True)
class InductiveExecutionConfig:
    """Runner-independent configuration for one inductive method fit."""

    method_id: str
    device: DeviceSpec = field(default_factory=DeviceSpec)
    params: Mapping[str, Any] = field(default_factory=dict)
    model: ModelBuildConfig | None = None
    seed: int = 0
    strict: bool = False
    requires_torch: bool = False
    during_fit_splits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", str(self.method_id))
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "strict", bool(self.strict))
        object.__setattr__(self, "requires_torch", bool(self.requires_torch))
        object.__setattr__(
            self,
            "during_fit_splits",
            tuple(str(split) for split in self.during_fit_splits),
        )


@dataclass(frozen=True)
class InductiveExecutionInput:
    """Native upstream artifacts required to prepare an inductive dataset."""

    preprocess: PreprocessResult
    sampling: SamplingResult
    views: ViewsResult | None = None
    X_u_w: Any | None = None
    X_u_s: Any | None = None
    X_u_s_1: Any | None = None
    online_augmentation: Any | None = None
    graph: Any | None = None
    graph_sampling_policy: InductiveGraphSamplingPolicy | str = InductiveGraphSamplingPolicy.REJECT
    routing_events: tuple[SamplingRoutingEvent, ...] = ()
    execution_context: ExecutionContext | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "routing_events", tuple(self.routing_events))


@dataclass(frozen=True)
class InductiveExecutionResult:
    """Fitted method, exact method-facing data, and resolved runtime metadata."""

    method: Any
    data: InductiveDataset
    resolution: Mapping[str, Any]
    evaluation_runtime: MethodEvaluationRuntime | None = None


@dataclass(frozen=True)
class _PreparedInputs:
    sampling: SamplingResult
    routing_events: tuple[SamplingRoutingEvent, ...]
    X_train: Any
    labeled_indices: np.ndarray
    unlabeled_indices: np.ndarray
    X_l: Any
    y_l: Any
    X_u: Any | None
    X_u_w: Any | None
    X_u_s: Any | None
    X_u_s_1: Any | None
    views: Mapping[str, Any] | None


def _torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise InductiveExecutionError(
            "dependency_missing",
            "torch-backed inductive execution requires dependency 'torch'",
        ) from exc


def _array_backend_flags(value: Any) -> tuple[bool, bool]:
    if is_torch_tensor(value):
        return True, False
    if isinstance(value, Mapping):
        children = value.values()
        treat_unknown_as_numpy = False
    elif isinstance(value, (list, tuple, set)):
        children = value
        treat_unknown_as_numpy = True
    elif isinstance(value, np.ndarray):
        return False, True
    else:
        return False, False
    has_torch = False
    has_numpy = False
    for child in children:
        child_torch, child_numpy = _array_backend_flags(child)
        has_torch = has_torch or child_torch
        has_numpy = has_numpy or child_numpy
        if treat_unknown_as_numpy and not child_torch and not child_numpy:
            has_numpy = True
    return has_torch, has_numpy


def _is_torch_container(value: Any) -> bool:
    if is_torch_tensor(value):
        return True
    if isinstance(value, Mapping):
        has_torch, has_numpy = _array_backend_flags(value)
        return has_torch and not has_numpy
    return False


def _torch_container_device(value: Any) -> Any | None:
    if is_torch_tensor(value):
        return value.device
    if isinstance(value, Mapping):
        if "x" in value and is_torch_tensor(value["x"]):
            return value["x"].device
        children = value.values()
    elif isinstance(value, (list, tuple, set)):
        children = value
    else:
        return None
    for child in children:
        device = _torch_container_device(child)
        if device is not None:
            return device
    return None


def _feature_tensor(value: Any) -> Any | None:
    if value is None or is_torch_tensor(value) or isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping) and "x" in value:
        return value.get("x")
    return None


def _leading_size(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, Mapping) and "x" in value:
        value = value.get("x")
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    try:
        return int(shape[0])
    except (TypeError, ValueError):
        return None


def _indices_for(value: Any, indices: np.ndarray) -> Any:
    if not _is_torch_container(value):
        return indices
    torch = _torch_module()
    device = _torch_container_device(value) or "cpu"
    return torch.as_tensor(indices, device=device, dtype=torch.long)


def _smart_to_torch(value: Any, device: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: _smart_to_torch(child, device) for key, child in value.items()}
    if is_torch_tensor(value):
        if device is not None and value.device != device and hasattr(value, "to"):
            return value.to(device)
        return value
    torch = _torch_module()
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return torch.tensor(array, device=device, dtype=torch.float32).div_(255.0)
    dtype = torch.float32 if array.dtype == np.float64 else None
    return torch.as_tensor(array, device=device, dtype=dtype)


def requires_torch_inputs(config: InductiveExecutionConfig) -> bool:
    """Return whether the declared native method contract requires torch inputs."""

    if config.requires_torch:
        return True
    if config.model is not None and str(config.model.classifier_backend or "").lower() == "torch":
        return True
    classifier_backend = config.params.get("classifier_backend")
    if isinstance(classifier_backend, str) and classifier_backend.lower() == "torch":
        return True
    backend = config.params.get("backend")
    return isinstance(backend, str) and backend.lower() == "torch"


def _labels_for_backend(
    preprocess: PreprocessResult,
    X_l: Any,
    indices: np.ndarray,
    *,
    strict: bool,
) -> Any:
    labels = (
        preprocess.train_artifacts.get("labels.y")
        if preprocess.train_artifacts.has("labels.y")
        else None
    )
    source = labels if labels is not None else preprocess.dataset.train.y
    index_max = int(indices.max()) if indices.size else -1

    if labels is not None and index_max >= 0:
        source_size = _leading_size(labels)
        if source_size is not None and source_size <= index_max:
            if strict:
                raise InductiveExecutionError(
                    "labels_contract",
                    "preprocess labels.y is shorter than required labeled indices; "
                    "no strict fallback",
                )
            source = preprocess.dataset.train.y

    if is_torch_tensor(source):
        subset = source
        if getattr(subset, "ndim", 0) > 0 and subset.shape[0] > index_max:
            subset = subset[_indices_for(subset, indices)]
        if _is_torch_container(X_l):
            device = _torch_container_device(X_l) or "cpu"
            if subset.device != device:
                subset = subset.to(device)
        return subset

    labels_array = np.asarray(source)
    if labels_array.dtype == np.object_:
        labels_array = np.array(
            [-1 if value is None else value for value in labels_array.tolist()],
            dtype=np.int64,
        )
    else:
        labels_array = labels_array.astype(np.int64, copy=False)
    if labels_array.ndim > 0 and labels_array.shape[0] > index_max:
        labels_array = labels_array[indices]
    if _is_torch_container(X_l):
        torch = _torch_module()
        device = _torch_container_device(X_l) or "cpu"
        return torch.as_tensor(labels_array, device=device, dtype=torch.int64)
    return labels_array


def _validate_tensor_contract(name: str, value: Any) -> None:
    tensor = _feature_tensor(value)
    if tensor is None:
        return
    ndim = getattr(tensor, "ndim", None)
    if isinstance(ndim, int) and ndim < 2:
        raise InductiveExecutionError(
            "shape",
            f"{name} must be at least 2D, got ndim={ndim}",
        )
    if is_torch_tensor(tensor) and not _torch_module().is_floating_point(tensor):
        raise InductiveExecutionError(
            "dtype",
            f"{name} must be a floating torch tensor in strict mode; got dtype={tensor.dtype}",
        )


def _validate_strict_inputs(
    *,
    X_l: Any,
    y_l: Any,
    X_u: Any | None,
    X_u_w: Any | None,
    X_u_s: Any | None,
    X_u_s_1: Any | None,
    requires_torch: bool,
) -> None:
    for name, value in (
        ("X_l", X_l),
        ("X_u", X_u),
        ("X_u_w", X_u_w),
        ("X_u_s", X_u_s),
        ("X_u_s_1", X_u_s_1),
    ):
        if value is None:
            continue
        if requires_torch and not _is_torch_container(value):
            raise InductiveExecutionError(
                "torch_required",
                f"{name} must be torch-backed in strict mode (declare conversion in preprocessing)",
            )
        _validate_tensor_contract(name, value)

    X_l_size = _leading_size(X_l)
    y_l_size = _leading_size(y_l)
    if X_l_size is not None and y_l_size is not None and X_l_size != y_l_size:
        raise InductiveExecutionError(
            "shape",
            f"X_l/y_l row mismatch: X_l={X_l_size} y_l={y_l_size}",
        )


def _views_for_backend(
    views: ViewsResult,
    *,
    labeled_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    backend_reference: Any,
    strict: bool,
) -> dict[str, Any]:
    use_torch = _is_torch_container(backend_reference)
    device = _torch_container_device(backend_reference) if use_torch else None
    payload: dict[str, Any] = {}
    for name, dataset in views.views.items():
        X_l = select_rows(
            dataset.train.X,
            labeled_indices,
            context=f"inductive.views[{name}].labeled",
        )
        X_u = select_rows(
            dataset.train.X,
            unlabeled_indices,
            context=f"inductive.views[{name}].unlabeled",
        )
        if use_torch:
            if strict and (not _is_torch_container(X_l) or not _is_torch_container(X_u)):
                raise InductiveExecutionError(
                    "torch_required",
                    f"view {name!r} must be torch-backed in strict mode",
                )
            if not strict:
                X_l = _smart_to_torch(X_l, device)
                X_u = _smart_to_torch(X_u, device)
        payload[name] = {"X_l": X_l, "X_u": X_u}
    return payload


def _evaluation_splits_for_backend(
    preprocess: PreprocessResult,
    sampling: SamplingResult,
    *,
    splits: tuple[str, ...],
    backend_reference: Any,
    strict: bool,
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    destination = (
        _torch_container_device(backend_reference)
        if _is_torch_container(backend_reference)
        else None
    )
    for split in splits:
        if split not in sampling.indices:
            raise InductiveExecutionError(
                "evaluation_split",
                f"during-fit evaluation split {split!r} is absent from sampling",
            )
        reference = sampling.refs.get(split, "train")
        if reference == "train":
            base = preprocess.dataset.train
            store = preprocess.train_artifacts
        elif reference == "test" and preprocess.dataset.test is not None:
            base = preprocess.dataset.test
            store = preprocess.test_artifacts
        else:
            raise InductiveExecutionError(
                "evaluation_split",
                f"during-fit evaluation split {split!r} has no source dataset split",
            )
        indices = np.asarray(sampling.indices[split], dtype=np.int64)
        X = select_rows(base.X, indices, context=f"inductive.evaluation.{split}.X")
        labels = store.get("labels.y") if store is not None and store.has("labels.y") else base.y
        y = select_rows(labels, indices, context=f"inductive.evaluation.{split}.y")
        if destination is not None:
            if strict and (not _is_torch_container(X) or not is_torch_tensor(y)):
                raise InductiveExecutionError(
                    "torch_required",
                    f"during-fit evaluation split {split!r} is not torch-backed",
                )
            if not strict:
                X = _smart_to_torch(X, destination)
                y = _smart_to_torch(y, destination)
        payloads[split] = {"X": X, "y": y}
    return payloads


def _partition_artifact_sha256(stats: Mapping[str, Any]) -> Any | None:
    policy = stats.get("policy")
    if isinstance(policy, Mapping):
        nested = policy.get("partition_artifact_sha256")
        if nested is not None:
            return nested
    return stats.get("partition_artifact_sha256")


def _unlabeled_index_space(method: Any) -> Literal["source", "local"]:
    value = getattr(method, "unlabeled_index_space", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(getattr(method, "info", None), "unlabeled_index_space", "source")
    if value not in {"source", "local"}:
        raise InductiveExecutionError(
            "method_contract",
            "method unlabeled_index_space must be 'source' or 'local'",
        )
    return value


def _dtype_descriptor(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = value["x"] if isinstance(value, Mapping) and "x" in value else value
    dtype = getattr(payload, "dtype", None)
    shape = getattr(payload, "shape", None)
    descriptor: dict[str, Any] = {}
    if dtype is not None:
        descriptor["dtype"] = str(dtype)
    if shape is not None:
        try:
            descriptor["shape"] = list(shape)
        except TypeError:
            descriptor["shape"] = None
    return descriptor or None


def _method_backend(config: InductiveExecutionConfig, method: Any, spec: Any) -> str | None:
    backend = config.params.get("backend")
    if backend is None and spec is not None and hasattr(spec, "backend"):
        backend = spec.backend
    if backend is None:
        backend = getattr(method, "_backend", None)
    return str(backend) if backend is not None else None


def _build_method(
    config: InductiveExecutionConfig,
    *,
    X_l: Any,
    y_l: Any,
) -> tuple[Any, Any]:
    method_class = get_method_class(config.method_id)
    method_info = get_method_info(config.method_id)
    try:
        spec = build_method_spec(
            method_class,
            config.params,
            require_spec=(config.model is not None or method_info.model_binding.kind != "none"),
            strict=config.strict,
        )
        spec = bind_model_to_spec(
            spec,
            config.model,
            binding=method_info.model_binding,
            X_l=X_l,
            y_l=y_l,
            default_ema=bool(method_info.default_model_ema),
            seed=config.seed,
            strict=config.strict,
        )
    except MethodSpecError as exc:
        raise InductiveExecutionError(exc.kind, str(exc)) from exc
    except ModelBindingError as exc:
        raise InductiveExecutionError(exc.kind, str(exc)) from exc
    return (method_class(spec) if spec is not None else method_class()), spec


def _prepare_inputs(
    inputs: InductiveExecutionInput,
    config: InductiveExecutionConfig,
) -> _PreparedInputs:
    try:
        routed_sampling = route_sampling_for_regime(
            inputs.sampling,
            regime="inductive",
            inductive_graph_policy=inputs.graph_sampling_policy,
        )
    except SamplingValidationError as exc:
        raise InductiveExecutionError("graph_sampling", str(exc)) from exc
    sampling = routed_sampling.sampling
    preprocess = inputs.preprocess

    requested_backend = config.params.get("backend")
    if config.strict and isinstance(requested_backend, str) and requested_backend.lower() == "auto":
        raise InductiveExecutionError(
            "auto_backend",
            "method backend 'auto' is forbidden in strict mode",
        )
    if isinstance(requested_backend, str) and requested_backend.lower() == "torch":
        _torch_module()

    labeled_indices = np.asarray(sampling.indices["train_labeled"], dtype=np.int64)
    unlabeled_indices = np.asarray(sampling.indices["train_unlabeled"], dtype=np.int64)
    X_train = preprocess.dataset.train.X
    X_l = select_rows(X_train, labeled_indices, context="inductive.train_labeled")
    X_u = select_rows(X_train, unlabeled_indices, context="inductive.train_unlabeled")
    X_u_w, X_u_s, X_u_s_1 = inputs.X_u_w, inputs.X_u_s, inputs.X_u_s_1

    torch_inputs = requires_torch_inputs(config)
    if not config.strict and torch_inputs:
        target_device = (
            _torch_container_device(X_l)
            if _is_torch_container(X_l)
            else resolve_device_name(config.device.device)
        )
        X_l = _smart_to_torch(X_l, target_device)
        X_u = _smart_to_torch(X_u, target_device)
        X_u_w = _smart_to_torch(X_u_w, target_device)
        X_u_s = _smart_to_torch(X_u_s, target_device)
        X_u_s_1 = _smart_to_torch(X_u_s_1, target_device)

    y_l = _labels_for_backend(preprocess, X_l, labeled_indices, strict=config.strict)
    if config.strict:
        _validate_strict_inputs(
            X_l=X_l,
            y_l=y_l,
            X_u=X_u,
            X_u_w=X_u_w,
            X_u_s=X_u_s,
            X_u_s_1=X_u_s_1,
            requires_torch=config.requires_torch,
        )

    views_payload = (
        _views_for_backend(
            inputs.views,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            backend_reference=X_l,
            strict=config.strict,
        )
        if inputs.views is not None
        else None
    )
    if X_u_s_1 is not None:
        views_payload = dict(views_payload or {})
        views_payload["X_u_s_1"] = X_u_s_1

    return _PreparedInputs(
        sampling=sampling,
        routing_events=tuple(inputs.routing_events) + routed_sampling.events,
        X_train=X_train,
        labeled_indices=labeled_indices,
        unlabeled_indices=unlabeled_indices,
        X_l=X_l,
        y_l=y_l,
        X_u=X_u,
        X_u_w=X_u_w,
        X_u_s=X_u_s,
        X_u_s_1=X_u_s_1,
        views=views_payload,
    )


def _assemble_inductive_dataset(
    inputs: InductiveExecutionInput,
    config: InductiveExecutionConfig,
    *,
    method: Any,
    prepared: _PreparedInputs,
) -> InductiveDataset:
    preprocess = inputs.preprocess
    sampling = prepared.sampling
    X_train = prepared.X_train
    labeled_indices = prepared.labeled_indices
    unlabeled_indices = prepared.unlabeled_indices
    X_l = prepared.X_l
    X_u = prepared.X_u

    index_space = _unlabeled_index_space(method)
    meta: dict[str, Any] = {
        "dataset_fingerprint": preprocess.dataset.meta.get("dataset_fingerprint"),
        "split_fingerprint": sampling.split_fingerprint,
        "partition_sha256": _partition_artifact_sha256(sampling.stats),
        "preprocess_fingerprint": preprocess.preprocess_fingerprint,
        "idx_l": _indices_for(X_l, labeled_indices),
        "source_idx_l": _indices_for(X_l, labeled_indices),
    }
    if X_u is not None:
        source_idx_u = _indices_for(X_u, unlabeled_indices)
        meta["source_idx_u"] = source_idx_u
        if index_space == "local":
            local_idx_u = np.arange(int(unlabeled_indices.size), dtype=np.int64)
            meta["idx_u"] = _indices_for(X_u, local_idx_u)
            meta["ulb_size"] = int(unlabeled_indices.size)
        else:
            source_size = _leading_size(X_train)
            if source_size is None:
                raise InductiveExecutionError(
                    "method_contract",
                    "cannot determine source population size for method indices",
                )
            meta["idx_u"] = source_idx_u
            meta["ulb_size"] = source_size
    if inputs.online_augmentation is not None:
        meta["online_augmentation"] = inputs.online_augmentation
        meta["augmentation_seed"] = int(getattr(inputs.online_augmentation, "seed", config.seed))
    if config.during_fit_splits:
        meta["evaluation_splits"] = _evaluation_splits_for_backend(
            preprocess,
            sampling,
            splits=config.during_fit_splits,
            backend_reference=X_l,
            strict=config.strict,
        )
    if prepared.routing_events:
        meta["input_routing"] = [event.to_dict() for event in prepared.routing_events]
    if inputs.graph is not None:
        graph_n_nodes = getattr(inputs.graph, "n_nodes", None)
        if isinstance(graph_n_nodes, bool) or not isinstance(graph_n_nodes, (int, np.integer)):
            raise InductiveExecutionError(
                "graph_contract",
                "an inductive graph artifact must expose an integer n_nodes",
            )
        graph_n_nodes = int(graph_n_nodes)
        source_size = _leading_size(X_train)
        if source_size is not None and graph_n_nodes < source_size:
            raise InductiveExecutionError(
                "graph_contract",
                "inductive graph has fewer nodes than the method source population: "
                f"graph={graph_n_nodes} source={source_size}",
            )
        graph_meta = getattr(inputs.graph, "meta", None)
        if isinstance(graph_meta, Mapping) and graph_meta.get("fingerprint") is not None:
            meta["graph_fingerprint"] = graph_meta["fingerprint"]

    return InductiveDataset(
        X_l=X_l,
        y_l=prepared.y_l,
        X_u=X_u,
        X_u_w=prepared.X_u_w,
        X_u_s=prepared.X_u_s,
        X_u_s_1=prepared.X_u_s_1,
        views=prepared.views,
        graph=inputs.graph,
        meta=meta,
        execution_context=inputs.execution_context,
    )


def prepare_inductive_dataset(
    inputs: InductiveExecutionInput,
    config: InductiveExecutionConfig,
    *,
    method: Any,
) -> InductiveDataset:
    """Materialize the exact dataset contract consumed by an inductive method."""

    prepared = _prepare_inputs(inputs, config)
    return _assemble_inductive_dataset(
        inputs,
        config,
        method=method,
        prepared=prepared,
    )


def _resolve_execution_contract(
    *,
    config: InductiveExecutionConfig,
    method: Any,
    spec: Any,
    method_info: Any,
    data: InductiveDataset,
) -> tuple[ExecutionContractReport, str]:
    """Compose the exact method, input, and bound-model contract before fit."""

    contract = resolve_method_execution_contract(
        type(method),
        spec,
        method_info.capabilities,
        method_info.model_binding,
    )
    input_provisions = materialize_input_contracts(
        regime="inductive",
        consumed_input=data,
    )
    input_issues, input_unverified = validate_input_contracts(
        contract.inputs,
        contract.relations,
        input_provisions,
    )
    component_provisions = resolve_bound_component_contracts(
        spec,
        method_info.model_binding,
    )
    component_issues, component_unverified = validate_component_contracts(
        contract.components,
        contract.component_relations,
        component_provisions,
        input_provisions=input_provisions,
        optional_input_roles=(
            requirement.role for requirement in contract.inputs if requirement.optional
        ),
    )
    report = build_execution_contract_report(
        method_id=config.method_id,
        contract=contract,
        input_provisions=input_provisions,
        component_provisions=component_provisions,
        issues=(*input_issues, *component_issues),
        unverified=(*input_unverified, *component_unverified),
    )
    digest = execution_contract_sha256(report)
    enforce_execution_contract(report, strict=config.strict)
    return report, digest


def execute_inductive_method(
    inputs: InductiveExecutionInput,
    config: InductiveExecutionConfig,
) -> InductiveExecutionResult:
    """Prepare, build, fit, and resolve one registered inductive method."""

    prepared = _prepare_inputs(inputs, config)
    method, spec = _build_method(config, X_l=prepared.X_l, y_l=prepared.y_l)
    method_info = get_method_info(config.method_id)
    data = _assemble_inductive_dataset(
        inputs,
        config,
        method=method,
        prepared=prepared,
    )
    # The flat facade remains an availability check.  Model outputs are proved
    # below from the bound component contracts; configuring a model is not, by
    # itself, evidence that it exposes logits or features.
    classifier_outputs = {"predictions", "scores"}
    advisory_capabilities = replace(
        method_info.capabilities,
        required_classifier_outputs=frozenset(),
    )
    checkpointing_required = bool(
        inputs.execution_context is not None and inputs.execution_context.resume_policy != "never"
    )
    modality = str(inputs.preprocess.dataset.meta.get("modality") or "unknown")
    try:
        consumed_capabilities = validate_consumed_input_capabilities(
            config.method_id,
            advisory_capabilities,
            regime="inductive",
            modality=modality,
            consumed_input=data,
            classifier_outputs=classifier_outputs,
            runtime_backend=_method_backend(config, method, spec),
            device=str(_torch_container_device(data.X_l) or config.device.device),
            dtype=config.device.dtype,
            checkpointing_required=checkpointing_required,
        )
    except IncompatiblePipelineError as exc:
        raise InductiveExecutionError("capability", str(exc)) from exc
    try:
        contract_report, contract_sha256 = _resolve_execution_contract(
            config=config,
            method=method,
            spec=spec,
            method_info=method_info,
            data=data,
        )
    except ExecutionContractError as exc:
        raise InductiveExecutionError("execution_contract", str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise InductiveExecutionError(
            "execution_contract",
            f"invalid execution contract for method {config.method_id!r}: {exc}",
        ) from exc
    method.fit(data, device=config.device, seed=config.seed)
    enforce_method_execution(method)

    # Derive the prediction runtime from the exact method-facing features.  It
    # is public fitted state, so evaluation never needs to inspect private
    # classifiers or model bundles to discover their backend/device.
    evaluation_runtime = MethodEvaluationRuntime.from_features(data.X_l)
    try:
        method.evaluation_runtime_ = evaluation_runtime
    except (AttributeError, TypeError) as exc:
        raise InductiveExecutionError(
            "method_contract",
            "a fitted method must allow the native evaluation_runtime_ contract",
        ) from exc

    resolution: dict[str, Any] = {
        "backend": _method_backend(config, method, spec),
        "classifier_backend": (
            config.model.classifier_backend
            if config.model is not None
            else getattr(method, "_classifier_backend", None)
        ),
        "dtypes": {
            "X_l": _dtype_descriptor(data.X_l),
            "y_l": _dtype_descriptor(data.y_l),
            "X_u": _dtype_descriptor(data.X_u),
            "X_u_w": _dtype_descriptor(data.X_u_w),
            "X_u_s": _dtype_descriptor(data.X_u_s),
            "X_u_s_1": _dtype_descriptor(
                data.views.get("X_u_s_1") if data.views is not None else None
            ),
        },
        "normalization": {
            "implicit_method_conversion": False,
            "strict_contract_validated": config.strict,
        },
        "pipeline_capabilities": consumed_capabilities.to_dict(),
        "execution_contract": contract_report.to_dict(),
        "execution_contract_sha256": contract_sha256,
        "input_routing": [event.to_dict() for event in prepared.routing_events],
    }
    diagnostics = getattr(method, "diagnostics_", None)
    if isinstance(diagnostics, Mapping):
        resolution["diagnostics"] = dict(diagnostics)
    return InductiveExecutionResult(
        method=method,
        data=data,
        resolution=resolution,
        evaluation_runtime=evaluation_runtime,
    )


__all__ = [
    "InductiveExecutionConfig",
    "InductiveExecutionError",
    "InductiveExecutionErrorKind",
    "InductiveExecutionInput",
    "InductiveExecutionResult",
    "execute_inductive_method",
    "prepare_inductive_dataset",
    "requires_torch_inputs",
]
