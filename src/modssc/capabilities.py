"""Method-to-pipeline compatibility contracts.

This module is deliberately independent from the benchmark runner.  Methods
declare what they can consume; a caller describes what a materialized pipeline
provides and validates the two before any training starts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

LearningRegime = Literal["inductive", "transductive"]

_REGIMES = frozenset({"inductive", "transductive"})


def _normalize_name(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _normalize_optional_name(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    return _normalize_name(value, field=field)


def _normalize_names(
    values: Iterable[str] | None,
    *,
    field: str,
    allow_none: bool,
) -> frozenset[str] | None:
    if values is None:
        if allow_none:
            return None
        return frozenset()
    if isinstance(values, str):
        raise ValueError(f"{field} must be a collection of names, not a string")
    normalized = frozenset(_normalize_name(value, field=field) for value in values)
    if allow_none and not normalized:
        raise ValueError(f"{field} cannot be empty; use None to mean unrestricted")
    return normalized


def _validate_count(value: int, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


@dataclass(frozen=True)
class MethodCapabilities:
    """Requirements and execution features declared by a method.

    ``modalities`` and ``representations`` use ``None`` to mean that the method
    is generic for that dimension.  Names are stable identifiers owned by the
    corresponding ModSSC registries, not paper or campaign names.
    """

    regime: LearningRegime
    modalities: frozenset[str] | None = None
    representations: frozenset[str] | None = None
    target_kinds: frozenset[str] | None = None
    min_labeled_classes: int | None = None
    max_labeled_classes: int | None = None
    requires_unlabeled: bool = False
    requires_graph: bool = False
    min_views: int = 0
    requires_weak_augmentation: bool = False
    min_strong_augmentations: int = 0
    required_classifier_outputs: frozenset[str] = frozenset()
    backends: frozenset[str] | None = None
    devices: frozenset[str] | None = None
    dtypes: frozenset[str] | None = None
    supports_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.regime not in _REGIMES:
            raise ValueError(f"regime must be one of {sorted(_REGIMES)!r}")
        object.__setattr__(
            self,
            "modalities",
            _normalize_names(self.modalities, field="modalities", allow_none=True),
        )
        object.__setattr__(
            self,
            "representations",
            _normalize_names(
                self.representations,
                field="representations",
                allow_none=True,
            ),
        )
        object.__setattr__(
            self,
            "target_kinds",
            _normalize_names(
                self.target_kinds,
                field="target_kinds",
                allow_none=True,
            ),
        )
        object.__setattr__(
            self,
            "required_classifier_outputs",
            _normalize_names(
                self.required_classifier_outputs,
                field="required_classifier_outputs",
                allow_none=False,
            ),
        )
        for field in ("backends", "devices", "dtypes"):
            object.__setattr__(
                self,
                field,
                _normalize_names(getattr(self, field), field=field, allow_none=True),
            )
        _validate_count(self.min_views, field="min_views")
        _validate_count(self.min_strong_augmentations, field="min_strong_augmentations")
        for field in ("min_labeled_classes", "max_labeled_classes"):
            value = getattr(self, field)
            if value is not None:
                _validate_count(value, field=field)
        if (
            self.min_labeled_classes is not None
            and self.max_labeled_classes is not None
            and self.min_labeled_classes > self.max_labeled_classes
        ):
            raise ValueError("min_labeled_classes cannot exceed max_labeled_classes")


@dataclass(frozen=True)
class PipelineCapabilities:
    """Capabilities provided by a fully resolved data/model pipeline."""

    regime: LearningRegime
    modality: str
    representation: str
    target_kind: str | None = None
    labeled_class_count: int | None = None
    has_unlabeled: bool = False
    has_graph: bool = False
    view_count: int = 0
    has_weak_augmentation: bool = False
    strong_augmentation_count: int = 0
    classifier_outputs: frozenset[str] = frozenset()
    backend: str | None = None
    device: str | None = None
    dtype: str | None = None
    checkpointing_required: bool = False

    def __post_init__(self) -> None:
        if self.regime not in _REGIMES:
            raise ValueError(f"regime must be one of {sorted(_REGIMES)!r}")
        object.__setattr__(self, "modality", _normalize_name(self.modality, field="modality"))
        object.__setattr__(
            self,
            "representation",
            _normalize_name(self.representation, field="representation"),
        )
        object.__setattr__(
            self,
            "target_kind",
            _normalize_optional_name(self.target_kind, field="target_kind"),
        )
        object.__setattr__(
            self,
            "classifier_outputs",
            _normalize_names(
                self.classifier_outputs,
                field="classifier_outputs",
                allow_none=False,
            ),
        )
        for field in ("backend", "device", "dtype"):
            object.__setattr__(
                self,
                field,
                _normalize_optional_name(getattr(self, field), field=field),
            )
        _validate_count(self.view_count, field="view_count")
        _validate_count(self.strong_augmentation_count, field="strong_augmentation_count")
        if self.labeled_class_count is not None:
            _validate_count(self.labeled_class_count, field="labeled_class_count")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible description."""

        return {
            "regime": self.regime,
            "modality": self.modality,
            "representation": self.representation,
            "target_kind": self.target_kind,
            "labeled_class_count": self.labeled_class_count,
            "has_unlabeled": self.has_unlabeled,
            "has_graph": self.has_graph,
            "view_count": self.view_count,
            "has_weak_augmentation": self.has_weak_augmentation,
            "strong_augmentation_count": self.strong_augmentation_count,
            "classifier_outputs": sorted(self.classifier_outputs),
            "backend": self.backend,
            "device": self.device,
            "dtype": self.dtype,
            "checkpointing_required": self.checkpointing_required,
        }


@dataclass(frozen=True)
class CapabilityIssue:
    """One stable, machine-readable incompatibility."""

    code: str
    message: str


@dataclass(frozen=True)
class CompatibilityReport:
    """Complete compatibility result; all detected issues are reported together."""

    method_id: str
    issues: tuple[CapabilityIssue, ...]

    @property
    def compatible(self) -> bool:
        return not self.issues


class IncompatiblePipelineError(ValueError):
    """Raised when a resolved pipeline cannot satisfy a method contract."""

    def __init__(self, report: CompatibilityReport) -> None:
        self.report = report
        details = "; ".join(f"[{issue.code}] {issue.message}" for issue in report.issues)
        super().__init__(f"Pipeline is incompatible with method {report.method_id!r}: {details}")


def _representation_of(value: Any, *, modality: str) -> str:
    if isinstance(value, Mapping):
        keys = set(value)
        if {"input_ids", "attention_mask"} & keys:
            return "tokens"
        if "x" in value:
            return _representation_of(value["x"], modality=modality)
        return "structured"
    if hasattr(value, "tocsr") and hasattr(value, "nnz"):
        return "sparse"
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "S", "U"}:
            return "text" if modality == "text" else "objects"
        return "dense"
    module = type(value).__module__.split(".", 1)[0]
    if module == "torch" and hasattr(value, "shape"):
        layout = str(getattr(value, "layout", ""))
        if bool(getattr(value, "is_sparse", False)) or layout.startswith("torch.sparse"):
            return "sparse"
        return "dense"
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        return "text" if modality == "text" else "paths"
    if hasattr(value, "shape"):
        return "dense"
    return "objects"


def _backend_of(value: Any) -> str | None:
    if isinstance(value, Mapping):
        if "x" in value:
            return _backend_of(value["x"])
        backends = {_backend_of(item) for item in value.values()}
        backends.discard(None)
        return next(iter(backends)) if len(backends) == 1 else None
    module = type(value).__module__.split(".", 1)[0]
    if module == "torch":
        return "torch"
    if isinstance(value, np.ndarray) or hasattr(value, "tocsr"):
        return "numpy"
    return None


def _dtype_of(value: Any) -> str | None:
    if isinstance(value, Mapping):
        if "x" in value:
            return _dtype_of(value["x"])
        dtypes = {_dtype_of(item) for item in value.values()}
        dtypes.discard(None)
        return next(iter(dtypes)) if len(dtypes) == 1 else None
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    normalized = str(dtype)
    return normalized.removeprefix("torch.")


def _target_facts(value: Any) -> tuple[str | None, int | None]:
    if value is None:
        return None, None
    if type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 1:
        return "matrix", int(array.shape[1]) if array.ndim == 2 else None
    if array.dtype.kind not in {"i", "u"}:
        return "labels", int(np.unique(array).size) if array.size else 0
    visible = array[array >= 0]
    return "class_ids", int(np.unique(visible).size)


def _has_rows(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping) and "x" in value:
        return _has_rows(value["x"])
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return len(shape) == 0 or int(shape[0]) > 0
        except (TypeError, ValueError):
            return True
    try:
        return len(value) > 0
    except TypeError:
        return True


def _mask_has_values(value: Any) -> bool:
    if value is None:
        return False
    if type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value, dtype=bool).any())


_FIRST_STRONG_VIEW_KEYS = frozenset({"X_u_s0", "X_u_s_0", "X_u_s", "X_u_strong0"})
_SECOND_STRONG_VIEW_KEYS = frozenset({"X_u_s1", "X_u_s_1", "X_u_strong1", "X_u_s2", "X_u_s_2"})
_AUGMENTATION_VIEW_KEYS = _FIRST_STRONG_VIEW_KEYS | _SECOND_STRONG_VIEW_KEYS


def _inductive_consumed_facts(consumed_input: Any) -> tuple[Any, bool, bool, int, bool, int]:
    try:
        primary_input = consumed_input.X_l
    except AttributeError as exc:
        raise TypeError("inductive consumed_input must expose X_l") from exc

    X_u = getattr(consumed_input, "X_u", None)
    X_u_w = getattr(consumed_input, "X_u_w", None)
    X_u_s = getattr(consumed_input, "X_u_s", None)
    views = getattr(consumed_input, "views", None)
    if views is not None and not isinstance(views, Mapping):
        raise TypeError("inductive consumed_input.views must be a mapping when provided")

    first_strong = X_u_s
    second_strong = getattr(consumed_input, "X_u_s_1", None)
    if isinstance(views, Mapping):
        if first_strong is None:
            for key in _FIRST_STRONG_VIEW_KEYS:
                if _has_rows(views.get(key)):
                    first_strong = views[key]
                    break
        if second_strong is None:
            for key in _SECOND_STRONG_VIEW_KEYS:
                if _has_rows(views.get(key)):
                    second_strong = views[key]
                    break

    scientific_views = (
        [key for key in views if key not in _AUGMENTATION_VIEW_KEYS]
        if isinstance(views, Mapping)
        else []
    )
    has_weak = _has_rows(X_u_w)
    strong_count = int(_has_rows(first_strong)) + int(_has_rows(second_strong))
    has_unlabeled = any(_has_rows(value) for value in (X_u, X_u_w, first_strong, second_strong))
    return (
        primary_input,
        has_unlabeled,
        getattr(consumed_input, "graph", None) is not None,
        len(scientific_views),
        has_weak,
        strong_count,
    )


def _transductive_consumed_facts(consumed_input: Any) -> tuple[Any, bool, bool, int, bool, int]:
    method_input = getattr(consumed_input, "fit", consumed_input)
    try:
        primary_input = method_input.X
    except AttributeError as exc:
        raise TypeError("transductive consumed_input must expose X (or fit.X)") from exc
    masks = getattr(method_input, "masks", None)
    if masks is not None and not isinstance(masks, Mapping):
        raise TypeError("transductive consumed_input.masks must be a mapping when provided")
    unlabeled = None if masks is None else masks.get("unlabeled_mask", masks.get("unlabeled"))
    return (
        primary_input,
        _mask_has_values(unlabeled),
        getattr(method_input, "graph", None) is not None,
        0,
        False,
        0,
    )


def materialize_consumed_input_capabilities(
    *,
    regime: LearningRegime,
    modality: str,
    consumed_input: Any,
    classifier_outputs: Iterable[str] = ("predictions", "scores"),
    runtime_backend: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    checkpointing_required: bool = False,
) -> PipelineCapabilities:
    """Derive capabilities from the object a method will actually consume.

    Scientific facts (unlabeled data, graph, named views and augmentations) are
    inspected on the method-facing dataset.  They are not copied from global
    preprocessing, graph-building, or YAML configuration state.  Runtime facts
    that are not data properties remain explicit arguments.
    """

    if regime == "inductive":
        facts = _inductive_consumed_facts(consumed_input)
        target = getattr(consumed_input, "y_l", None)
    elif regime == "transductive":
        facts = _transductive_consumed_facts(consumed_input)
        target = getattr(getattr(consumed_input, "fit", consumed_input), "y", None)
    else:
        raise ValueError("regime must be 'inductive' or 'transductive'")
    primary_input, has_unlabeled, has_graph, view_count, has_weak, strong_count = facts
    input_backend = _backend_of(primary_input)
    backend = input_backend if input_backend is not None else runtime_backend
    target_kind, labeled_class_count = _target_facts(target)
    return PipelineCapabilities(
        regime=regime,
        modality=modality,
        representation=_representation_of(primary_input, modality=modality),
        target_kind=target_kind,
        labeled_class_count=labeled_class_count,
        has_unlabeled=has_unlabeled,
        has_graph=has_graph,
        view_count=view_count,
        has_weak_augmentation=has_weak,
        strong_augmentation_count=strong_count,
        classifier_outputs=frozenset(classifier_outputs),
        backend=backend,
        device=device,
        dtype=_dtype_of(primary_input) or dtype,
        checkpointing_required=checkpointing_required,
    )


def validate_consumed_input_capabilities(
    method_id: str,
    method: MethodCapabilities,
    *,
    regime: LearningRegime,
    modality: str,
    consumed_input: Any,
    classifier_outputs: Iterable[str] = ("predictions", "scores"),
    runtime_backend: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    checkpointing_required: bool = False,
) -> PipelineCapabilities:
    """Derive and validate one exact method-facing input capability contract."""

    capabilities = materialize_consumed_input_capabilities(
        regime=regime,
        modality=modality,
        consumed_input=consumed_input,
        classifier_outputs=classifier_outputs,
        runtime_backend=runtime_backend,
        device=device,
        dtype=dtype,
        checkpointing_required=checkpointing_required,
    )
    validate_pipeline_compatibility(method_id, method, capabilities)
    return capabilities


def _has_unlabeled(sampling: Any) -> bool:
    if sampling.is_graph():
        mask = sampling.masks.get("unlabeled")
        return bool(mask is not None and np.asarray(mask, dtype=bool).any())
    indices = sampling.indices.get("train_unlabeled")
    return bool(indices is not None and np.asarray(indices).size)


def materialize_pipeline_capabilities(
    *,
    regime: LearningRegime,
    modality: str,
    primary_input: Any,
    sampling: Any,
    view_count: int = 0,
    has_graph: bool = False,
    has_weak_augmentation: bool = False,
    strong_augmentation_count: int = 0,
    configured_backend: str | None = None,
    model_configured: bool = False,
    requires_torch: bool = False,
    device: str | None = None,
    dtype: str | None = None,
    checkpointing_required: bool = False,
) -> PipelineCapabilities:
    """Describe a fully materialized native pipeline.

    The caller supplies only resolved configuration and materialized objects.
    Representation, backend and unlabeled-data semantics remain owned by
    :mod:`modssc`, independently of any YAML runner.
    """

    backend = configured_backend if isinstance(configured_backend, str) else None
    if backend is not None and backend.lower() == "auto":
        backend = None
    if backend is None:
        backend = "torch" if requires_torch else _backend_of(primary_input)

    classifier_outputs = {"predictions", "scores"}
    if requires_torch or model_configured:
        classifier_outputs.add("logits")

    return PipelineCapabilities(
        regime=regime,
        modality=modality,
        representation=_representation_of(primary_input, modality=modality),
        has_unlabeled=_has_unlabeled(sampling),
        has_graph=has_graph,
        view_count=view_count,
        has_weak_augmentation=has_weak_augmentation,
        strong_augmentation_count=strong_augmentation_count,
        classifier_outputs=frozenset(classifier_outputs),
        backend=backend,
        device=device,
        dtype=dtype,
        checkpointing_required=checkpointing_required,
    )


def check_pipeline_compatibility(
    method_id: str,
    method: MethodCapabilities,
    pipeline: PipelineCapabilities,
) -> CompatibilityReport:
    """Return every incompatibility between ``method`` and ``pipeline``."""

    method_id = _normalize_name(method_id, field="method_id")
    issues: list[CapabilityIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(CapabilityIssue(code=code, message=message))

    if method.regime != pipeline.regime:
        add(
            "E_CAPABILITY_REGIME",
            f"requires regime {method.regime!r}, pipeline provides {pipeline.regime!r}",
        )
    if method.modalities is not None and pipeline.modality not in method.modalities:
        add(
            "E_CAPABILITY_MODALITY",
            f"accepts modalities {sorted(method.modalities)!r}, pipeline provides "
            f"{pipeline.modality!r}",
        )
    if method.representations is not None and pipeline.representation not in method.representations:
        add(
            "E_CAPABILITY_REPRESENTATION",
            f"accepts representations {sorted(method.representations)!r}, pipeline provides "
            f"{pipeline.representation!r}",
        )
    if method.target_kinds is not None and pipeline.target_kind not in method.target_kinds:
        add(
            "E_CAPABILITY_TARGET_KIND",
            f"accepts target kinds {sorted(method.target_kinds)!r}, pipeline provides "
            f"{pipeline.target_kind!r}",
        )
    if (
        method.min_labeled_classes is not None
        and pipeline.labeled_class_count is not None
        and pipeline.labeled_class_count < method.min_labeled_classes
    ):
        add(
            "E_CAPABILITY_CLASS_COUNT",
            f"requires at least {method.min_labeled_classes} labeled classes, pipeline provides "
            f"{pipeline.labeled_class_count}",
        )
    if (
        method.max_labeled_classes is not None
        and pipeline.labeled_class_count is not None
        and pipeline.labeled_class_count > method.max_labeled_classes
    ):
        add(
            "E_CAPABILITY_CLASS_COUNT",
            f"accepts at most {method.max_labeled_classes} labeled classes, pipeline provides "
            f"{pipeline.labeled_class_count}",
        )
    if method.requires_unlabeled and not pipeline.has_unlabeled:
        add("E_CAPABILITY_UNLABELED", "requires unlabeled samples")
    if method.requires_graph and not pipeline.has_graph:
        add("E_CAPABILITY_GRAPH", "requires a graph")
    if pipeline.view_count < method.min_views:
        add(
            "E_CAPABILITY_VIEWS",
            f"requires at least {method.min_views} named views, pipeline provides "
            f"{pipeline.view_count}",
        )
    if method.requires_weak_augmentation and not pipeline.has_weak_augmentation:
        add("E_CAPABILITY_WEAK_AUGMENTATION", "requires a weak augmentation")
    if pipeline.strong_augmentation_count < method.min_strong_augmentations:
        add(
            "E_CAPABILITY_STRONG_AUGMENTATION",
            f"requires at least {method.min_strong_augmentations} strong augmentations, "
            f"pipeline provides {pipeline.strong_augmentation_count}",
        )
    missing_outputs = method.required_classifier_outputs - pipeline.classifier_outputs
    if missing_outputs:
        add(
            "E_CAPABILITY_CLASSIFIER_OUTPUT",
            f"requires classifier outputs {sorted(missing_outputs)!r}",
        )
    for field, code in (
        ("backend", "E_CAPABILITY_BACKEND"),
        ("device", "E_CAPABILITY_DEVICE"),
        ("dtype", "E_CAPABILITY_DTYPE"),
    ):
        accepted = getattr(method, f"{field}s")
        provided = getattr(pipeline, field)
        if accepted is not None and provided not in accepted:
            add(
                code,
                f"accepts {field}s {sorted(accepted)!r}, pipeline provides {provided!r}",
            )
    if pipeline.checkpointing_required and not method.supports_checkpointing:
        add("E_CAPABILITY_CHECKPOINTING", "pipeline requires method checkpoint support")

    return CompatibilityReport(method_id=method_id, issues=tuple(issues))


def validate_pipeline_compatibility(
    method_id: str,
    method: MethodCapabilities,
    pipeline: PipelineCapabilities,
) -> CompatibilityReport:
    """Validate compatibility or raise :class:`IncompatiblePipelineError`."""

    report = check_pipeline_compatibility(method_id, method, pipeline)
    if not report.compatible:
        raise IncompatiblePipelineError(report)
    return report


DEFAULT_INDUCTIVE_CAPABILITIES = MethodCapabilities(
    regime="inductive",
    requires_unlabeled=True,
)

TORCH_INDUCTIVE_CAPABILITIES = MethodCapabilities(
    regime="inductive",
    requires_unlabeled=True,
    required_classifier_outputs=frozenset({"logits"}),
    backends=frozenset({"torch"}),
)

WEAK_STRONG_TORCH_INDUCTIVE_CAPABILITIES = MethodCapabilities(
    regime="inductive",
    requires_unlabeled=True,
    requires_weak_augmentation=True,
    min_strong_augmentations=1,
    required_classifier_outputs=frozenset({"logits"}),
    backends=frozenset({"torch"}),
)

DUAL_STRONG_TORCH_INDUCTIVE_CAPABILITIES = MethodCapabilities(
    regime="inductive",
    requires_unlabeled=True,
    requires_weak_augmentation=True,
    min_strong_augmentations=2,
    required_classifier_outputs=frozenset({"logits"}),
    backends=frozenset({"torch"}),
)

DEFAULT_TRANSDUCTIVE_CAPABILITIES = MethodCapabilities(
    regime="transductive",
    requires_unlabeled=True,
    requires_graph=True,
)

DENSE_TRANSDUCTIVE_CAPABILITIES = MethodCapabilities(
    regime="transductive",
    representations=frozenset({"dense"}),
    requires_unlabeled=True,
    requires_graph=True,
)


__all__ = [
    "CapabilityIssue",
    "CompatibilityReport",
    "DEFAULT_INDUCTIVE_CAPABILITIES",
    "DEFAULT_TRANSDUCTIVE_CAPABILITIES",
    "DENSE_TRANSDUCTIVE_CAPABILITIES",
    "DUAL_STRONG_TORCH_INDUCTIVE_CAPABILITIES",
    "IncompatiblePipelineError",
    "LearningRegime",
    "MethodCapabilities",
    "PipelineCapabilities",
    "TORCH_INDUCTIVE_CAPABILITIES",
    "WEAK_STRONG_TORCH_INDUCTIVE_CAPABILITIES",
    "check_pipeline_compatibility",
    "materialize_pipeline_capabilities",
    "materialize_consumed_input_capabilities",
    "validate_consumed_input_capabilities",
    "validate_pipeline_compatibility",
]
