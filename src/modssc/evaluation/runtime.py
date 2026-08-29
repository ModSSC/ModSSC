"""Native evaluation runtime for fitted ModSSC methods.

The benchmark runner supplies named data splits.  Prediction contracts,
diagnostic outputs, backend materialization, metric computation, and the
transductive truth boundary are owned here so they are equally available to
programmatic users that do not use benchmark YAML files.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from modssc.data_augmentation.utils import is_torch_tensor
from modssc.data_loader.selection import select_rows

from .metrics import compute_metrics, labels_1d, predict_labels

_LOGGER = logging.getLogger(__name__)

EvaluationErrorKind = Literal["contract", "split", "torch_required", "shape"]

_ERROR_CODES: dict[EvaluationErrorKind, str] = {
    "contract": "E_EVALUATION_CONTRACT",
    "split": "E_EVALUATION_SPLIT_INVALID",
    "torch_required": "E_EVALUATION_PREPROCESS_TO_TORCH_REQUIRED",
    "shape": "E_EVALUATION_SHAPE_CONTRACT",
}


class EvaluationError(ValueError):
    """Raised when a native evaluation contract cannot be fulfilled."""

    def __init__(self, kind: EvaluationErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.code = _ERROR_CODES[kind]


@dataclass(frozen=True)
class MethodEvaluationRuntime:
    """Public fitted-runtime information needed to materialize predictions.

    Native method execution attaches this value as ``evaluation_runtime_`` on
    a fitted method.  Custom callers may instead pass it explicitly to
    :func:`evaluate_inductive_method`.  No evaluator needs to inspect a
    method's private classifier or model objects.
    """

    backend: str | None = None
    device: Any | None = None

    def __post_init__(self) -> None:
        backend = None if self.backend is None else str(self.backend).lower()
        if backend not in {None, "numpy", "torch"}:
            raise ValueError("evaluation backend must be 'numpy', 'torch', or None")
        if backend != "torch" and self.device is not None:
            raise ValueError("evaluation device requires backend='torch'")
        object.__setattr__(self, "backend", backend)

    @classmethod
    def from_features(
        cls,
        features: Any,
        *,
        backend: str | None = None,
    ) -> MethodEvaluationRuntime:
        """Build the public runtime from the exact fitted feature container."""

        inferred_backend = backend
        device = _first_torch_device(features)
        if inferred_backend is None:
            inferred_backend = "torch" if device is not None else "numpy"
        return cls(backend=inferred_backend, device=device)


@dataclass(frozen=True)
class InductiveEvaluationSplit:
    """One selected evaluation split before method-backend materialization."""

    X: Any
    y_true: Any
    views: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.views is not None:
            object.__setattr__(self, "views", MappingProxyType(dict(self.views)))


InductiveSplitProvider = Callable[[str], InductiveEvaluationSplit]


def _selected_labels(preprocess: Any, *, reference: str, base: Any) -> Any:
    store = preprocess.train_artifacts if reference == "train" else preprocess.test_artifacts
    if store is not None and store.has("labels.y"):
        return store.get("labels.y")
    return base.y


def _select_inductive_split(
    *,
    preprocess: Any,
    sampling: Any,
    views: Any | None,
    split: str,
) -> InductiveEvaluationSplit:
    if sampling.is_graph():
        raise EvaluationError(
            "split",
            "inductive evaluation does not support graph sampling",
        )
    try:
        raw_indices = sampling.indices[split]
    except KeyError as exc:
        raise EvaluationError("split", f"unknown evaluation split {split!r}") from exc

    indices = np.asarray(raw_indices, dtype=np.int64)
    reference = sampling.refs.get(split, "train")
    if reference == "train":
        base = preprocess.dataset.train
    elif reference == "test":
        base = preprocess.dataset.test
        if base is None:
            raise EvaluationError(
                "split",
                "requested test split but dataset has no test split",
            )
    else:
        raise EvaluationError(
            "split",
            f"evaluation split {split!r} has unknown reference {reference!r}",
        )

    labels = _selected_labels(preprocess, reference=reference, base=base)
    selected_views: dict[str, Any] | None = None
    if views is not None:
        selected_views = {}
        for name, dataset in views.views.items():
            view_base = dataset.train if reference == "train" else dataset.test
            if view_base is None:
                raise EvaluationError(
                    "split",
                    f"requested test split but view '{name}' has no test split",
                )
            selected_views[name] = {
                "X": select_rows(
                    view_base.X,
                    indices,
                    context=f"evaluation.views[{name}].{split}",
                )
            }

    return InductiveEvaluationSplit(
        X=select_rows(base.X, indices, context=f"evaluation.{split}.X"),
        y_true=select_rows(labels, indices, context=f"evaluation.{split}.y"),
        views=selected_views,
    )


def make_inductive_split_provider(
    *,
    preprocess: Any,
    sampling: Any,
    views: Any | None = None,
) -> InductiveSplitProvider:
    """Return a cached native selector for named inductive evaluation splits."""

    cache: dict[str, InductiveEvaluationSplit] = {}

    def provide(split: str) -> InductiveEvaluationSplit:
        cached = cache.get(split)
        if cached is None:
            cached = _select_inductive_split(
                preprocess=preprocess,
                sampling=sampling,
                views=views,
                split=split,
            )
            cache[split] = cached
        return cached

    return provide


@runtime_checkable
class EvaluationOutputProvider(Protocol):
    """Optional generic protocol for additional named prediction outputs."""

    def predict_evaluation_outputs(self, data: Any) -> Mapping[str, Any]: ...


@runtime_checkable
class EvaluationPredictionProvider(Protocol):
    """Optional method-owned contract for the primary reported predictor.

    This separates a method's general ``predict_proba`` default from the
    predictor prescribed by an evaluation protocol, such as an EMA model.
    """

    def predict_evaluation_proba(self, data: Any) -> Any: ...


@runtime_checkable
class EvaluationMetricSetProvider(Protocol):
    """Optional fitted-method protocol for historically reported metric sets."""

    def evaluation_metric_sets(self) -> Mapping[str, Mapping[str, Any]]: ...


@runtime_checkable
class EvaluationMetricRecorder(Protocol):
    """Optional hook for methods that retain evaluation diagnostics."""

    def record_evaluation_metrics(
        self,
        *,
        split: str,
        output: str,
        metrics: Mapping[str, float],
    ) -> None: ...


def _array_backend_flags(value: Any) -> tuple[bool, bool]:
    if is_torch_tensor(value):
        return True, False
    if isinstance(value, Mapping):
        has_torch = False
        has_numpy = False
        for child in value.values():
            child_torch, child_numpy = _array_backend_flags(child)
            has_torch = has_torch or child_torch
            has_numpy = has_numpy or child_numpy
        return has_torch, has_numpy
    if isinstance(value, (list, tuple, set)):
        has_torch = False
        has_numpy = False
        for child in value:
            child_torch, child_numpy = _array_backend_flags(child)
            has_torch = has_torch or child_torch
            has_numpy = has_numpy or child_numpy
        return has_torch, has_numpy
    if isinstance(value, np.ndarray):
        return False, True
    return False, False


def _is_torch_container(value: Any) -> bool:
    if is_torch_tensor(value):
        return True
    if isinstance(value, Mapping):
        has_torch, has_numpy = _array_backend_flags(value)
        return has_torch and not has_numpy
    return False


def _first_torch_device(value: Any) -> Any | None:
    if value is None or isinstance(value, np.ndarray):
        return None
    if isinstance(value, Mapping):
        for child in value.values():
            device = _first_torch_device(child)
            if device is not None:
                return device
        return None
    if isinstance(value, (list, tuple, set)):
        for child in value:
            device = _first_torch_device(child)
            if device is not None:
                return device
        return None
    if is_torch_tensor(value):
        return getattr(value, "device", None)
    return None


def _smart_to_torch(value: Any, device: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: _smart_to_torch(child, device) for key, child in value.items()}

    torch = importlib.import_module("torch")
    if is_torch_tensor(value):
        return value.to(device) if hasattr(value, "to") else value

    array = np.asarray(value)
    if array.dtype == np.uint8:
        return torch.tensor(array, device=device, dtype=torch.float32).div_(255.0)
    dtype = torch.float32 if array.dtype == np.float64 else None
    return torch.as_tensor(array, device=device, dtype=dtype)


def _public_method_runtime(method: Any) -> MethodEvaluationRuntime:
    runtime = getattr(method, "evaluation_runtime_", None)
    if runtime is not None:
        if not isinstance(runtime, MethodEvaluationRuntime):
            raise EvaluationError(
                "contract",
                "method.evaluation_runtime_ must be a MethodEvaluationRuntime",
            )
        return runtime

    # Absence is an explicit part of the contract.  Direct callers that did
    # not use native method execution may pass ``runtime=...``; the evaluator
    # never guesses a fitted device from unrelated method attributes.
    return MethodEvaluationRuntime()


def _materialize_for_runtime(
    value: Any,
    *,
    runtime: MethodEvaluationRuntime,
    strict: bool,
    context: str,
) -> Any:
    if runtime.backend != "torch":
        return value
    if _is_torch_container(value):
        if runtime.device is not None:
            return _smart_to_torch(value, runtime.device)
        return value
    if strict:
        raise EvaluationError(
            "torch_required",
            f"{context} is not torch-backed in benchmark_mode",
        )
    destination = runtime.device if runtime.device is not None else "cpu"
    return _smart_to_torch(value, destination)


def _materialize_views(
    views: Mapping[str, Any] | None,
    *,
    runtime: MethodEvaluationRuntime,
    strict: bool,
    context: str,
) -> Mapping[str, Any] | None:
    if views is None:
        return None
    return {
        name: _materialize_for_runtime(
            value,
            runtime=runtime,
            strict=strict,
            context=f"{context}.views[{name}]",
        )
        for name, value in views.items()
    }


def _prediction_input_contract(method: Any) -> str:
    info = getattr(method, "info", None)
    value = getattr(info, "prediction_input", "features")
    if value not in {"features", "dataset"}:
        raise EvaluationError(
            "contract",
            f"unknown method prediction_input contract: {value!r}",
        )
    return str(value)


def _evaluation_reference_splits(method: Any) -> tuple[str, ...]:
    declared = getattr(method, "evaluation_reference_splits", None)
    if callable(declared):
        declared = declared()
    if declared is None:
        declared = getattr(getattr(method, "info", None), "evaluation_reference_splits", ())
    if not isinstance(declared, tuple) or any(
        not isinstance(split, str) or not split for split in declared
    ):
        raise EvaluationError(
            "contract",
            "evaluation_reference_splits must be a tuple of non-empty split names",
        )
    if len(set(declared)) != len(declared):
        raise EvaluationError(
            "contract",
            "evaluation_reference_splits must not contain duplicates",
        )
    return declared


def _prediction_payload(
    *,
    method: Any,
    split_name: str,
    split: InductiveEvaluationSplit,
    split_provider: InductiveSplitProvider,
    runtime: MethodEvaluationRuntime,
    strict: bool,
) -> Any:
    X = _materialize_for_runtime(
        split.X,
        runtime=runtime,
        strict=strict,
        context="evaluation features",
    )
    if _prediction_input_contract(method) == "features":
        return X
    if split.views is None:
        raise EvaluationError(
            "contract",
            "the fitted method requires views for evaluation",
        )

    # Imported lazily to avoid a package cycle when inductive execution imports
    # MethodEvaluationRuntime during package initialization.
    from modssc.inductive.types import InductiveDataset

    views = _materialize_views(
        split.views,
        runtime=runtime,
        strict=strict,
        context="evaluation",
    )
    references: dict[str, InductiveDataset] = {}
    for reference_name in _evaluation_reference_splits(method):
        reference = split_provider(reference_name)
        reference_X = _materialize_for_runtime(
            reference.X,
            runtime=runtime,
            strict=strict,
            context=f"evaluation reference split '{reference_name}'",
        )
        reference_y = _materialize_for_runtime(
            reference.y_true,
            runtime=runtime,
            # Strict evaluation historically authenticates feature
            # preprocessing. Labels are method-owned reference metadata and
            # may be losslessly materialized to the fitted backend.
            strict=False,
            context=f"evaluation reference labels '{reference_name}'",
        )
        reference_views = _materialize_views(
            reference.views,
            runtime=runtime,
            strict=strict,
            context=f"evaluation reference split '{reference_name}'",
        )
        references[reference_name] = InductiveDataset(
            X_l=reference_X,
            y_l=reference_y,
            views=reference_views,
        )
    return InductiveDataset(
        X_l=X,
        y_l=None,
        views=views,
        meta={
            "evaluation_split": split_name,
            "evaluation_reference_splits": references,
        },
    )


def _prediction_distribution(y_pred: np.ndarray) -> dict[str, int]:
    if y_pred.size == 0:
        return {}
    classes, counts = np.unique(y_pred, return_counts=True)
    return {str(cls): int(count) for cls, count in zip(classes, counts, strict=True)}


def _log_split_metrics(
    *,
    kind: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_out: Mapping[str, float],
) -> None:
    prediction_distribution = _prediction_distribution(y_pred)
    args = (
        kind,
        split,
        int(y_pred.size),
        int(np.unique(y_true).size) if y_true.size else 0,
        len(prediction_distribution),
        prediction_distribution,
        dict(metrics_out),
    )
    message = (
        "Evaluation (%s) split=%s n=%s true_classes=%s pred_classes=%s pred_dist=%s metrics=%s"
    )
    if y_pred.size > 0 and len(prediction_distribution) <= 1:
        _LOGGER.warning(message, *args)
    else:
        _LOGGER.info(message, *args)


def _metrics_for_scores(
    *,
    scores: Any,
    y_true: np.ndarray,
    metrics: tuple[str, ...],
    context: str,
) -> tuple[dict[str, float], np.ndarray]:
    y_pred = predict_labels(scores)
    if y_pred.ndim != 1 or int(y_pred.shape[0]) != int(y_true.shape[0]):
        raise EvaluationError(
            "shape",
            f"{context} predictions have shape {y_pred.shape}; expected ({y_true.shape[0]},)",
        )
    return compute_metrics(y_true, y_pred, metrics), y_pred


def _additional_outputs(method: Any, payload: Any) -> Mapping[str, Any]:
    provider = getattr(method, "predict_evaluation_outputs", None)
    if provider is None:
        return {}
    if not callable(provider):
        raise EvaluationError(
            "contract",
            "method.predict_evaluation_outputs must be callable",
        )
    outputs = provider(payload)
    if not isinstance(outputs, Mapping):
        raise EvaluationError(
            "contract",
            "predict_evaluation_outputs() must return a mapping",
        )
    normalized: dict[str, Any] = {}
    for raw_name, scores in outputs.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise EvaluationError(
                "contract",
                "evaluation output names must be non-empty strings",
            )
        if raw_name in normalized:
            raise EvaluationError("contract", f"duplicate evaluation output {raw_name!r}")
        normalized[raw_name] = scores
    return normalized


def _primary_evaluation_scores(method: Any, payload: Any) -> Any:
    predictor = getattr(method, "predict_evaluation_proba", None)
    if predictor is not None:
        if not callable(predictor):
            raise EvaluationError(
                "contract",
                "method.predict_evaluation_proba must be callable",
            )
        return predictor(payload)
    predictor = getattr(method, "predict_proba", None)
    if not callable(predictor):
        raise EvaluationError("contract", "method.predict_proba must be callable")
    return predictor(payload)


def _reported_metric_sets(method: Any) -> dict[str, dict[str, Any]]:
    provider = getattr(method, "evaluation_metric_sets", None)
    if provider is None:
        return {}
    if not callable(provider):
        raise EvaluationError("contract", "method.evaluation_metric_sets must be callable")
    raw_sets = provider()
    if not isinstance(raw_sets, Mapping):
        raise EvaluationError(
            "contract",
            "evaluation_metric_sets() must return a mapping",
        )
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_metrics in raw_sets.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise EvaluationError(
                "contract",
                "reported evaluation metric-set names must be non-empty strings",
            )
        if not isinstance(raw_metrics, Mapping):
            raise EvaluationError(
                "contract",
                f"reported evaluation metric set {raw_name!r} must be a mapping",
            )
        if any(not isinstance(key, str) or not key for key in raw_metrics):
            raise EvaluationError(
                "contract",
                f"reported evaluation metric set {raw_name!r} has an invalid key",
            )
        normalized[raw_name] = dict(raw_metrics)
    return normalized


def evaluate_inductive_method(
    *,
    method: Any,
    split_provider: InductiveSplitProvider,
    report_splits: Iterable[str],
    metrics: Iterable[str],
    strict: bool = False,
    runtime: MethodEvaluationRuntime | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate a fitted inductive method over caller-selected named splits."""

    split_names = tuple(report_splits)
    metric_names = tuple(metrics)
    if any(not isinstance(name, str) or not name for name in split_names):
        raise EvaluationError("contract", "report split names must be non-empty strings")
    if len(set(split_names)) != len(split_names):
        raise EvaluationError("contract", "report split names must be unique")
    if runtime is not None and not isinstance(runtime, MethodEvaluationRuntime):
        raise EvaluationError("contract", "runtime must be a MethodEvaluationRuntime")
    resolved_runtime = runtime if runtime is not None else _public_method_runtime(method)
    results: dict[str, dict[str, Any]] = {}

    for split_name in split_names:
        if split_name in results:
            raise EvaluationError(
                "contract",
                f"evaluation result collision for split {split_name!r}",
            )
        split = split_provider(split_name)
        if not isinstance(split, InductiveEvaluationSplit):
            raise EvaluationError(
                "contract",
                "split_provider must return InductiveEvaluationSplit values",
            )
        payload = _prediction_payload(
            method=method,
            split_name=split_name,
            split=split,
            split_provider=split_provider,
            runtime=resolved_runtime,
            strict=bool(strict),
        )
        y_true = labels_1d(split.y_true)
        primary_metrics, primary_predictions = _metrics_for_scores(
            scores=_primary_evaluation_scores(method, payload),
            y_true=y_true,
            metrics=metric_names,
            context=f"evaluation split '{split_name}'",
        )
        results[split_name] = primary_metrics
        _log_split_metrics(
            kind="inductive",
            split=split_name,
            y_true=y_true,
            y_pred=primary_predictions,
            metrics_out=primary_metrics,
        )

        for output_name, scores in _additional_outputs(method, payload).items():
            output_metrics, output_predictions = _metrics_for_scores(
                scores=scores,
                y_true=y_true,
                metrics=metric_names,
                context=f"evaluation output '{output_name}' for split '{split_name}'",
            )
            result_name = f"{split_name}_{output_name}"
            if result_name in results:
                raise EvaluationError(
                    "contract",
                    f"evaluation output collision for result name {result_name!r}",
                )
            results[result_name] = output_metrics
            _log_split_metrics(
                kind="inductive_named",
                split=result_name,
                y_true=y_true,
                y_pred=output_predictions,
                metrics_out=output_metrics,
            )
            recorder = getattr(method, "record_evaluation_metrics", None)
            if recorder is not None:
                if not callable(recorder):
                    raise EvaluationError(
                        "contract",
                        "method.record_evaluation_metrics must be callable",
                    )
                recorder(
                    split=split_name,
                    output=output_name,
                    metrics=output_metrics,
                )
    for set_name, metric_set in _reported_metric_sets(method).items():
        if set_name in results:
            raise EvaluationError(
                "contract",
                f"reported evaluation metric-set collision for {set_name!r}",
            )
        results[set_name] = metric_set
    return results


def evaluate_transductive_method(
    *,
    method: Any,
    data: Any,
    report_splits: Iterable[str],
    metrics: Iterable[str],
    declared_masks: Mapping[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate a fitted transductive method without exposing truth to it."""

    split_names = tuple(report_splits)
    metric_names = tuple(metrics)
    if any(not isinstance(name, str) or not name for name in split_names):
        raise EvaluationError("contract", "report split names must be non-empty strings")
    if len(set(split_names)) != len(split_names):
        raise EvaluationError("contract", "report split names must be unique")
    try:
        fit_data = data.fit
        evaluation = data.evaluation
        y_true = labels_1d(evaluation.y_true)
        evaluation_masks = evaluation.masks
    except AttributeError as exc:
        raise EvaluationError(
            "contract",
            "transductive evaluation requires PreparedNodeData",
        ) from exc

    predictor = getattr(method, "predict_proba", None)
    if not callable(predictor):
        raise EvaluationError("contract", "method.predict_proba must be callable")
    y_pred_all = predict_labels(predictor(fit_data))
    if y_pred_all.ndim != 1 or int(y_pred_all.shape[0]) != int(y_true.shape[0]):
        raise EvaluationError(
            "shape",
            "transductive predictions must contain exactly one value per node",
        )

    results: dict[str, dict[str, float]] = {}
    for split_name in split_names:
        key = f"{split_name}_mask"
        if key not in evaluation_masks:
            raise EvaluationError("contract", f"missing mask for split '{split_name}'")
        mask = np.asarray(evaluation_masks[key], dtype=bool)
        if mask.ndim != 1 or int(mask.shape[0]) != int(y_true.shape[0]):
            raise EvaluationError(
                "contract",
                f"mask size mismatch for split '{split_name}': {mask.shape} vs {y_true.shape}",
            )
        if declared_masks is not None:
            declared = (
                np.asarray(declared_masks[key], dtype=bool) if key in declared_masks else None
            )
            if declared is None or not np.array_equal(declared, mask):
                raise EvaluationError(
                    "contract",
                    f"runtime mask differs from prepared evaluation mask for split '{split_name}'",
                )

        split_truth = y_true[mask]
        split_predictions = y_pred_all[mask]
        split_metrics = compute_metrics(split_truth, split_predictions, metric_names)
        results[split_name] = split_metrics
        _log_split_metrics(
            kind="transductive",
            split=split_name,
            y_true=split_truth,
            y_pred=split_predictions,
            metrics_out=split_metrics,
        )
    return results


__all__ = [
    "EvaluationError",
    "EvaluationMetricRecorder",
    "EvaluationMetricSetProvider",
    "EvaluationOutputProvider",
    "EvaluationPredictionProvider",
    "InductiveEvaluationSplit",
    "InductiveSplitProvider",
    "MethodEvaluationRuntime",
    "evaluate_inductive_method",
    "evaluate_transductive_method",
    "make_inductive_split_provider",
]
