from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modssc.inductive.adapters import NumpyDataset, TorchDataset, to_numpy_dataset, to_torch_dataset
from modssc.inductive.errors import InductiveValidationError, OptionalDependencyError
from modssc.inductive.optional import optional_import
from modssc.inductive.types import DeviceSpec
from modssc.supervised.api import create_classifier
from modssc.supervised.types import ClassifierRuntime


@dataclass(frozen=True)
class BaseClassifierSpec:
    classifier_id: str = "knn"
    classifier_backend: str = "numpy"
    classifier_params: Mapping[str, Any] = field(default_factory=dict)


def ensure_cpu_device(device: DeviceSpec) -> None:
    if device.device not in {"cpu", "auto"}:
        raise InductiveValidationError(
            f"CPU-only method; device={device.device!r} is not supported."
        )


def ensure_numpy_data(data: Any) -> NumpyDataset:
    return to_numpy_dataset(data)


def flatten_if_numpy(x: Any) -> np.ndarray:
    """Flatten data to (N, D) if >2D, required for sklearn-like classifiers."""
    if not isinstance(x, np.ndarray):
        return x
    if x.ndim > 2:
        return x.reshape(x.shape[0], -1)
    return x


def _torch():
    return optional_import("torch", extra="inductive-torch")


def is_torch_tensor(x: Any) -> bool:
    try:
        torch = _torch()
    except OptionalDependencyError:
        return False
    if isinstance(x, torch.Tensor):
        return True
    if isinstance(x, dict):
        # Allow dicts (e.g. PyG data objects) if they contain at least one tensor
        return any(isinstance(v, torch.Tensor) for v in x.values())
    return False


def ensure_torch_data(data: Any, *, device: DeviceSpec) -> TorchDataset:
    return to_torch_dataset(data, device=device, require_same_device=True)


def detect_backend(x: Any) -> str:
    if isinstance(x, np.ndarray):
        return "numpy"
    if is_torch_tensor(x):
        return "torch"
    raise InductiveValidationError(
        "X_l must be a numpy.ndarray or torch.Tensor (or dict of Tensors). Use preprocess core.to_numpy or core.to_torch."
    )


def ensure_1d_labels(y: Any, *, name: str = "y_l") -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise InductiveValidationError(f"{name} must be 1D integer labels.")
    if arr.size == 0:
        raise InductiveValidationError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise InductiveValidationError(f"{name} must have an integer dtype.")
    return arr


def ensure_1d_labels_torch(y: Any, *, name: str = "y_l"):
    torch = _torch()
    if not isinstance(y, torch.Tensor):
        raise InductiveValidationError(f"{name} must be a torch.Tensor.")
    if y.ndim != 1:
        raise InductiveValidationError(f"{name} must be 1D integer labels.")
    if y.numel() == 0:
        raise InductiveValidationError(f"{name} must be non-empty.")
    if y.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise InductiveValidationError(f"{name} must have an integer dtype.")
    return y


def build_classifier(spec: BaseClassifierSpec, *, seed: int) -> Any:
    runtime = ClassifierRuntime(seed=int(seed))
    return create_classifier(
        spec.classifier_id,
        backend=spec.classifier_backend,
        params=dict(spec.classifier_params),
        runtime=runtime,
    )


def ensure_classifier_backend(spec: BaseClassifierSpec, *, backend: str) -> None:
    if backend == "torch":
        if spec.classifier_backend != "torch":
            raise InductiveValidationError("Torch inputs require classifier_backend='torch'.")
    elif backend == "numpy":
        if spec.classifier_backend == "torch":
            raise InductiveValidationError(
                "Numpy inputs require classifier_backend in {'numpy','sklearn'}."
            )
    else:
        raise InductiveValidationError(f"Unknown backend: {backend!r}")


def _predict_scores_numpy(model: Any, X: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        raise InductiveValidationError(
            "Numpy backend requires numpy.ndarray inputs. Use preprocess core.to_numpy."
        )
    if hasattr(model, "predict_scores"):
        scores = model.predict_scores(X)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    else:
        if not hasattr(model, "predict"):
            raise InductiveValidationError("Base classifier must implement predict().")
        pred = np.asarray(model.predict(X))
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = np.unique(pred)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx = np.vectorize(class_to_idx.get, otypes=[int])(pred)
        scores = np.eye(len(classes), dtype=np.float32)[idx]

    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise InductiveValidationError("predict_scores must return shape (n_samples, n_classes).")
    return scores


def _predict_scores_torch(model: Any, X: Any):
    torch = _torch()
    is_dict = isinstance(X, dict) and "x" in X
    if not isinstance(X, torch.Tensor) and not is_dict:
        raise InductiveValidationError(
            "Torch backend requires torch.Tensor inputs. Use preprocess core.to_torch."
        )
    if hasattr(model, "predict_scores"):
        scores = model.predict_scores(X)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    else:
        raise InductiveValidationError(
            "Torch classifier must implement predict_scores() or predict_proba()."
        )
    if not isinstance(scores, torch.Tensor):
        raise InductiveValidationError("Torch classifier must return torch.Tensor scores.")
    if scores.ndim != 2:
        raise InductiveValidationError("predict_scores must return shape (n_samples, n_classes).")
    x_device = X["x"].device if (isinstance(X, dict) and "x" in X) else X.device
    if scores.device != x_device:
        raise InductiveValidationError("Torch classifier returned scores on a different device.")
    return scores


def predict_scores(model: Any, X: Any, *, backend: str):
    if backend == "numpy":
        return _predict_scores_numpy(model, X)
    if backend == "torch":
        return _predict_scores_torch(model, X)
    raise InductiveValidationError(f"Unknown backend: {backend!r}")


def select_confident(
    scores: np.ndarray,
    *,
    threshold: float | None = None,
    max_new: int | None = None,
) -> np.ndarray:
    conf = scores.max(axis=1)
    idx = np.arange(conf.shape[0], dtype=np.int64)
    if threshold is not None:
        idx = idx[conf >= float(threshold)]
        conf = conf[conf >= float(threshold)]
    if max_new is not None and idx.size > int(max_new):
        order = np.argsort(conf)[::-1][: int(max_new)]
        idx = idx[order]
    return idx


def select_top_per_class(
    scores: np.ndarray,
    *,
    k_per_class: int,
    threshold: float | None = None,
) -> np.ndarray:
    pred = scores.argmax(axis=1)
    idx_all: list[int] = []
    for cls in range(scores.shape[1]):
        cls_idx = np.where(pred == cls)[0]
        if cls_idx.size == 0:
            continue
        cls_scores = scores[cls_idx, cls]
        if threshold is not None:
            keep = cls_scores >= float(threshold)
            cls_idx = cls_idx[keep]
            cls_scores = cls_scores[keep]
        if cls_idx.size == 0:
            continue
        order = np.argsort(cls_scores)[::-1]
        take = order[: int(k_per_class)]
        idx_all.extend(cls_idx[take].tolist())
    return np.asarray(sorted(set(idx_all)), dtype=np.int64)


def select_confident_torch(
    scores: Any,
    *,
    threshold: float | None = None,
    max_new: int | None = None,
):
    torch = _torch()
    conf, _ = scores.max(dim=1)
    idx = torch.arange(int(conf.shape[0]), device=scores.device)
    if threshold is not None:
        mask = conf >= float(threshold)
        idx = idx[mask]
        conf = conf[mask]
    if max_new is not None and int(idx.numel()) > int(max_new):
        order = torch.topk(conf, k=int(max_new)).indices
        idx = idx[order]
    return idx


def select_top_per_class_torch(
    scores: Any,
    *,
    k_per_class: int,
    threshold: float | None = None,
):
    torch = _torch()
    pred = scores.argmax(dim=1)
    idx_all = []
    for cls in range(int(scores.shape[1])):
        cls_idx = (pred == cls).nonzero(as_tuple=False).reshape(-1)
        if cls_idx.numel() == 0:
            continue
        cls_scores = scores[cls_idx, cls]
        if threshold is not None:
            mask = cls_scores >= float(threshold)
            cls_idx = cls_idx[mask]
            cls_scores = cls_scores[mask]
        if cls_idx.numel() == 0:
            continue
        k = min(int(k_per_class), int(cls_idx.numel()))
        order = torch.topk(cls_scores, k=k).indices
        idx_all.append(cls_idx[order])
    if not idx_all:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    merged = torch.unique(torch.cat(idx_all), sorted=True)
    return merged
