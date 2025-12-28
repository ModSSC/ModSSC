from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def labels_1d(y: Any) -> np.ndarray:
    arr = to_numpy(y)
    if arr.ndim == 2:
        return arr.argmax(axis=1)
    return arr.reshape(-1)


def predict_labels(scores_or_labels: Any) -> np.ndarray:
    scores = to_numpy(scores_or_labels)
    if scores.ndim == 1:
        return scores.astype(int, copy=False)
    return scores.argmax(axis=1)


def _pred_indices_for_classes(y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    if classes.size == 0:
        return np.full(y_pred.shape, -1, dtype=np.int64)
    try:
        idx = np.searchsorted(classes, y_pred)
        idx_clip = np.clip(idx, 0, int(classes.size) - 1)
        valid = (idx >= 0) & (idx < classes.size)
        valid &= classes[idx_clip] == y_pred
        out = np.full(y_pred.shape, -1, dtype=np.int64)
        out[valid] = idx_clip[valid].astype(np.int64, copy=False)
        return out
    except Exception:
        mapping = {c: i for i, c in enumerate(classes.tolist())}
        flat = y_pred.reshape(-1)
        out = np.fromiter((mapping.get(x, -1) for x in flat), dtype=np.int64, count=flat.size)
        return out.reshape(y_pred.shape)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    try:
        mask = ~np.isnan(y_true)
    except Exception:
        mask = None
    if mask is not None:
        flat_mask = mask.reshape(-1)
        if flat_mask.size != y_true.size:
            flat_mask = np.ones(y_true.reshape(-1).shape, dtype=bool)
        y_true = y_true.reshape(-1)[flat_mask]
        y_pred = y_pred.reshape(-1)[flat_mask]
        if y_true.size == 0:
            return float("nan")
    classes, true_idx = np.unique(y_true, return_inverse=True)
    if classes.size == 0:
        return float("nan")
    pred_idx = _pred_indices_for_classes(y_pred, classes)
    n_classes = int(classes.size)
    true_counts = np.bincount(true_idx, minlength=n_classes).astype(np.float64, copy=False)
    correct = pred_idx == true_idx
    tp = np.bincount(true_idx[correct], minlength=n_classes).astype(np.float64, copy=False)
    recall = np.divide(tp, true_counts, out=np.zeros_like(tp), where=true_counts > 0)
    return float(np.mean(recall))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    classes, true_idx = np.unique(y_true, return_inverse=True)
    if classes.size == 0:
        return float("nan")
    pred_idx = _pred_indices_for_classes(y_pred, classes)
    n_classes = int(classes.size)
    true_counts = np.bincount(true_idx, minlength=n_classes).astype(np.float64, copy=False)
    pred_counts = np.bincount(pred_idx[pred_idx >= 0], minlength=n_classes).astype(
        np.float64, copy=False
    )
    correct = pred_idx == true_idx
    tp = np.bincount(true_idx[correct], minlength=n_classes).astype(np.float64, copy=False)
    prec = np.divide(tp, pred_counts, out=np.zeros_like(tp), where=pred_counts > 0)
    rec = np.divide(tp, true_counts, out=np.zeros_like(tp), where=true_counts > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0)
    return float(np.mean(f1))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "macro_f1": macro_f1,
}


def list_metrics() -> list[str]:
    return sorted(METRICS)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, metrics: Iterable[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    available = list_metrics()
    for name in metrics:
        metric_fn = METRICS.get(name)
        if metric_fn is None:
            raise ValueError(f"Unknown metric: {name}. Available: {available}")
        out[name] = metric_fn(y_true, y_pred)
    return out


def evaluate(
    y_true: Any,
    scores_or_labels: Any,
    metrics: Iterable[str],
) -> dict[str, float]:
    y_true_arr = labels_1d(y_true)
    scores = to_numpy(scores_or_labels)
    y_pred = predict_labels(scores)
    return compute_metrics(y_true_arr, y_pred, metrics)
