from __future__ import annotations

import numpy as np


def labels_to_onehot(y: np.ndarray, *, n_classes: int) -> np.ndarray:
    if n_classes <= 0:
        raise ValueError(f"n_classes must be positive, got {n_classes}")
    y_arr = np.asarray(y).reshape(-1)
    if np.issubdtype(y_arr.dtype, np.floating):
        if not np.isfinite(y_arr).all():
            raise ValueError("y must contain finite integer class ids")
        y_int = y_arr.astype(np.int64)
        if not np.all(y_arr == y_int):
            raise ValueError("y must contain integer class ids")
    else:
        y_int = y_arr.astype(np.int64, copy=False)
    out = np.zeros((y_int.shape[0], n_classes), dtype=np.float32)
    valid = y_int >= 0
    if valid.any() and int(y_int[valid].max()) >= int(n_classes):
        raise ValueError("y contains class ids outside [0, n_classes)")
    out[np.arange(y_int.shape[0])[valid], y_int[valid]] = 1.0
    return out


def hard_clamp(F: np.ndarray, Y: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Force F on train nodes to match Y exactly."""
    train_mask = np.asarray(train_mask, dtype=bool)
    out = np.asarray(F, dtype=np.float32).copy()
    out[train_mask] = Y[train_mask]
    return out


def row_normalize(F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    sums = F.sum(axis=1, keepdims=True)
    return F / np.maximum(sums, eps)
