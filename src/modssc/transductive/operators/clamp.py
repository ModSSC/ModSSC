from __future__ import annotations

import numpy as np


def labels_to_onehot(y: np.ndarray, *, n_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    valid = y >= 0
    out[np.arange(y.shape[0])[valid], y[valid]] = 1.0
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
