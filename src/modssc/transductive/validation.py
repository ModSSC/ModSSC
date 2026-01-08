from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import NodeDatasetLike
from .errors import TransductiveValidationError


def _as_numpy(a: Any) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    # torch tensor support without importing torch
    if hasattr(a, "detach") and hasattr(a, "cpu") and hasattr(a, "numpy"):
        return a.detach().cpu().numpy()
    if hasattr(a, "cpu") and hasattr(a, "numpy"):
        return a.cpu().numpy()
    if hasattr(a, "numpy"):
        return a.numpy()
    return np.asarray(a)


def validate_node_dataset(data: NodeDatasetLike) -> None:
    """Validate the minimal invariants needed by transductive algorithms."""
    if data is None:
        raise TransductiveValidationError("data must not be None")

    X = _as_numpy(data.X)
    y = _as_numpy(data.y)

    if X.ndim != 2:
        raise TransductiveValidationError(f"X must be 2D (n, d), got shape {X.shape}")
    if y.ndim != 1:
        raise TransductiveValidationError(f"y must be 1D (n,), got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise TransductiveValidationError("X and y must have the same first dimension")

    y_arr = np.asarray(y)
    if np.issubdtype(y_arr.dtype, np.integer):
        y_int = y_arr.astype(np.int64, copy=False)
    elif np.issubdtype(y_arr.dtype, np.floating):
        if not np.isfinite(y_arr).all():
            raise TransductiveValidationError(
                "y must contain finite integer class ids; found non-finite values"
            )
        y_int = y_arr.astype(np.int64)
        if not np.all(y_arr == y_int):
            raise TransductiveValidationError(
                "y must contain integer class ids; run preprocess step 'labels.encode'"
            )
    else:
        raise TransductiveValidationError(
            "y must contain integer class ids; run preprocess step 'labels.encode'"
        )

    y_valid = y_int[y_int >= 0]
    if y_valid.size:
        classes = np.unique(y_valid)
        if not np.array_equal(classes, np.arange(int(classes.size))):
            raise TransductiveValidationError(
                "y must contain contiguous class ids starting at 0; run preprocess step 'labels.encode'"
            )

    if data.graph is None:
        raise TransductiveValidationError("data.graph must not be None")

    raw_edge_index = getattr(data.graph, "edge_index", None)
    if raw_edge_index is None:
        raise TransductiveValidationError("graph.edge_index is required")
    edge_index = _as_numpy(raw_edge_index)

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise TransductiveValidationError(
            f"edge_index must have shape (2, E), got {edge_index.shape}"
        )

    n = int(X.shape[0])
    if edge_index.size > 0 and (edge_index.min() < 0 or edge_index.max() >= n):
        raise TransductiveValidationError("edge_index has out of range node indices")

    masks: Mapping[str, Any] = data.masks or {}
    for key in ("train_mask", "val_mask", "test_mask", "unlabeled_mask"):
        if key not in masks:
            continue
        m = _as_numpy(masks[key]).astype(bool)
        if m.shape != (n,):
            raise TransductiveValidationError(f"{key} must have shape (n,), got {m.shape}")
