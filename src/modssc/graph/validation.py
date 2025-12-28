from __future__ import annotations

from typing import Any

import numpy as np

from .errors import GraphValidationError
from .specs import GraphBuilderSpec, GraphFeaturizerSpec


def validate_features(X: Any) -> None:
    """Validate that X is array-like with shape (n, d)."""
    if not hasattr(X, "shape"):
        raise GraphValidationError("X must have a shape attribute")
    shape = X.shape
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise GraphValidationError("X must be a 2D array-like")
    if int(shape[0]) < 0 or int(shape[1]) < 0:
        raise GraphValidationError("X must have non-negative dimensions")


def validate_builder_spec(spec: GraphBuilderSpec) -> None:
    spec.validate()


def validate_featurizer_spec(spec: GraphFeaturizerSpec) -> None:
    spec.validate()


def validate_edge_index(edge_index: Any, *, n_nodes: int) -> None:
    ei = np.asarray(edge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise GraphValidationError("edge_index must have shape (2, E)")
    if ei.size == 0:
        return
    if ei.dtype.kind not in ("i", "u"):
        raise GraphValidationError("edge_index must be integer typed")
    if ei.min() < 0 or ei.max() >= int(n_nodes):
        raise GraphValidationError("edge_index has out-of-range node ids")


def validate_view_matrix(X: Any, *, n_nodes: int, name: str) -> None:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise GraphValidationError(f"view {name!r} must be 2D")
    if int(arr.shape[0]) != int(n_nodes):
        raise GraphValidationError(f"view {name!r} must have {n_nodes} rows")
    if not np.isfinite(arr).all():
        raise GraphValidationError(f"view {name!r} contains non-finite values")
