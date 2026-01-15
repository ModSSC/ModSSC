from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import InductiveDatasetLike
from .errors import InductiveValidationError


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


def _require_2d(x: Any, *, name: str) -> np.ndarray:
    arr = _as_numpy(x)
    # Relax validation to allow any dimension >= 2 (e.g. images N,C,H,W)
    if arr.ndim < 2:
        raise InductiveValidationError(
            f"{name} must be at least 2D (n, ...), got shape {arr.shape}"
        )
    return arr


def _require_y(y: Any, *, n: int) -> np.ndarray:
    arr = _as_numpy(y)
    if arr.ndim not in (1, 2):
        raise InductiveValidationError(f"y_l must be 1D or 2D, got shape {arr.shape}")
    if arr.shape[0] != n:
        raise InductiveValidationError("X_l and y_l must have the same first dimension")
    return arr


def _validate_views(views: Mapping[str, Any] | None) -> None:
    if views is None:
        return
    if not isinstance(views, Mapping):
        raise InductiveValidationError("views must be a mapping when provided")
    for key in views:
        if not isinstance(key, str):
            raise InductiveValidationError("views keys must be strings")


def _validate_meta(meta: Mapping[str, Any] | None) -> None:
    if meta is None:
        return
    if not isinstance(meta, Mapping):
        raise InductiveValidationError("meta must be a mapping when provided")


def validate_inductive_dataset(data: InductiveDatasetLike) -> None:
    """Validate the minimal invariants needed by inductive algorithms."""
    if data is None:
        raise InductiveValidationError("data must not be None")

    X_l = _require_2d(data.X_l, name="X_l")
    _require_y(data.y_l, n=int(X_l.shape[0]))

    n_features = int(X_l.shape[1])

    if data.X_u is not None:
        X_u = _require_2d(data.X_u, name="X_u")
        if int(X_u.shape[1]) != n_features:
            raise InductiveValidationError("X_u must have the same feature dimension as X_l")

    if data.X_u_w is not None:
        X_u_w = _require_2d(data.X_u_w, name="X_u_w")
        if int(X_u_w.shape[1]) != n_features:
            raise InductiveValidationError("X_u_w must have the same feature dimension as X_l")

    if data.X_u_s is not None:
        X_u_s = _require_2d(data.X_u_s, name="X_u_s")
        if int(X_u_s.shape[1]) != n_features:
            raise InductiveValidationError("X_u_s must have the same feature dimension as X_l")

    if data.X_u_w is not None and data.X_u_s is not None:
        X_u_w = _as_numpy(data.X_u_w)
        X_u_s = _as_numpy(data.X_u_s)
        if X_u_w.shape[0] != X_u_s.shape[0]:
            raise InductiveValidationError("X_u_w and X_u_s must have the same number of rows")

    _validate_views(data.views)
    _validate_meta(data.meta)
