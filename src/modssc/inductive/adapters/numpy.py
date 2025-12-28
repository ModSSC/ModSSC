from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..base import InductiveDatasetLike
from ..errors import InductiveValidationError
from ..validation import validate_inductive_dataset


def _suggest_step(name: str) -> str:
    if name.startswith("y") or name.endswith(".y") or name in {"y_l", "labels.y"}:
        return "labels.to_numpy"
    return "core.to_numpy"


def _require_numpy(x: Any, *, name: str) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        step = _suggest_step(name)
        raise InductiveValidationError(
            f"{name} must be a numpy.ndarray. Run preprocess step {step} upstream."
        )
    return x


def _require_numpy_views(views: Mapping[str, Any] | None) -> Mapping[str, np.ndarray] | None:
    if views is None:
        return None
    if not isinstance(views, Mapping):
        raise InductiveValidationError("views must be a mapping when provided")
    out: dict[str, np.ndarray] = {}
    for key, value in views.items():
        if not isinstance(key, str):
            raise InductiveValidationError("views keys must be strings")
        out[key] = _require_numpy(value, name=f"views[{key}]")
    return out


@dataclass(frozen=True)
class NumpyDataset:
    """Strict numpy view of an inductive dataset (no implicit conversion)."""

    X_l: np.ndarray
    y_l: np.ndarray
    X_u: np.ndarray | None = None
    X_u_w: np.ndarray | None = None
    X_u_s: np.ndarray | None = None
    views: Mapping[str, np.ndarray] | None = None
    meta: Mapping[str, Any] | None = None


def to_numpy_dataset(data: InductiveDatasetLike) -> NumpyDataset:
    """Validate and wrap an inductive dataset backed by numpy arrays."""
    validate_inductive_dataset(data)

    X_l = _require_numpy(data.X_l, name="X_l")
    y_l = _require_numpy(data.y_l, name="y_l")
    X_u = _require_numpy(data.X_u, name="X_u") if data.X_u is not None else None
    X_u_w = _require_numpy(data.X_u_w, name="X_u_w") if data.X_u_w is not None else None
    X_u_s = _require_numpy(data.X_u_s, name="X_u_s") if data.X_u_s is not None else None
    views = _require_numpy_views(data.views)

    return NumpyDataset(
        X_l=X_l,
        y_l=y_l,
        X_u=X_u,
        X_u_w=X_u_w,
        X_u_s=X_u_s,
        views=views,
        meta=data.meta,
    )
