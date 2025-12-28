from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..backends import torch_backend
from ..base import InductiveDatasetLike
from ..errors import InductiveValidationError
from ..optional import optional_import
from ..types import DeviceSpec


def _torch():
    return optional_import("torch", extra="inductive-torch")


def _suggest_step(name: str) -> str:
    if name.startswith("y") or name.endswith(".y") or name in {"y_l", "labels.y"}:
        return "labels.to_torch"
    return "core.to_torch"


def _require_tensor(x: Any, *, name: str):
    torch = _torch()
    if not isinstance(x, torch.Tensor):
        step = _suggest_step(name)
        raise InductiveValidationError(
            f"{name} must be a torch.Tensor. Run preprocess step {step} upstream."
        )
    return x


def _require_views(views: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if views is None:
        return None
    if not isinstance(views, Mapping):
        raise InductiveValidationError("views must be a mapping when provided")
    out: dict[str, Any] = {}
    for key, value in views.items():
        if not isinstance(key, str):
            raise InductiveValidationError("views keys must be strings")
        out[key] = _require_tensor(value, name=f"views[{key}]")
    return out


def _check_2d(t, *, name: str) -> None:
    if int(t.ndim) != 2:
        raise InductiveValidationError(f"{name} must be 2D (n, d), got shape {tuple(t.shape)}")


def _check_y(t, *, n: int) -> None:
    if int(t.ndim) not in (1, 2):
        raise InductiveValidationError(f"y_l must be 1D or 2D, got shape {tuple(t.shape)}")
    if int(t.shape[0]) != int(n):
        raise InductiveValidationError("X_l and y_l must have the same first dimension")


def _check_feature_dim(t, *, n_features: int, name: str) -> None:
    if int(t.shape[1]) != int(n_features):
        raise InductiveValidationError(f"{name} must have {n_features} features (got {t.shape[1]})")


def _check_same_device(tensors: list[Any]) -> None:
    torch = _torch()
    dev = None
    for t in tensors:
        if t is None:
            continue
        if not isinstance(t, torch.Tensor):
            continue
        if dev is None:
            dev = t.device
            continue
        if t.device != dev:
            raise InductiveValidationError("All tensors must be on the same device")


@dataclass(frozen=True)
class TorchDataset:
    """Strict torch view of an inductive dataset (no implicit conversion)."""

    X_l: Any
    y_l: Any
    X_u: Any | None = None
    X_u_w: Any | None = None
    X_u_s: Any | None = None
    views: Mapping[str, Any] | None = None
    meta: Mapping[str, Any] | None = None


def to_torch_dataset(
    data: InductiveDatasetLike,
    *,
    device: DeviceSpec | None = None,
    require_same_device: bool = True,
) -> TorchDataset:
    """Validate and wrap an inductive dataset backed by torch tensors.

    If device.device == "auto", only device consistency is enforced.
    """
    X_l = _require_tensor(data.X_l, name="X_l")
    y_l = _require_tensor(data.y_l, name="y_l")
    _check_2d(X_l, name="X_l")
    _check_y(y_l, n=int(X_l.shape[0]))

    X_u = _require_tensor(data.X_u, name="X_u") if data.X_u is not None else None
    X_u_w = _require_tensor(data.X_u_w, name="X_u_w") if data.X_u_w is not None else None
    X_u_s = _require_tensor(data.X_u_s, name="X_u_s") if data.X_u_s is not None else None

    n_features = int(X_l.shape[1])
    if X_u is not None:
        _check_2d(X_u, name="X_u")
        _check_feature_dim(X_u, n_features=n_features, name="X_u")
    if X_u_w is not None:
        _check_2d(X_u_w, name="X_u_w")
        _check_feature_dim(X_u_w, n_features=n_features, name="X_u_w")
    if X_u_s is not None:
        _check_2d(X_u_s, name="X_u_s")
        _check_feature_dim(X_u_s, n_features=n_features, name="X_u_s")
    if X_u_w is not None and X_u_s is not None and int(X_u_w.shape[0]) != int(X_u_s.shape[0]):
        raise InductiveValidationError("X_u_w and X_u_s must have the same number of rows")

    views = _require_views(data.views)

    tensors = [X_l, y_l, X_u, X_u_w, X_u_s]
    if views:
        tensors.extend(views.values())

    if require_same_device:
        _check_same_device(tensors)

    if device is not None and device.device != "auto":
        expected = torch_backend.resolve_device(device)
        for t in tensors:
            if t is None:
                continue
            if t.device != expected:
                raise InductiveValidationError(
                    f"Tensor device mismatch: expected {expected}, got {t.device}"
                )

    return TorchDataset(
        X_l=X_l,
        y_l=y_l,
        X_u=X_u,
        X_u_w=X_u_w,
        X_u_s=X_u_s,
        views=views,
        meta=data.meta,
    )
