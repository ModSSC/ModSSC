from __future__ import annotations

from functools import lru_cache
from typing import Any

_TORCH_SENTINEL = object()


@lru_cache(maxsize=1)
def mps_is_available(torch: Any) -> bool:
    if torch is None:
        return False
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is None:
        return False
    is_built = getattr(mps, "is_built", None)
    if callable(is_built):
        try:
            if not is_built():
                return False
        except Exception:
            return False
    is_available = getattr(mps, "is_available", None)
    if not callable(is_available):
        return False
    try:
        available = is_available()
    except Exception:
        return False
    if not isinstance(available, bool) or not available:
        return False
    try:
        torch.zeros(1, device="mps")
    except Exception:
        return False
    return True


def _best_device_from_torch(torch: Any) -> str:
    if torch is None:
        return "cpu"
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and getattr(cuda, "is_available", None):
        available = cuda.is_available()
        if isinstance(available, bool) and available:
            return "cuda"
    if mps_is_available(torch):
        return "mps"
    return "cpu"


def _try_import_torch() -> Any | None:
    try:
        import importlib

        return importlib.import_module("torch")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _cached_best_device_name() -> str:
    return _best_device_from_torch(_try_import_torch())


def resolve_device_name(device: str | None, *, torch: Any = _TORCH_SENTINEL) -> str | None:
    if device is None:
        return None
    if device != "auto":
        return device
    if torch is _TORCH_SENTINEL:
        return _cached_best_device_name()
    return _best_device_from_torch(torch)
