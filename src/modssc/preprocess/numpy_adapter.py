from __future__ import annotations

import contextlib
from typing import Any

import numpy as np


def to_numpy(x: Any) -> np.ndarray:
    """Best-effort conversion to numpy without requiring optional backends.

    Supports common tensor-like objects (PyTorch, JAX) via duck-typing.
    """
    if isinstance(x, np.ndarray):
        return x
    obj = x
    if hasattr(obj, "detach"):
        try:
            obj = obj.detach()
        except Exception:
            obj = x
    if hasattr(obj, "cpu"):
        with contextlib.suppress(Exception):
            obj = obj.cpu()
    if hasattr(obj, "numpy"):
        try:
            arr = obj.numpy()
            if isinstance(arr, np.ndarray):
                return arr
        except Exception:
            pass
    return np.asarray(obj)
