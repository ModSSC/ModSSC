from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return {"__type__": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        try:
            shape = list(obj.shape)
        except Exception:
            shape = None
        return {
            "__type__": type(obj).__name__,
            "shape": shape,
            "dtype": str(getattr(obj, "dtype", "")),
        }
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if len(obj) > 50:
            return {"__type__": type(obj).__name__, "len": len(obj)}
        return [to_jsonable(v) for v in obj]
    return {"__type__": type(obj).__name__}


def mapping_to_jsonable(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k): to_jsonable(v) for k, v in meta.items()}
