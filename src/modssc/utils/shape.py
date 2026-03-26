from __future__ import annotations

from typing import Any


def shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(s) for s in shape)
    except Exception:
        return None
