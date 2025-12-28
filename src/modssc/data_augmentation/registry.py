from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from .errors import DataAugmentationValidationError
from .types import AugmentationOp, Modality

_OPS: dict[str, type[AugmentationOp]] = {}


def register_op(op_id: str) -> Callable[[type[AugmentationOp]], type[AugmentationOp]]:
    """Decorator to register an augmentation operation class."""

    def _decorator(cls: type[AugmentationOp]) -> type[AugmentationOp]:
        if op_id in _OPS:
            raise DataAugmentationValidationError(f"Duplicate op_id: {op_id}")
        # basic sanity: the class should expose op_id/modality defaults
        if not hasattr(cls, "op_id") and not hasattr(cls, "modality"):
            raise DataAugmentationValidationError(
                f"Op class {cls.__name__} must define 'op_id' and 'modality'."
            )
        _OPS[op_id] = cls
        return cls

    return _decorator


def available_ops(*, modality: Modality | None = None) -> list[str]:
    """List registered operation ids."""
    if modality is None:
        return sorted(_OPS.keys())
    out: list[str] = []
    for k, cls in _OPS.items():
        m = getattr(cls, "modality", "any")
        if m == modality or m == "any":
            out.append(k)
    return sorted(out)


def get_op(op_id: str, **params: Any) -> AugmentationOp:
    """Instantiate an operation from the registry."""
    if op_id not in _OPS:
        raise DataAugmentationValidationError(f"Unknown op_id: {op_id!r}")
    cls = _OPS[op_id]
    try:
        return cls(**params)  # type: ignore[call-arg]
    except TypeError as e:
        raise DataAugmentationValidationError(f"Invalid parameters for op {op_id!r}: {e}") from e


def op_info(op_id: str) -> Mapping[str, Any]:
    """Return basic metadata about an operation."""
    if op_id not in _OPS:
        raise DataAugmentationValidationError(f"Unknown op_id: {op_id!r}")
    cls = _OPS[op_id]
    # instantiate with defaults (dataclass ops should allow this)
    inst = cls()  # type: ignore[call-arg]
    defaults: Any = asdict(inst) if is_dataclass(inst) else inst.__dict__
    return {
        "op_id": op_id,
        "modality": getattr(inst, "modality", "any"),
        "doc": (cls.__doc__ or "").strip(),
        "defaults": defaults,
    }


def _ensure_builtin_registered() -> None:
    """Import builtins so they self-register."""
    # Import side-effect modules once.
    from . import ops  # noqa: F401  # pylint: disable=unused-import


# Public: ensure registry is populated on first import.
_ensure_builtin_registered()
