from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from modssc.utils.imports import load_object

from .errors import DataAugmentationValidationError
from .types import AugmentationOp, Modality

_RUNTIME_OPS: dict[str, type[AugmentationOp]] = {}
_OPS = _RUNTIME_OPS
_BUILTIN_OPS: dict[str, tuple[str, Modality]] = {
    "audio.add_noise": ("modssc.data_augmentation.ops.audio:AddNoise", "audio"),
    "audio.time_shift": ("modssc.data_augmentation.ops.audio:TimeShift", "audio"),
    "core.ensure_float32": ("modssc.data_augmentation.ops.core:EnsureFloat32", "any"),
    "core.identity": ("modssc.data_augmentation.ops.core:Identity", "any"),
    "graph.edge_dropout": ("modssc.data_augmentation.ops.graph:EdgeDropout", "graph"),
    "graph.feature_mask": ("modssc.data_augmentation.ops.graph:FeatureMask", "graph"),
    "tabular.feature_dropout": (
        "modssc.data_augmentation.ops.tabular:FeatureDropout",
        "tabular",
    ),
    "tabular.gaussian_noise": (
        "modssc.data_augmentation.ops.tabular:GaussianNoise",
        "tabular",
    ),
    "tabular.swap_noise": ("modssc.data_augmentation.ops.tabular:SwapNoise", "tabular"),
    "text.lowercase": ("modssc.data_augmentation.ops.text:Lowercase", "text"),
    "text.random_swap": ("modssc.data_augmentation.ops.text:RandomSwap", "text"),
    "text.token_mask": ("modssc.data_augmentation.ops.text:TokenMask", "text"),
    "text.token_swap": ("modssc.data_augmentation.ops.text:TokenSwap", "text"),
    "text.word_dropout": ("modssc.data_augmentation.ops.text:WordDropout", "text"),
    "vision.cutout": ("modssc.data_augmentation.ops.vision:Cutout", "vision"),
    "vision.gaussian_noise": (
        "modssc.data_augmentation.ops.vision:GaussianNoise",
        "vision",
    ),
    "vision.random_crop_pad": (
        "modssc.data_augmentation.ops.vision:RandomCropPad",
        "vision",
    ),
    "vision.random_horizontal_flip": (
        "modssc.data_augmentation.ops.vision:RandomHorizontalFlip",
        "vision",
    ),
}


def register_op(op_id: str) -> Callable[[type[AugmentationOp]], type[AugmentationOp]]:
    """Decorator to register an augmentation operation class."""

    def _decorator(cls: type[AugmentationOp]) -> type[AugmentationOp]:
        existing = _RUNTIME_OPS.get(op_id)
        if existing is not None and existing is not cls:
            raise DataAugmentationValidationError(f"Duplicate op_id: {op_id}")
        if not hasattr(cls, "op_id") or not hasattr(cls, "modality"):
            raise DataAugmentationValidationError(
                f"Op class {cls.__name__} must define 'op_id' and 'modality'."
            )
        _RUNTIME_OPS[op_id] = cls
        return cls

    return _decorator


def _load_builtin_cls(op_id: str) -> type[AugmentationOp]:
    try:
        import_path, _ = _BUILTIN_OPS[op_id]
    except KeyError as exc:
        raise DataAugmentationValidationError(f"Unknown op_id: {op_id!r}") from exc
    cls = load_object(import_path)
    if not isinstance(cls, type):
        raise DataAugmentationValidationError(f"Builtin op {op_id!r} did not resolve to a class.")
    return cls


def _get_cls(op_id: str) -> type[AugmentationOp]:
    cls = _RUNTIME_OPS.get(op_id)
    if cls is not None:
        return cls
    return _load_builtin_cls(op_id)


def available_ops(*, modality: Modality | None = None) -> list[str]:
    """List registered operation ids."""
    all_ids = set(_BUILTIN_OPS) | set(_RUNTIME_OPS)
    if modality is None:
        return sorted(all_ids)

    out: list[str] = []
    for op_id in sorted(all_ids):
        runtime_cls = _RUNTIME_OPS.get(op_id)
        if runtime_cls is not None:
            cls_modality = getattr(runtime_cls, "modality", "any")
        else:
            cls_modality = _BUILTIN_OPS[op_id][1]
        if cls_modality == modality or cls_modality == "any":
            out.append(op_id)
    return out


def get_op(op_id: str, **params: Any) -> AugmentationOp:
    """Instantiate an operation from the registry."""
    cls = _get_cls(op_id)
    try:
        return cls(**params)  # type: ignore[call-arg]
    except TypeError as exc:
        raise DataAugmentationValidationError(
            f"Invalid parameters for op {op_id!r}: {exc}"
        ) from exc


def op_info(op_id: str) -> Mapping[str, Any]:
    """Return basic metadata about an operation."""
    cls = _get_cls(op_id)
    inst = cls()  # type: ignore[call-arg]
    defaults: Any = asdict(inst) if is_dataclass(inst) else inst.__dict__
    return {
        "op_id": op_id,
        "modality": getattr(inst, "modality", "any"),
        "doc": (cls.__doc__ or "").strip(),
        "defaults": defaults,
    }
