from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from .errors import DataAugmentationValidationError
from .types import Modality

__all__ = ["AugmentationPlan", "StepConfig", "parse_augmentation_plan"]


@dataclass(frozen=True)
class StepConfig:
    """A single augmentation step.

    Parameters
    ----------
    op_id:
        Registry id of the augmentation operation (e.g. ``"vision.random_horizontal_flip"``).
    params:
        Keyword parameters forwarded to the op constructor.
    """

    op_id: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AugmentationPlan:
    """A sequence of augmentation steps.

    Notes
    -----
    Unlike preprocessing, augmentation is usually applied *online* (during training).
    Plans are still useful to describe pipelines declaratively and reproducibly.
    """

    steps: tuple[StepConfig, ...]
    modality: Modality | None = None
    description: str | None = None


def _reject_unknown_keys(data: Mapping[str, Any], *, allowed: set[str], path: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise DataAugmentationValidationError(f"Unknown keys in {path}: {sorted(unknown)}")


def parse_augmentation_plan(
    value: AugmentationPlan | Mapping[str, Any],
    *,
    modality: Modality | None = None,
) -> AugmentationPlan:
    """Parse a declarative augmentation plan into the native plan type.

    Both ``id`` (the YAML spelling) and ``op_id`` (the Python spelling) are
    accepted for steps.  A modality declared in the plan must agree with the
    modality supplied by the caller, so the executable contract cannot silently
    differ from the serialized one.
    """

    if isinstance(value, AugmentationPlan):
        if modality is not None and value.modality not in {None, modality}:
            raise DataAugmentationValidationError(
                f"Augmentation plan modality {value.modality!r} conflicts with {modality!r}"
            )
        if modality is None or value.modality == modality:
            return value
        return AugmentationPlan(
            steps=value.steps,
            modality=modality,
            description=value.description,
        )

    if not isinstance(value, Mapping):
        raise DataAugmentationValidationError("augmentation plan must be a mapping")
    _reject_unknown_keys(
        value,
        allowed={"steps", "modality", "description"},
        path="augmentation plan",
    )

    declared_modality_raw = value.get("modality")
    declared_modality = (
        None if declared_modality_raw is None else cast(Modality, str(declared_modality_raw))
    )
    if modality is not None and declared_modality not in {None, modality}:
        raise DataAugmentationValidationError(
            f"Augmentation plan modality {declared_modality!r} conflicts with {modality!r}"
        )
    resolved_modality = modality if modality is not None else declared_modality

    steps_raw = value.get("steps", [])
    if not isinstance(steps_raw, list):
        raise DataAugmentationValidationError("augmentation.steps must be a list")

    steps: list[StepConfig] = []
    for index, item in enumerate(steps_raw):
        if not isinstance(item, Mapping):
            raise DataAugmentationValidationError("Each augmentation step must be a mapping")
        _reject_unknown_keys(
            item,
            allowed={"id", "op_id", "params"},
            path=f"augmentation.steps[{index}]",
        )
        if "id" in item and "op_id" in item and str(item["id"]) != str(item["op_id"]):
            raise DataAugmentationValidationError(
                f"augmentation.steps[{index}] has conflicting 'id' and 'op_id' values"
            )
        op_id = str(item.get("id") or item.get("op_id") or "")
        if not op_id:
            raise DataAugmentationValidationError("Each augmentation step must define 'id'")
        params_raw = item.get("params", {})
        params = {} if params_raw is None else params_raw
        if not isinstance(params, Mapping):
            raise DataAugmentationValidationError(f"params for op {op_id!r} must be a mapping")
        steps.append(StepConfig(op_id=op_id, params=dict(params)))

    description_raw = value.get("description")
    description = None if description_raw is None else str(description_raw)
    return AugmentationPlan(
        steps=tuple(steps),
        modality=resolved_modality,
        description=description,
    )
