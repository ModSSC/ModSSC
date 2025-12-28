from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import Modality


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
