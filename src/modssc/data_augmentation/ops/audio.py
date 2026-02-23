from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..registry import register_op
from ..types import AugmentationContext, Modality
from ..utils import is_torch_tensor
from .base import AugmentationOp
from .common import apply_gaussian_noise


@register_op("audio.add_noise")
@dataclass
class AddNoise(AugmentationOp):
    """Add gaussian noise to a waveform."""

    op_id: str = "audio.add_noise"
    modality: Modality = "audio"
    std: float = 0.005

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        return apply_gaussian_noise(
            x,
            std=float(self.std),
            rng=rng,
            is_torch_tensor_fn=is_torch_tensor,
        )


@register_op("audio.time_shift")
@dataclass
class TimeShift(AugmentationOp):
    """Circular time shift along the last axis."""

    op_id: str = "audio.time_shift"
    modality: Modality = "audio"
    max_frac: float = 0.1

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        max_frac = float(self.max_frac)
        if not (0.0 <= max_frac <= 1.0):
            raise ValueError("max_frac must be in [0, 1]")
        if max_frac == 0.0:
            return x
        if is_torch_tensor(x):
            length = int(x.shape[-1])
            max_shift = int(round(max_frac * length))
            shift = int(rng.integers(-max_shift, max_shift + 1))
            return x.roll(shifts=shift, dims=-1)
        arr = np.asarray(x)
        length = int(arr.shape[-1])
        max_shift = int(round(max_frac * length))
        shift = int(rng.integers(-max_shift, max_shift + 1))
        return np.roll(arr, shift=shift, axis=-1)
