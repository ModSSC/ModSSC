from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..registry import register_op
from ..types import AugmentationContext, Modality
from ..utils import is_torch_tensor
from .base import AugmentationOp


@register_op("audio.add_noise")
@dataclass
class AddNoise(AugmentationOp):
    """Add gaussian noise to a waveform."""

    op_id: str = "audio.add_noise"
    modality: Modality = "audio"
    std: float = 0.005

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        std = float(self.std)
        if std < 0:
            raise ValueError("std must be >= 0")
        if std == 0:
            return x
        if is_torch_tensor(x):
            import importlib

            torch = importlib.import_module("torch")
            seed = int(rng.integers(0, 1 << 31))
            gen = torch.Generator(device=x.device).manual_seed(seed)
            noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * std
            return x + noise
        arr = np.asarray(x)
        noise = rng.normal(0.0, std, size=arr.shape).astype(arr.dtype, copy=False)
        return arr + noise


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
