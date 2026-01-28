from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..registry import register_op
from ..types import AugmentationContext, Modality
from ..utils import is_torch_tensor
from .base import AugmentationOp


@register_op("tabular.gaussian_noise")
@dataclass
class GaussianNoise(AugmentationOp):
    """Add gaussian noise to numeric features."""

    op_id: str = "tabular.gaussian_noise"
    modality: Modality = "tabular"
    std: float = 0.1

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


@register_op("tabular.feature_dropout")
@dataclass
class FeatureDropout(AugmentationOp):
    """Randomly set a subset of features to zero."""

    op_id: str = "tabular.feature_dropout"
    modality: Modality = "tabular"
    p: float = 0.1

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        p = float(self.p)
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        if p == 0.0:
            return x

        if is_torch_tensor(x):
            import importlib

            torch = importlib.import_module("torch")
            mask = torch.from_numpy((rng.random(size=tuple(x.shape)) >= p).astype(np.float32))
            mask = mask.to(device=x.device, dtype=x.dtype)
            return x * mask
        arr = np.asarray(x)
        mask = (rng.random(size=arr.shape) >= p).astype(arr.dtype, copy=False)
        return arr * mask


@register_op("tabular.swap_noise")
@dataclass
class SwapNoise(AugmentationOp):
    """Replace a subset of features with random noise (N(0, 1) approximation)."""

    op_id: str = "tabular.swap_noise"
    modality: Modality = "tabular"
    p: float = 0.15

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:
        p = float(self.p)
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        if p == 0.0:
            return x

        # We assume features are roughly standard normal (common in preprocessed tabular data)
        # SOT: Replace x_ij with sampled value.

        if is_torch_tensor(x):
            import importlib

            torch = importlib.import_module("torch")

            mask_swap = torch.rand(x.shape, device=x.device) < p
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype)

            # Where mask_swap is True, use noise, else x
            return torch.where(mask_swap, noise, x)

        arr = np.asarray(x)
        mask_swap = rng.random(size=arr.shape) < p
        noise = rng.normal(0.0, 1.0, size=arr.shape).astype(arr.dtype)

        out = arr.copy()
        out[mask_swap] = noise[mask_swap]
        return out
