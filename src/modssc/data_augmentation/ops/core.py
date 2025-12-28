from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..registry import register_op
from ..types import AugmentationContext, Modality
from ..utils import is_torch_tensor
from .base import AugmentationOp


@register_op("core.identity")
@dataclass
class Identity(AugmentationOp):
    """Return the input as-is."""

    op_id: str = "core.identity"
    modality: Modality = "any"

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        return x


@register_op("core.ensure_float32")
@dataclass
class EnsureFloat32(AugmentationOp):
    """Cast numeric arrays/tensors to float32.

    This is mainly a convenience op for pipelines that mix augmentations and want stable
    downstream behavior.
    """

    op_id: str = "core.ensure_float32"
    modality: Modality = "any"

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        if is_torch_tensor(x):
            import importlib

            torch = importlib.import_module("torch")
            return x.to(dtype=torch.float32)
        arr = np.asarray(x)
        if arr.dtype == np.float32:
            return x
        return arr.astype(np.float32)
