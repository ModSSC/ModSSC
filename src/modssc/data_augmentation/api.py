from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import DataAugmentationValidationError
from .plan import AugmentationPlan, StepConfig
from .registry import available_ops
from .registry import get_op as _get_op
from .types import AugmentationContext
from .utils import make_numpy_rng

__all__ = [
    "AugmentationPipeline",
    "AugmentationStrategy",
    "available_ops",
    "build_pipeline",
    "build_strategy",
    "get_op",
    "make_context_rng",
]


def make_context_rng(ctx: AugmentationContext) -> np.random.Generator:
    """Build a deterministic RNG for a given augmentation context."""
    return make_numpy_rng(seed=int(ctx.seed), epoch=int(ctx.epoch), sample_id=int(ctx.sample_id))


@dataclass(frozen=True)
class AugmentationPipeline:
    """A compiled augmentation pipeline."""

    plan: AugmentationPlan
    ops: tuple[Any, ...]

    def apply(self, x: Any, *, ctx: AugmentationContext) -> Any:
        rng = make_context_rng(ctx)
        out = x
        for op in self.ops:
            out = op(out, rng=rng, ctx=ctx)
        return out

    def __call__(self, x: Any, *, ctx: AugmentationContext) -> Any:
        return self.apply(x, ctx=ctx)


@dataclass(frozen=True)
class AugmentationStrategy:
    """Weak/strong strategy container (useful for FixMatch-style algorithms)."""

    weak: AugmentationPipeline
    strong: AugmentationPipeline

    def apply(self, x: Any, *, ctx: AugmentationContext) -> tuple[Any, Any]:
        xw = self.weak.apply(x, ctx=ctx)
        xs = self.strong.apply(x, ctx=ctx)
        return xw, xs


def get_op(op_id: str, **params: Any) -> Any:
    return _get_op(op_id, **params)


def build_pipeline(plan: AugmentationPlan) -> AugmentationPipeline:
    """Compile an :class:`AugmentationPlan` into an executable pipeline."""
    if not isinstance(plan.steps, tuple):
        raise DataAugmentationValidationError("plan.steps must be a tuple of StepConfig")
    ops = []
    for step in plan.steps:
        if not isinstance(step, StepConfig):
            raise DataAugmentationValidationError("Each plan step must be a StepConfig")
        op = _get_op(step.op_id, **(step.params or {}))
        if plan.modality is not None:
            op_modality = getattr(op, "modality", "any")
            if op_modality not in ("any", plan.modality):
                raise DataAugmentationValidationError(
                    f"Op {step.op_id!r} has modality {op_modality!r} but plan expects {plan.modality!r}"
                )
        ops.append(op)
    return AugmentationPipeline(plan=plan, ops=tuple(ops))


def build_strategy(
    *,
    weak: AugmentationPlan,
    strong: AugmentationPlan,
) -> AugmentationStrategy:
    """Convenience helper to build a weak/strong strategy from two plans."""
    return AugmentationStrategy(weak=build_pipeline(weak), strong=build_pipeline(strong))
