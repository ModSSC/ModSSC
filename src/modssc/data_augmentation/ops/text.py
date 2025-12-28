from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..registry import register_op
from ..types import AugmentationContext, Modality
from .base import AugmentationOp


def _as_list(x: Any) -> tuple[list[str], bool]:
    if isinstance(x, str):
        return [x], False
    if isinstance(x, (list, tuple)):
        return [str(s) for s in x], True
    raise TypeError(f"Expected str or list[str], got {type(x).__name__}")


@register_op("text.lowercase")
@dataclass
class Lowercase(AugmentationOp):
    """Lowercase text."""

    op_id: str = "text.lowercase"
    modality: Modality = "text"

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        items, many = _as_list(x)
        out = [s.lower() for s in items]
        return out if many else out[0]


@register_op("text.word_dropout")
@dataclass
class WordDropout(AugmentationOp):
    """Randomly drop words from whitespace-tokenized text."""

    op_id: str = "text.word_dropout"
    modality: Modality = "text"
    p: float = 0.1

    def _apply_one(self, s: str, *, rng: np.random.Generator) -> str:
        tokens = s.split()
        if not tokens:
            return s
        p = float(self.p)
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        keep = [t for t in tokens if rng.random() >= p]
        if not keep:
            keep = [tokens[int(rng.integers(0, len(tokens)))]]
        return " ".join(keep)

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        items, many = _as_list(x)
        out = [self._apply_one(s, rng=rng) for s in items]
        return out if many else out[0]


@register_op("text.random_swap")
@dataclass
class RandomSwap(AugmentationOp):
    """Randomly swap two words N times."""

    op_id: str = "text.random_swap"
    modality: Modality = "text"
    n_swaps: int = 1

    def _apply_one(self, s: str, *, rng: np.random.Generator) -> str:
        tokens = s.split()
        if len(tokens) < 2:
            return s
        n_swaps = max(0, int(self.n_swaps))
        for _ in range(n_swaps):
            i, j = rng.integers(0, len(tokens), size=(2,))
            i = int(i)
            j = int(j)
            tokens[i], tokens[j] = tokens[j], tokens[i]
        return " ".join(tokens)

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:  # noqa: ARG002
        items, many = _as_list(x)
        out = [self._apply_one(s, rng=rng) for s in items]
        return out if many else out[0]
