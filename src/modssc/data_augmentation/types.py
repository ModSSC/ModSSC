from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

Modality = Literal["vision", "text", "tabular", "audio", "graph", "any"]
Backend = Literal["auto", "numpy", "torch"]

ArrayLike = Any


@dataclass(frozen=True)
class AugmentationContext:
    """Deterministic context for augmentation.

    Parameters
    ----------
    seed:
        Global seed for the experiment.
    sample_id:
        A stable identifier for the sample (e.g. dataset index).
    epoch:
        Current training epoch (or 0 for stateless usage).
    backend:
        Backend preference. "auto" uses torch if the input is a torch tensor.
    modality:
        Optional modality hint (used for validation only).
    """

    seed: int
    sample_id: int = 0
    epoch: int = 0
    backend: Backend = "auto"
    modality: Modality | None = None


@dataclass(frozen=True)
class GraphSample:
    """Minimal graph container for augmentation.

    This is intentionally small and compatible with common graph toolkits:
    it mirrors the key fields of PyG's ``Data`` (``x``, ``edge_index``, ``edge_weight``).

    Notes
    -----
    ``edge_index`` is expected to be shaped ``(2, E)`` or ``(E, 2)``. Augmentations will
    normalize it to ``(2, E)`` internally.
    """

    x: ArrayLike
    edge_index: ArrayLike
    edge_weight: ArrayLike | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def num_nodes(self) -> int:
        x = np.asarray(self.x)
        return int(x.shape[0])


class SupportsApply(Protocol):
    op_id: str
    modality: Modality

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any: ...


@dataclass
class AugmentationOp:
    """Base class for augmentation ops."""

    op_id: str
    modality: Modality = "any"

    def apply(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:
        raise NotImplementedError

    def __call__(self, x: Any, *, rng: np.random.Generator, ctx: AugmentationContext) -> Any:
        return self.apply(x, rng=rng, ctx=ctx)
