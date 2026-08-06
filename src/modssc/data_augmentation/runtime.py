from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .api import AugmentationStrategy
from .types import AugmentationContext
from .utils import is_torch_tensor


def _index_list(indices: Any) -> list[int]:
    """Return device-independent integer indices without changing their order."""

    if is_torch_tensor(indices):
        return [int(value) for value in indices.detach().cpu().reshape(-1).tolist()]
    return [int(value) for value in np.asarray(indices, dtype=np.int64).reshape(-1).tolist()]


def _stack_like(reference: Any, values: list[Any]) -> Any:
    if not values:
        return reference[:0]
    if is_torch_tensor(reference):
        import importlib

        torch = importlib.import_module("torch")
        return torch.stack(values, dim=0)
    return np.stack(values, axis=0)


@dataclass(frozen=True)
class OnlineAugmentation:
    """Recompute deterministic weak/strong views for every optimization step.

    Randomness is keyed by ``(seed, optimization_step, absolute_sample_id)``.
    Consequently a resumed task obtains the same view for the same logical step,
    independently of the order in which samples are materialized.
    """

    strategy: AugmentationStrategy
    seed: int
    modality: str | None = None

    def _context(self, *, sample_id: int, step: int, view: int = 0) -> AugmentationContext:
        if int(step) < 0:
            raise ValueError("step must be >= 0")
        return AugmentationContext(
            seed=int(self.seed) + int(view),
            sample_id=int(sample_id),
            epoch=int(step),
            modality=self.modality,
        )

    def weak_batch(self, X: Any, *, indices: Any, sample_ids: Any, step: int) -> Any:
        local = _index_list(indices)
        absolute = _index_list(sample_ids)
        if len(local) != len(absolute):
            raise ValueError("indices and sample_ids must have the same length")
        values = [
            self.strategy.weak.apply(
                X[index],
                ctx=self._context(sample_id=sample_id, step=int(step)),
            )
            for index, sample_id in zip(local, absolute, strict=True)
        ]
        return _stack_like(X, values)

    def pair_batch(
        self,
        X: Any,
        *,
        indices: Any,
        sample_ids: Any,
        step: int,
    ) -> tuple[Any, Any]:
        local = _index_list(indices)
        absolute = _index_list(sample_ids)
        if len(local) != len(absolute):
            raise ValueError("indices and sample_ids must have the same length")
        weak: list[Any] = []
        strong: list[Any] = []
        for index, sample_id in zip(local, absolute, strict=True):
            xw, xs = self.strategy.apply(
                X[index],
                ctx=self._context(sample_id=sample_id, step=int(step)),
            )
            weak.append(xw)
            strong.append(xs)
        return _stack_like(X, weak), _stack_like(X, strong)
