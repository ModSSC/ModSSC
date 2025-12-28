from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from .patching import flatten_patch
from .samplers import sample_distribution, sample_from_values, validate_distribution
from .types import DistributionSpec, HpoError, Trial


@dataclass(frozen=True)
class _Leaf:
    path: tuple[str, ...]
    values: list[Any] | None
    dist: DistributionSpec | None


class Space:
    def __init__(self, leaves: list[_Leaf]) -> None:
        if not leaves:
            raise HpoError("space must define at least one leaf")
        self._leaves = tuple(leaves)

    @classmethod
    def from_dict(cls, d: dict) -> Space:
        if not isinstance(d, Mapping):
            raise HpoError("space must be a mapping")
        if not d:
            raise HpoError("space cannot be empty")

        leaves: list[_Leaf] = []

        def _walk(node: Any, prefix: tuple[str, ...]) -> None:
            if isinstance(node, list):
                if not node:
                    raise HpoError("space leaves must be non-empty lists")
                leaves.append(_Leaf(prefix, list(node), None))
                return

            if isinstance(node, Mapping):
                if not node:
                    raise HpoError("space cannot contain empty mappings")
                if "dist" in node:
                    if not prefix:
                        raise HpoError("space root must be a mapping of parameter keys")
                    allowed = {"dist", "low", "high", "values"}
                    extra = set(node.keys()) - allowed
                    if extra:
                        raise HpoError(f"Unknown keys in dist spec: {sorted(extra)}")
                    dist = node.get("dist")
                    low = node.get("low")
                    high = node.get("high")
                    values = node.get("values")
                    if values is not None and not isinstance(values, list):
                        raise HpoError("values must be a list when provided")
                    spec = DistributionSpec(dist=str(dist), low=low, high=high, values=values)
                    validate_distribution(spec)
                    leaf_values = (
                        list(values) if spec.dist == "choice" and values is not None else None
                    )
                    leaves.append(_Leaf(prefix, leaf_values, spec))
                    return

                for key in sorted(node.keys()):
                    if not isinstance(key, str) or not key:
                        raise HpoError("space keys must be non-empty strings")
                    _walk(node[key], prefix + (key,))
                return

            raise HpoError("space leaves must be lists or dist specs")

        _walk(d, ())
        return cls(leaves)

    def iter_grid(self) -> Iterator[Trial]:
        grids: list[list[Any]] = []
        for leaf in self._leaves:
            if leaf.values is None:
                raise HpoError("grid search requires list or choice distributions")
            grids.append(list(leaf.values))

        for index, combo in enumerate(product(*grids)):
            patch: dict[str, Any] = {}
            for leaf, value in zip(self._leaves, combo, strict=True):
                _assign(patch, leaf.path, value)
            yield Trial(index=index, patch=patch, params=flatten_patch(patch))

    def iter_random(self, seed: int, n_trials: int) -> Iterator[Trial]:
        if n_trials <= 0:
            raise HpoError("n_trials must be > 0")
        rng = np.random.default_rng(seed)

        for index in range(int(n_trials)):
            patch: dict[str, Any] = {}
            for leaf in self._leaves:
                if leaf.dist is None:
                    value = sample_from_values(leaf.values or [], rng)
                else:
                    value = sample_distribution(leaf.dist, rng)
                _assign(patch, leaf.path, value)
            yield Trial(index=index, patch=patch, params=flatten_patch(patch))


def _assign(patch: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = patch
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value
