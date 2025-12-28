from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.store import ArtifactStore


def get_X(store: ArtifactStore) -> Any:
    """Return the best available X input (prefer features.X over raw.X)."""
    if "features.X" in store:
        return store.get("features.X")
    return store.require("raw.X")


def set_X(store: ArtifactStore, value: Any, *, key: str = "features.X") -> dict[str, Any]:
    return {key: value}


def fit_subset(X: Any, *, fit_indices: np.ndarray) -> Any:
    idx = np.asarray(fit_indices, dtype=np.int64)
    if idx.ndim != 1:
        raise PreprocessValidationError("fit_indices must be 1D")
    arr = to_numpy(X)
    if arr.ndim == 0:
        raise PreprocessValidationError("Cannot subset scalar X")
    return arr[idx]


@dataclass
class TransformStep:
    """A pure transform step."""

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class FittableStep(TransformStep):
    """A step that fits state on a subset of the training split."""

    def fit(
        self, store: ArtifactStore, *, fit_indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        raise NotImplementedError
