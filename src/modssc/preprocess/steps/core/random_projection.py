from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.store import ArtifactStore


@dataclass
class RandomProjectionStep:
    n_components: int = 128
    normalize: bool = True

    W_: np.ndarray | None = None

    def fit(
        self, store: ArtifactStore, *, fit_indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        X = np.asarray(store.require("features.X"))
        if X.ndim != 2:
            raise PreprocessValidationError("RandomProjection expects 2D features.X")
        n_features = int(X.shape[1])
        k = int(self.n_components)
        if k <= 0:
            raise PreprocessValidationError("n_components must be > 0")
        W = rng.standard_normal((n_features, k), dtype=np.float32)
        if self.normalize:
            W = W / np.sqrt(float(k))
        self.W_ = W

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        if self.W_ is None:
            raise PreprocessValidationError("RandomProjectionStep.transform called before fit()")
        X = np.asarray(store.require("features.X"), dtype=np.float32)
        Z = X @ self.W_
        return {"features.X": np.asarray(Z, dtype=np.float32)}
