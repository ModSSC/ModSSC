from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.store import ArtifactStore


@dataclass
class ZcaWhiteningStep:
    eps: float = 1e-5
    max_features: int = 4096

    mean_: np.ndarray | None = None
    W_: np.ndarray | None = None
    orig_shape_: tuple[int, ...] | None = None

    def fit(
        self, store: ArtifactStore, *, fit_indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        X = to_numpy(store.require("raw.X")).astype(np.float64, copy=False)
        if X.ndim < 2:
            raise PreprocessValidationError("ZCA expects an array with at least 2 dimensions")

        X_fit = X[np.asarray(fit_indices, dtype=np.int64)]
        n = int(X_fit.shape[0])
        if n == 0:
            raise PreprocessValidationError("Cannot fit ZCA on empty selection")

        flat = X_fit.reshape(n, -1)
        d = int(flat.shape[1])
        if d > int(self.max_features):
            raise PreprocessValidationError(
                f"ZCA feature dimension too large ({d}). Increase max_features or use an embedding step."
            )

        mu = flat.mean(axis=0)
        Xc = flat - mu
        cov = (Xc.T @ Xc) / float(max(1, n))
        s, u = np.linalg.eigh(cov)
        s = np.maximum(s, 0.0)
        inv = 1.0 / np.sqrt(s + float(self.eps))
        W = (u * inv) @ u.T

        self.mean_ = mu
        self.W_ = W
        self.orig_shape_ = tuple(X.shape[1:])

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        if self.mean_ is None or self.W_ is None or self.orig_shape_ is None:
            raise PreprocessValidationError("ZcaWhiteningStep.transform called before fit()")

        X = to_numpy(store.require("raw.X")).astype(np.float64, copy=False)
        if X.ndim < 2:
            raise PreprocessValidationError("ZCA expects an array with at least 2 dimensions")
        n = int(X.shape[0])
        flat = X.reshape(n, -1)
        if flat.shape[1] != self.mean_.shape[0]:
            raise PreprocessValidationError("ZCA input dimension mismatch")
        out = (flat - self.mean_) @ self.W_
        out = out.reshape((n,) + self.orig_shape_)
        return {"raw.X": out.astype(np.float32)}
