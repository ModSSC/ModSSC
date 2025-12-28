from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.base import fit_subset, get_X
from modssc.preprocess.store import ArtifactStore


@dataclass
class PcaStep:
    n_components: int = 32
    center: bool = True

    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    impute_: np.ndarray | None = None

    def fit(
        self, store: ArtifactStore, *, fit_indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        X = get_X(store)
        X_fit = np.array(fit_subset(X, fit_indices=fit_indices), dtype=np.float64, copy=True)

        if X_fit.ndim != 2:
            raise PreprocessValidationError("PCA expects a 2D features matrix")
        n_samples, n_features = X_fit.shape
        if n_samples == 0:
            raise PreprocessValidationError("Cannot fit PCA on empty selection")

        finite = np.isfinite(X_fit)
        if finite.all():
            impute = X_fit.mean(axis=0)
        else:
            impute = np.mean(X_fit, axis=0, where=finite)
            impute = np.where(np.isfinite(impute), impute, 0.0)
            np.copyto(X_fit, impute, where=~finite)

        mean = X_fit.mean(axis=0) if self.center else np.zeros(n_features, dtype=np.float64)
        if self.center:
            X_fit -= mean
        # SVD for PCA components
        # Xc = U S Vt
        _, _, vt = np.linalg.svd(X_fit, full_matrices=False)
        k = int(self.n_components)
        k = max(1, min(k, int(vt.shape[0]), int(n_features)))
        self.mean_ = mean.astype(np.float64, copy=False)
        self.components_ = vt[:k].astype(np.float64, copy=False)
        self.impute_ = np.asarray(impute, dtype=np.float64)

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        if self.mean_ is None or self.components_ is None or self.impute_ is None:
            raise PreprocessValidationError("PcaStep.transform called before fit()")
        X = np.array(store.require("features.X"), dtype=np.float64, copy=True)
        finite = np.isfinite(X)
        if not finite.all():
            np.copyto(X, self.impute_, where=~finite)
        if self.center:
            X -= self.mean_
        Z = X @ self.components_.T
        return {"features.X": np.asarray(Z, dtype=np.float32)}
