from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modssc.preprocess.errors import OptionalDependencyError, PreprocessValidationError
from modssc.preprocess.optional import require
from modssc.preprocess.steps.base import get_X
from modssc.preprocess.store import ArtifactStore


@dataclass
class TabularStandardScalerStep:
    """Standardize features by removing the mean and scaling to unit variance."""

    with_mean: bool = True
    with_std: bool = True

    _scaler: Any = field(default=None, init=False, repr=False)

    def fit(
        self, store: ArtifactStore, *, fit_indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        try:
            pre = require(
                module="sklearn.preprocessing",
                extra="preprocess-sklearn",
                purpose="Standard scalar scaling for tabular data",
            )
        except OptionalDependencyError:
            raise

        X = get_X(store)
        # Verify it is 2D, or rely on scalar to handle it?
        # Typically ensure_2d runs before this.

        idx = np.asarray(fit_indices, dtype=np.int64)
        # Using numpy slicing. X could be a list if ensuring 2d didn't happen
        # But get_X usually returns the raw object.
        # It's safer to ensure array.
        X_arr = np.asarray(X)
        X_fit = X.iloc[idx].to_numpy() if hasattr(X, "iloc") else X_arr[idx]

        self._scaler = pre.StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        self._scaler.fit(X_fit)

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        if self._scaler is None:
            raise PreprocessValidationError(
                "TabularStandardScalerStep.transform called before fit()"
            )

        X = get_X(store)
        # We need to handle list/df to array conversion if needed,
        # but usually sklearn handles array-likes.
        # However, to be safe and consistent with return types:
        X_scaled = self._scaler.transform(X)

        # Casting to float32 is usually good practice for DL pipelines
        return {"features.X": X_scaled.astype(np.float32)}
