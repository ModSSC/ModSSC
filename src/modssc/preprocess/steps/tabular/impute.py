from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modssc.preprocess.errors import OptionalDependencyError, PreprocessValidationError
from modssc.preprocess.optional import require
from modssc.preprocess.steps.base import get_X
from modssc.preprocess.store import ArtifactStore


@dataclass
class TabularImputeStep:
    """Impute missing values using sklearn SimpleImputer."""

    strategy: str = "mean"
    fill_value: Any = None
    add_indicator: bool = False

    _imputer: Any = field(default=None, init=False, repr=False)

    def fit(
        self, store: ArtifactStore, *, fit_indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        try:
            pre = require(
                module="sklearn.impute",
                extra="preprocess-sklearn",
                purpose="Imputation for tabular data",
            )
        except OptionalDependencyError:
            raise

        X = get_X(store)
        idx = np.asarray(fit_indices, dtype=np.int64)
        X_arr = np.asarray(X)
        X_fit = X_arr[idx]

        # Check if X_fit contains NaNs
        # ModSSC might load '?' as strings if not parsed correctly, or NaNs.
        # SimpleImputer handles NaNs by default.

        self._imputer = pre.SimpleImputer(
            strategy=self.strategy, fill_value=self.fill_value, add_indicator=self.add_indicator
        )
        self._imputer.fit(X_fit)

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        if self._imputer is None:
            raise PreprocessValidationError("TabularImputeStep.transform called before fit()")

        X = get_X(store)
        X_imputed = self._imputer.transform(X)
        return {"features.X": X_imputed}
