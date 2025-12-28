from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modssc.preprocess.errors import OptionalDependencyError, PreprocessValidationError
from modssc.preprocess.optional import require
from modssc.preprocess.store import ArtifactStore


def _as_text_array(raw: Any) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        return raw
    if isinstance(raw, (list, tuple)):
        return np.asarray(raw, dtype=object)
    return np.asarray(list(raw), dtype=object)


@dataclass
class TfidfStep:
    max_features: int | None = 20000
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int | float = 1
    max_df: int | float = 1.0

    _vec: Any = field(default=None, init=False, repr=False)

    def fit(
        self, store: ArtifactStore, *, fit_indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        try:
            fe = require(
                module="sklearn.feature_extraction.text",
                extra="preprocess-sklearn",
                purpose="TF-IDF",
            )
        except OptionalDependencyError:
            raise

        raw = store.require("raw.X")
        texts = _as_text_array(raw)
        idx = np.asarray(fit_indices, dtype=np.int64)
        if idx.ndim != 1:
            raise PreprocessValidationError("fit_indices must be 1D")
        texts_fit = [str(x) for x in np.take(texts, idx, axis=0)]

        vec = fe.TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=tuple(self.ngram_range),
            min_df=self.min_df,
            max_df=self.max_df,
        )
        self._vec = vec.fit(texts_fit)

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        if self._vec is None:
            raise PreprocessValidationError("TfidfStep.transform called before fit()")
        raw = store.require("raw.X")
        texts = [str(x) for x in _as_text_array(raw)]
        X = self._vec.transform(texts)
        return {"features.X": X}
