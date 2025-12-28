from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.store import ArtifactStore


@dataclass
class Ensure2DStep:
    """Ensure a numeric array representation and store it as features.X."""

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        x = store.get("features.X", store.require("raw.X"))
        arr = to_numpy(x)

        if arr.ndim == 0:
            raise PreprocessValidationError("X must not be a scalar")
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        if arr.dtype == object:
            raise PreprocessValidationError(
                "Ensure2DStep expects numeric arrays. Got dtype=object, did you forget a featurizer step?"
            )

        return {"features.X": arr}
