from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.store import ArtifactStore


@dataclass
class SqueezeStep:
    dim: int = -1
    as_list: bool = True

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        X = store.require("raw.X")
        arr = to_numpy(X)

        # Handle (N, H, W, 1) -> (N, H, W)
        if arr.ndim == 4 and arr.shape[self.dim] == 1:
            out = np.squeeze(arr, axis=self.dim)
            if self.as_list:
                return {"raw.X": list(out)}
            return {"raw.X": out}

        return {"raw.X": X}
