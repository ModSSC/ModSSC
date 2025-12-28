from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.store import ArtifactStore


@dataclass
class CastDtypeStep:
    dtype: str = "float32"

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        x = store.require("features.X")
        arr = np.asarray(x)
        return {"features.X": arr.astype(self.dtype, copy=False)}
