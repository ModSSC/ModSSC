from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.store import ArtifactStore


@dataclass
class CastFp16Step:
    """Cast features.X to float16 (FP16)."""

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        x = store.require("features.X")
        arr = np.asarray(x)
        return {"features.X": arr.astype(np.float16, copy=False)}
