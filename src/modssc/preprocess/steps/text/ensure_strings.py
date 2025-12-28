from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.store import ArtifactStore


@dataclass
class EnsureStringsStep:
    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        X = store.require("raw.X")
        out = [str(x) for x in X.reshape(-1)] if isinstance(X, np.ndarray) else [str(x) for x in X]
        return {"raw.X": out}
