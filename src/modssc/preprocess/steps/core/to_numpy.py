from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.store import ArtifactStore


@dataclass
class ToNumpyStep:
    """Convert features.X to a numpy array (no copy unless needed)."""

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        x = store.require("features.X")
        return {"features.X": to_numpy(x)}
