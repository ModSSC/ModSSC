from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.store import ArtifactStore


@dataclass
class LabelsToNumpyStep:
    """Convert raw.y to numpy and store as labels.y."""

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        y = store.require("raw.y")
        return {"labels.y": to_numpy(y)}
