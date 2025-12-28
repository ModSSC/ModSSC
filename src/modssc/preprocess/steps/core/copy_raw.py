from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.store import ArtifactStore


@dataclass
class CopyRawStep:
    """Copy raw.X into features.X without changing layout or dtype."""

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        x = store.require("raw.X")
        return {"features.X": x}
