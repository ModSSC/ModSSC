from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.store import ArtifactStore


@dataclass
class AttachEdgeWeightStep:
    weight: float = 1.0

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        if store.has("graph.edge_weight"):
            return {"graph.edge_weight": store.require("graph.edge_weight")}

        edge_index = to_numpy(store.require("graph.edge_index"))
        if edge_index.ndim != 2:
            edge_index = np.asarray(edge_index)
        E = int(edge_index.shape[1]) if edge_index.shape[0] == 2 else int(edge_index.shape[0])
        w = np.full((E,), float(self.weight), dtype=np.float32)
        return {"graph.edge_weight": w}
