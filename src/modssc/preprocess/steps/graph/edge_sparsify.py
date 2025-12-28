from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.store import ArtifactStore


@dataclass
class EdgeSparsifyStep:
    keep_fraction: float = 1.0

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        frac = float(self.keep_fraction)
        if not (0.0 < frac <= 1.0):
            raise PreprocessValidationError("keep_fraction must be in (0, 1]")

        edge_index = to_numpy(store.require("graph.edge_index"))
        edge_weight = store.get("graph.edge_weight")

        if edge_index.ndim != 2:
            edge_index = np.asarray(edge_index)

        if edge_index.shape[0] == 2:
            E = int(edge_index.shape[1])
            take = rng.random(E) < frac
            if take.sum() == 0 and E > 0:
                take[rng.integers(0, E)] = True
            ei = edge_index[:, take]
        else:
            E = int(edge_index.shape[0])
            take = rng.random(E) < frac
            if take.sum() == 0 and E > 0:
                take[rng.integers(0, E)] = True
            ei = edge_index[take, :]

        out: dict[str, Any] = {"graph.edge_index": ei}
        if edge_weight is not None:
            w = to_numpy(edge_weight)
            out["graph.edge_weight"] = w[take]
        return out
