from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.store import ArtifactStore


@dataclass
class EnsureOneHotLabelsStep:
    unlabeled_value: int = -1
    dtype: str = "float32"

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        y = np.asarray(store.require("raw.y"))
        y_flat = y.reshape(-1)
        labeled = y_flat != int(self.unlabeled_value)

        if labeled.any():
            classes = np.unique(y_flat[labeled])
        else:
            classes = np.asarray([], dtype=y_flat.dtype)

        C = int(classes.size)
        onehot = np.zeros((y_flat.size, C), dtype=self.dtype)
        if C > 0 and labeled.any():
            rows = np.nonzero(labeled)[0]
            cols = np.searchsorted(classes, y_flat[labeled])
            onehot[rows, cols] = 1.0

        return {"labels.y_onehot": onehot, "labels.is_labeled": labeled.astype(bool)}
