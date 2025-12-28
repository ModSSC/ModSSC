from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.steps.vision.layout import prepare_image_array
from modssc.preprocess.store import ArtifactStore


@dataclass
class NormalizeStep:
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        X = store.require("raw.X")
        arr = to_numpy(X).astype(np.float32, copy=False)
        prepared = prepare_image_array(arr)
        if prepared is None:
            return {"raw.X": X}

        arr4, single, layout = prepared
        ch_axis = 1 if layout == "NCHW" else -1

        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)

        # reshape for broadcasting
        shape = [1] * arr4.ndim
        shape[ch_axis] = mean.size
        mean_b = mean.reshape(shape)
        std_b = std.reshape(shape)

        out = (arr4 - mean_b) / std_b
        if single:
            out = out[0]
        return {"raw.X": out}
