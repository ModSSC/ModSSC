from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.steps.vision.layout import prepare_image_array
from modssc.preprocess.store import ArtifactStore


@dataclass
class EnsureNumChannelsStep:
    num_channels: int = 3

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        X = store.require("raw.X")
        arr = to_numpy(X)
        prepared = prepare_image_array(arr)
        if prepared is None:
            return {"raw.X": X}

        arr4, single, layout = prepared
        ch_axis = 1 if layout == "NCHW" else -1

        C = int(arr4.shape[ch_axis])
        target = int(self.num_channels)

        out = arr4
        if target == C:
            pass
        elif C == 1 and target > 1:
            reps = [1] * out.ndim
            reps[ch_axis] = target
            out = np.repeat(out, repeats=target, axis=ch_axis)
        elif target < C:
            sl = [slice(None)] * out.ndim
            sl[ch_axis] = slice(0, target)
            out = out[tuple(sl)]
        else:
            pad_width = [(0, 0)] * out.ndim
            pad_width[ch_axis] = (0, target - C)
            out = np.pad(out, pad_width=pad_width, mode="constant")

        if single:
            out = out[0]
        return {"raw.X": out}
