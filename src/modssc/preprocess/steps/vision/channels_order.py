from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.steps.vision.layout import prepare_image_array
from modssc.preprocess.store import ArtifactStore


@dataclass
class ChannelsOrderStep:
    order: str = "NCHW"  # or "NHWC"

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        X = store.require("raw.X")
        arr = to_numpy(X)
        prepared = prepare_image_array(arr)
        if prepared is None:
            # Not an image batch; leave untouched.
            return {"raw.X": X}

        arr4, single, cur = prepared
        desired = self.order.upper()
        if desired not in {"NCHW", "NHWC"}:
            return {"raw.X": X}

        out = arr4
        if cur != desired:
            if desired == "NCHW":
                out = np.transpose(arr4, (0, 3, 1, 2))
            else:
                out = np.transpose(arr4, (0, 2, 3, 1))

        if single:
            out = out[0]
        return {"raw.X": out}
