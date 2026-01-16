from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.steps.vision.layout import prepare_image_array
from modssc.preprocess.store import ArtifactStore


def _resize_nhwc(arr: np.ndarray, *, out_h: int, out_w: int) -> np.ndarray:
    n, h, w, c = arr.shape
    if h == out_h and w == out_w:
        return arr
    row_idx = np.linspace(0, h - 1, out_h).astype(np.int64)
    col_idx = np.linspace(0, w - 1, out_w).astype(np.int64)
    out = arr[:, row_idx, :, :][:, :, col_idx, :]
    return out


@dataclass
class ResizeStep:
    height: int = 224
    width: int = 224

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        X = store.require("raw.X")
        arr = to_numpy(X)
        prepared = prepare_image_array(arr)
        if prepared is None:
            return {"raw.X": X}

        arr4, single, layout = prepared
        arr_nhwc = np.transpose(arr4, (0, 2, 3, 1)) if layout == "NCHW" else arr4

        out_nhwc = _resize_nhwc(arr_nhwc, out_h=int(self.height), out_w=int(self.width))

        out = np.transpose(out_nhwc, (0, 3, 1, 2)) if layout == "NCHW" else out_nhwc
        if single:
            out = out[0]
        # Force uint8 to ensure efficient valid storage (Action 3)
        return {"raw.X": out.astype(np.uint8)}
