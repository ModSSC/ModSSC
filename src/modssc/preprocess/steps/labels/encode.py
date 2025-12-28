from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modssc.preprocess.store import ArtifactStore
from modssc.supervised.utils import encode_labels


@dataclass
class EncodeLabelsStep:
    """Encode raw.y to contiguous int64 labels and store labels.y/classes."""

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, object]:  # noqa: ARG002
        y = store.require("raw.y")
        y_enc, classes = encode_labels(y)
        return {"labels.y": y_enc, "labels.classes": classes}
