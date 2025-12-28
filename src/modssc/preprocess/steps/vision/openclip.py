from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modssc.preprocess.models import load_encoder
from modssc.preprocess.store import ArtifactStore


@dataclass
class OpenClipStep:
    model_id: str = "openclip:ViT-B-32/openai"
    batch_size: int = 32
    device: str | None = None

    _encoder: Any = field(default=None, init=False, repr=False)
    _encoder_device: str | None = field(default=None, init=False, repr=False)

    def _get_encoder(self) -> Any:
        device = self.device
        if self._encoder is None or self._encoder_device != device:
            if device is not None:
                self._encoder = load_encoder(self.model_id, device=device)
            else:
                self._encoder = load_encoder(self.model_id)
            self._encoder_device = device
        return self._encoder

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        X = store.require("raw.X")
        encoder = self._get_encoder()
        emb = encoder.encode(X, batch_size=int(self.batch_size), rng=rng)
        return {"features.X": np.asarray(emb, dtype=np.float32)}
