from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modssc.preprocess.models import load_encoder
from modssc.preprocess.steps.embeddings.common import make_cached_encoder_transform


@dataclass
class Wav2Vec2Step:
    model_id: str = "wav2vec2:base"
    batch_size: int = 8
    device: str | None = None

    _encoder: Any = field(default=None, init=False, repr=False)
    _encoder_device: str | None = field(default=None, init=False, repr=False)

    transform = make_cached_encoder_transform(load_fn_getter=lambda: load_encoder)
