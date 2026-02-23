from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modssc.preprocess.models import load_encoder
from modssc.preprocess.steps.embeddings.common import make_cached_encoder_transform


@dataclass
class OpenClipStep:
    model_id: str = "openclip:ViT-B-32/openai"
    batch_size: int = 32
    device: str | None = None

    _encoder: Any = field(default=None, init=False, repr=False)
    _encoder_device: str | None = field(default=None, init=False, repr=False)

    transform = make_cached_encoder_transform(load_fn_getter=lambda: load_encoder)
