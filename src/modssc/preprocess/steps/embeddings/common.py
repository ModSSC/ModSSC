from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from modssc.preprocess.store import ArtifactStore


def get_cached_encoder(
    *,
    encoder: Any,
    encoder_device: str | None,
    model_id: str,
    device: str | None,
    load_fn: Callable[..., Any],
) -> tuple[Any, str | None]:
    if encoder is None or encoder_device != device:
        encoder = load_fn(model_id, device=device) if device is not None else load_fn(model_id)
        encoder_device = device
    return encoder, encoder_device


def encode_features_with_cached_encoder(
    *,
    step: Any,
    store: ArtifactStore,
    rng: np.random.Generator,
    load_fn: Callable[..., Any],
    input_mapper: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    raw = store.require("raw.X")
    inputs = input_mapper(raw) if input_mapper is not None else raw
    step._encoder, step._encoder_device = get_cached_encoder(
        encoder=step._encoder,
        encoder_device=step._encoder_device,
        model_id=step.model_id,
        device=step.device,
        load_fn=load_fn,
    )
    emb = step._encoder.encode(inputs, batch_size=int(step.batch_size), rng=rng)
    return {"features.X": np.asarray(emb, dtype=np.float32)}


def make_cached_encoder_transform(
    *,
    load_fn_getter: Callable[[], Callable[..., Any]],
    input_mapper: Callable[[Any], Any] | None = None,
) -> Callable[[Any, ArtifactStore], dict[str, Any]]:
    def transform(step: Any, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        return encode_features_with_cached_encoder(
            step=step,
            store=store,
            rng=rng,
            load_fn=load_fn_getter(),
            input_mapper=input_mapper,
        )

    return transform
