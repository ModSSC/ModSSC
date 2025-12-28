from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.models import load_encoder
from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.store import ArtifactStore

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def _first_item(X: Any) -> Any | None:
    if isinstance(X, np.ndarray):
        if int(X.size) == 0:
            return None
        return X.flat[0]
    if isinstance(X, (list, tuple)):
        return X[0] if X else None
    return None


def _is_audio_path(value: Any) -> bool:
    if not isinstance(value, (str, bytes, bytearray, Path)):
        return False
    try:
        suffix = Path(str(value)).suffix.lower()
    except Exception:
        return False
    return suffix in _AUDIO_EXTS


def _looks_like_text(X: Any) -> bool:
    first = _first_item(X)
    if isinstance(first, str):
        return not _is_audio_path(first)
    if isinstance(X, np.ndarray) and X.dtype.kind in {"U", "S"}:
        if int(X.size) == 0:
            return False
        return not _is_audio_path(X.flat[0])
    return False


def _looks_like_images(X: Any) -> bool:
    return isinstance(X, np.ndarray) and X.ndim in {3, 4}


def _looks_like_audio(X: Any) -> bool:
    first = _first_item(X)
    if _is_audio_path(first):
        return True
    if isinstance(first, (list, tuple, np.ndarray)):
        arr = to_numpy(first)
        return arr.ndim in {1, 2}
    return False


@dataclass
class AutoEmbeddingStep:
    """Choose an encoder by inspecting raw.X and write dense embeddings to features.X.

    Defaults are offline stub encoders, so this step is safe in base installs.
    """

    model_id_text: str = "stub:text"
    model_id_vision: str = "stub:vision"
    model_id_audio: str = "stub:audio"
    batch_size: int = 32

    _encoder_text: Any = field(default=None, init=False, repr=False)
    _encoder_vision: Any = field(default=None, init=False, repr=False)
    _encoder_audio: Any = field(default=None, init=False, repr=False)

    def _get_encoder(self, kind: str) -> Any:
        if kind == "text":
            if self._encoder_text is None:
                self._encoder_text = load_encoder(self.model_id_text)
            return self._encoder_text
        if kind == "vision":
            if self._encoder_vision is None:
                self._encoder_vision = load_encoder(self.model_id_vision)
            return self._encoder_vision
        if kind == "audio":
            if self._encoder_audio is None:
                self._encoder_audio = load_encoder(self.model_id_audio)
            return self._encoder_audio
        raise PreprocessValidationError(f"Unknown encoder kind: {kind!r}")

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        X = store.require("raw.X")

        if _looks_like_text(X):
            encoder = self._get_encoder("text")
            emb = encoder.encode(X, batch_size=int(self.batch_size), rng=rng)
            return {"features.X": np.asarray(emb, dtype=np.float32)}

        if _looks_like_images(X):
            encoder = self._get_encoder("vision")
            emb = encoder.encode(X, batch_size=int(self.batch_size), rng=rng)
            return {"features.X": np.asarray(emb, dtype=np.float32)}

        if _looks_like_audio(X):
            encoder = self._get_encoder("audio")
            emb = encoder.encode(X, batch_size=int(self.batch_size), rng=rng)
            return {"features.X": np.asarray(emb, dtype=np.float32)}

        # Fallback: numeric arrays.
        arr = to_numpy(X)
        if arr.ndim == 0:
            raise PreprocessValidationError("Cannot embed scalar raw.X")
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.dtype == object:
            raise PreprocessValidationError("AutoEmbeddingStep could not infer modality for raw.X")
        return {"features.X": np.asarray(arr, dtype=np.float32)}
