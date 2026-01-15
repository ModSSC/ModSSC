from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modssc.device import resolve_device_name
from modssc.preprocess.errors import OptionalDependencyError
from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.optional import require

logger = logging.getLogger(__name__)


@dataclass
class Wav2Vec2Encoder:
    bundle: str = "WAV2VEC2_BASE"
    device: str | None = None

    def __post_init__(self) -> None:
        try:
            torch = require(module="torch", extra="preprocess-audio", purpose="wav2vec2")
            torchaudio = require(module="torchaudio", extra="preprocess-audio", purpose="wav2vec2")
        except OptionalDependencyError:
            raise

        self._torch = torch
        self._torchaudio = torchaudio

        pipelines = torchaudio.pipelines
        if not hasattr(pipelines, self.bundle):
            raise ValueError(f"Unknown torchaudio pipeline bundle: {self.bundle!r}")
        b = getattr(pipelines, self.bundle)
        model = b.get_model()
        model.eval()
        self.device = resolve_device_name(self.device, torch=torch)
        self._model = model.to(self.device or "cpu")

    def _load_safe(self, path: str) -> np.ndarray:
        try:
            t_wav, _ = self._torchaudio.load(path)
            return t_wav.numpy()
        except RuntimeError as e:
            if "libtorchcodec" in str(e):
                logger.warning(
                    f"FALLBACK: Torchaudio failed with libtorchcodec error for {path}. "
                    "Falling back to SciPy. Reproducibility vs Torchaudio not guaranteed."
                )
                import scipy.io.wavfile

                try:
                    sr, data = scipy.io.wavfile.read(path)
                except Exception as ex:
                    raise RuntimeError(f"Failed to load audio with scipy fallback: {ex}") from ex

                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128.0) / 128.0
                else:
                    data = data.astype(np.float32)

                data = data[None, :] if data.ndim == 1 else data.T
                return data
            raise e

    def encode(
        self, X: Any, *, batch_size: int = 8, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        torch = self._torch

        # X is a sequence of waveforms (1D arrays) or paths.
        samples = X if isinstance(X, list) else list(X)
        outs: list[np.ndarray] = []
        for wav in samples:
            if isinstance(wav, (str, Path)):
                arr = self._load_safe(str(wav))
            else:
                arr = to_numpy(wav).astype(np.float32, copy=False)

            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 1:
                raise ValueError("wav2vec2 expects 1D waveforms (or shape (1, T))")

            device = next(self._model.parameters()).device
            t = torch.from_numpy(arr).unsqueeze(0).to(device)
            with torch.no_grad():
                feat, _ = self._model(t)
            emb = feat.mean(dim=1).cpu().numpy()
            outs.append(np.asarray(emb, dtype=np.float32))
        return np.concatenate(outs, axis=0) if outs else np.empty((0, 0), dtype=np.float32)
