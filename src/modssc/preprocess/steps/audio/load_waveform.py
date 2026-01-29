from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.optional import require
from modssc.preprocess.store import ArtifactStore

logger = logging.getLogger(__name__)

# Track if we have already warned about libtorchcodec fallback to avoid log spam
_LIBTORCHCODEC_FALLBACK_WARNED = False


def _is_path_like(value: Any) -> bool:
    return isinstance(value, (str, bytes, bytearray, Path))


def _as_numpy_waveform(arr: Any, *, mono: bool) -> np.ndarray:
    wave = to_numpy(arr)
    if wave.ndim == 1:
        return wave.astype(np.float32, copy=False)
    if wave.ndim == 2:
        if mono:
            return wave.mean(axis=0, dtype=np.float32)
        # flatten channels for downstream (C, T) -> (T,) via mean to keep shape stable
        return wave.mean(axis=0, dtype=np.float32)
    raise PreprocessValidationError("Audio waveform must be 1D or 2D.")


def _trim_waveform(wave: np.ndarray, *, max_length: int, trim: str) -> np.ndarray:
    length = int(wave.shape[0])
    if length <= max_length:
        return wave
    if trim == "center":
        start = (length - max_length) // 2
        return wave[start : start + max_length]
    if trim == "end":
        return wave[length - max_length :]
    return wave[:max_length]


def _pad_waveform(wave: np.ndarray, *, max_length: int, pad_value: float) -> np.ndarray:
    length = int(wave.shape[0])
    if length >= max_length:
        return wave
    out = np.full((max_length,), float(pad_value), dtype=np.float32)
    out[:length] = wave
    return out


@dataclass
class LoadWaveformStep:
    """Load audio waveforms from file paths into dense features.X."""

    target_sample_rate: int | None = None
    max_length: int | None = None
    pad_value: float = 0.0
    mono: bool = True
    trim: str = "start"  # start, center, end
    allow_fallback: bool = False

    def _load_path(self, path: Any) -> tuple[np.ndarray, int]:
        torchaudio = require(
            module="torchaudio", extra="preprocess-audio", purpose="audio.load_waveform"
        )
        try:
            waveform, sr = torchaudio.load(str(path))
        except RuntimeError as e:
            if "libtorchcodec" in str(e):
                if not self.allow_fallback:
                    raise PreprocessValidationError(
                        "Torchaudio failed to load audio due to a libtorchcodec error. "
                        "Install a torchaudio build with codec support or set "
                        "LoadWaveformStep.allow_fallback=True to use the SciPy fallback."
                    ) from e
                global _LIBTORCHCODEC_FALLBACK_WARNED
                if not _LIBTORCHCODEC_FALLBACK_WARNED:
                    logger.warning(
                        f"FALLBACK: Torchaudio failed with libtorchcodec error for {path}. "
                        "Falling back to SciPy. Reproducibility vs Torchaudio not guaranteed. "
                        "(Subsequent warnings suppressed)"
                    )
                    _LIBTORCHCODEC_FALLBACK_WARNED = True
                else:
                    logger.debug(
                        f"FALLBACK: Torchaudio failed with libtorchcodec error for {path}. "
                        "Falling back to SciPy."
                    )

                import scipy.io.wavfile
                import torch

                try:
                    sr, data = scipy.io.wavfile.read(str(path))
                    if data.dtype == np.int16:
                        data = data.astype(np.float32) / 32768.0
                    elif data.dtype == np.int32:
                        data = data.astype(np.float32) / 2147483648.0
                    elif data.dtype == np.uint8:
                        data = (data.astype(np.float32) - 128.0) / 128.0
                    else:
                        data = data.astype(np.float32)

                    data = data[None, :] if data.ndim == 1 else data.T
                    waveform = torch.from_numpy(data)
                except Exception as ex:
                    raise RuntimeError(f"Failed to load audio with scipy fallback: {ex}") from ex
            else:
                raise e

        if self.target_sample_rate is not None and int(sr) != int(self.target_sample_rate):
            waveform = torchaudio.functional.resample(
                waveform, int(sr), int(self.target_sample_rate)
            )
            sr = int(self.target_sample_rate)
        wave_np = _as_numpy_waveform(waveform, mono=bool(self.mono))
        return wave_np, int(sr)

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        X = store.require("raw.X")

        if isinstance(X, np.ndarray) and X.dtype != object and X.ndim == 2:
            waves = [np.asarray(row) for row in X]
            sample_rates: list[int | None] = [None] * len(waves)
        else:
            items: Iterable[Any]
            if isinstance(X, np.ndarray):
                items = [X.item()] if X.ndim == 0 else X
            elif isinstance(X, (list, tuple)):
                items = X
            else:
                items = [X]

            waves = []
            sample_rates = []
            for item in items:
                if _is_path_like(item):
                    wave, sr = self._load_path(item)
                    waves.append(wave)
                    sample_rates.append(sr)
                else:
                    wave = _as_numpy_waveform(item, mono=bool(self.mono))
                    waves.append(wave)
                    sample_rates.append(None)

        if not waves:
            return {"features.X": np.asarray([], dtype=np.float32).reshape(0, 0)}

        if self.trim not in {"start", "center", "end"}:
            raise PreprocessValidationError("trim must be one of: start, center, end.")

        if self.max_length is not None:
            max_len = int(self.max_length)
            if max_len <= 0:
                raise PreprocessValidationError("max_length must be >= 1.")
            processed = []
            for wave in waves:
                wave = _trim_waveform(wave, max_length=max_len, trim=self.trim)
                wave = _pad_waveform(wave, max_length=max_len, pad_value=float(self.pad_value))
                processed.append(wave)
            waves = processed
        else:
            lengths = {int(w.shape[0]) for w in waves}
            if len(lengths) > 1:
                raise PreprocessValidationError(
                    "audio.load_waveform requires max_length to batch variable-length audio."
                )

        arr = np.stack([w.astype(np.float32, copy=False) for w in waves], axis=0)
        if self.target_sample_rate is not None:
            missing_sr = any(sr is None for sr in sample_rates)
            if missing_sr:
                raise PreprocessValidationError(
                    "target_sample_rate is set but some inputs do not have a sample rate."
                )
        return {"features.X": arr}
