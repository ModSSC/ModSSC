from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.optional import require
from modssc.preprocess.store import ArtifactStore


@dataclass
class LogMelSpectrogramStep:
    """Compute Log-Mel Spectrograms from waveforms."""

    sample_rate: int = 16000
    n_fft: int = 400
    win_length: int | None = None
    hop_length: int | None = None
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = None
    top_db: float = 80.0

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        torch = require(
            module="torch", extra="preprocess-audio", purpose="audio.log_mel_spectrogram"
        )
        torchaudio = require(
            module="torchaudio", extra="preprocess-audio", purpose="audio.log_mel_spectrogram"
        )

        waveforms_raw = store.require("features.X")

        # Ensure we have a consistent tensor or list of tensors
        if isinstance(waveforms_raw, np.ndarray):
            waveforms = torch.from_numpy(waveforms_raw)
        else:
            # list of arrays
            waveforms = torch.as_tensor(np.stack(waveforms_raw), dtype=torch.float32)

        # waveforms shape should be (N, T) or (N, 1, T)
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)  # (N, 1, T)

        # Define transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(self.sample_rate),
            n_fft=int(self.n_fft),
            win_length=int(self.win_length) if self.win_length else None,
            hop_length=int(self.hop_length) if self.hop_length else None,
            n_mels=int(self.n_mels),
            f_min=float(self.f_min),
            f_max=float(self.f_max) if self.f_max else None,
        )

        amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=float(self.top_db))

        # Apply transform
        # torchaudio transforms usually handle batch
        spec = mel_transform(waveforms)
        log_spec = amplitude_to_db(spec)

        # Output shape: (N, 1, n_mels, T_spec) -> Squeeze channel if preferred?
        # CNN usually takes (N, C, H, W). Here we can treat n_mels as H, time as W. C=1.
        # Let's keep C=1: (N, 1, n_mels, T)

        return {"features.X": log_spec.numpy()}
