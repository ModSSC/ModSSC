from __future__ import annotations

import numpy as np

from modssc.preprocess.steps.audio import spectrogram
from modssc.preprocess.steps.audio.spectrogram import LogMelSpectrogramStep
from modssc.preprocess.store import ArtifactStore


class _FakeTensor:
    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr)
        self.shape = self._arr.shape
        self.ndim = self._arr.ndim

    def unsqueeze(self, dim: int):
        return _FakeTensor(np.expand_dims(self._arr, axis=dim))

    def numpy(self):
        return self._arr


class _FakeTorch:
    float32 = "float32"

    def from_numpy(self, arr):
        return _FakeTensor(arr)

    def as_tensor(self, arr, dtype=None):
        return _FakeTensor(arr)


class _FakeMel:
    def __init__(self, **_kwargs):
        self.kwargs = _kwargs

    def __call__(self, x):
        return x


class _FakeAmp:
    def __init__(self, **_kwargs):
        self.kwargs = _kwargs

    def __call__(self, x):
        return x


class _FakeTorchaudio:
    class transforms:  # noqa: D401
        MelSpectrogram = _FakeMel
        AmplitudeToDB = _FakeAmp


def _install_fakes(monkeypatch):
    fake_torch = _FakeTorch()
    fake_ta = _FakeTorchaudio()

    def _require(*, module: str, **_kwargs):
        return fake_torch if module == "torch" else fake_ta

    monkeypatch.setattr(spectrogram, "require", _require)
    return fake_torch, fake_ta


def test_log_mel_spectrogram_numpy_unsqueeze(monkeypatch):
    _install_fakes(monkeypatch)
    step = LogMelSpectrogramStep()
    store = ArtifactStore({"features.X": np.random.randn(2, 8).astype(np.float32)})
    out = step.transform(store, rng=np.random.default_rng(0))
    assert out["features.X"].shape == (2, 1, 8)


def test_log_mel_spectrogram_list_no_unsqueeze(monkeypatch):
    _install_fakes(monkeypatch)
    step = LogMelSpectrogramStep()
    waves = [np.random.randn(1, 6).astype(np.float32), np.random.randn(1, 6).astype(np.float32)]
    store = ArtifactStore({"features.X": waves})
    out = step.transform(store, rng=np.random.default_rng(0))
    assert out["features.X"].shape == (2, 1, 6)
