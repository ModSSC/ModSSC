from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.supervised.backends.torch.audio_cnn_scratch import TorchAudioCNNClassifier


def test_audio_cnn_scratch_fit_predict_numpy_cpu():
    X = np.random.randn(2, 32, 32).astype(np.float32)
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchAudioCNNClassifier(batch_size=1, max_epochs=1, n_jobs=0, seed=0)
    assert clf.supports_proba
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape[0] == 2
    pred = clf.predict(X)
    assert pred.shape[0] == 2


def test_audio_cnn_scratch_seed_none_branch():
    X = np.random.randn(2, 16, 16).astype(np.float32)
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchAudioCNNClassifier(batch_size=1, max_epochs=1, n_jobs=0, seed=None)
    clf.fit(X, y)


def test_audio_cnn_scratch_predict_tensor_branch(monkeypatch):
    X = np.random.randn(2, 16, 16).astype(np.float32)
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchAudioCNNClassifier(batch_size=1, max_epochs=1, n_jobs=0, seed=0)
    clf.fit(X, y)
    X_t = torch.randn(2, 1, 16, 16)
    proba = clf.predict_proba(X_t)
    assert proba.shape[0] == 2

    monkeypatch.setattr(clf, "predict_proba", lambda _x: torch.tensor([[0.2, 0.8]]))
    pred = clf.predict(X_t)
    assert pred.shape[0] == 1


def test_audio_cnn_scratch_cuda_branch(monkeypatch):
    X = torch.randn(2, 1, 4, 4)
    y = torch.tensor([0, 1], dtype=torch.int64)
    clf = TorchAudioCNNClassifier(batch_size=1, max_epochs=1, n_jobs=1, seed=0)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _boom(self, *_args, **_kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(torch.nn.Module, "to", _boom)

    with pytest.raises(RuntimeError, match="stop"):
        clf.fit(X, y)
