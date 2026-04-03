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
    assert isinstance(proba, torch.Tensor)
    assert proba.shape[0] == 2
    pred = clf.predict(X)
    assert isinstance(pred, torch.Tensor)
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
    assert isinstance(proba, torch.Tensor)
    assert proba.shape[0] == 2

    monkeypatch.setattr(clf, "_scores", lambda _x: torch.tensor([[0.2, 0.8]]))
    pred = clf.predict(X_t)
    assert pred.shape[0] == 1


def test_audio_cnn_scratch_cuda_branch(monkeypatch):
    X = np.random.randn(2, 4, 4).astype(np.float32)
    y = torch.tensor([0, 1], dtype=torch.int64)
    clf = TorchAudioCNNClassifier(batch_size=1, max_epochs=1, n_jobs=1, seed=0)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _boom(self, *_args, **_kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(torch.nn.Module, "to", _boom)

    with pytest.raises(RuntimeError, match="stop"):
        clf.fit(X, y)


def test_audio_cnn_scratch_validation_and_score_error_paths(monkeypatch):
    clf = TorchAudioCNNClassifier(batch_size=1, max_epochs=1, n_jobs=0, seed=0)

    with pytest.raises(Exception, match="3D or 4D"):
        clf._prepare_X(np.zeros((2, 4), dtype=np.float32), torch)
    with pytest.raises(Exception, match="non-empty"):
        clf._prepare_X(np.zeros((0, 1, 4, 4), dtype=np.float32), torch)

    y_flat = clf._prepare_y(np.asarray([[0], [1]], dtype=np.int64), torch)
    assert y_flat.shape == (2,)
    with pytest.raises(Exception, match="non-empty"):
        clf._prepare_y(np.asarray([], dtype=np.int64), torch)

    x_cpu = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    y_meta = torch.empty((1,), dtype=torch.int64, device="meta")
    with pytest.raises(Exception, match="same device"):
        clf._fit_device(x_cpu, y_meta, torch)

    cuda_clf = TorchAudioCNNClassifier(batch_size=1, max_epochs=1, n_jobs=1, seed=0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert (
        cuda_clf._fit_device(
            np.zeros((2, 4, 4), dtype=np.float32),
            np.asarray([0, 1], dtype=np.int64),
            torch,
        ).type
        == "cuda"
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(Exception, match="matching first dimension"):
        clf.fit(np.zeros((2, 4, 4), dtype=np.float32), np.asarray([0], dtype=np.int64))

    with pytest.raises(RuntimeError, match="not fitted"):
        clf._scores(np.zeros((2, 4, 4), dtype=np.float32))

    fitted = TorchAudioCNNClassifier(batch_size=1, max_epochs=1, n_jobs=0, seed=0)
    X = np.random.randn(2, 8, 8).astype(np.float32)
    y = np.asarray([0, 1], dtype=np.int64)
    fitted.fit(X, y)
    with pytest.raises(Exception, match="same device as the model"):
        fitted._scores(torch.empty((2, 1, 8, 8), device="meta"))
