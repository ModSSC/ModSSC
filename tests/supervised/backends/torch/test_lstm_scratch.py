from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.supervised.backends.torch.lstm_scratch import TorchLSTMClassifier


class _SparseLike:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def to_dense(self):
        return self._arr


def test_lstm_scratch_fit_predict_numpy_cpu():
    X = _SparseLike(np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64))
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchLSTMClassifier(batch_size=1, max_epochs=1, n_jobs=0, seed=0, vocab_size=10)
    assert clf.supports_proba
    clf.fit(X, y)
    proba = clf.predict_proba(np.array([[1, 2, 3]], dtype=np.int64))
    assert proba.shape[0] == 1
    pred = clf.predict(np.array([[1, 2, 3]], dtype=np.int64))
    assert pred.shape[0] == 1


def test_lstm_scratch_seed_none_branch():
    X = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64)
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchLSTMClassifier(batch_size=1, max_epochs=1, n_jobs=0, seed=None, vocab_size=10)
    clf.fit(X, y)


def test_lstm_scratch_predict_tensor_branch(monkeypatch):
    X = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64)
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchLSTMClassifier(batch_size=1, max_epochs=1, n_jobs=0, seed=0, vocab_size=10)
    clf.fit(X, y)
    X_t = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    proba = clf.predict_proba(X_t)
    assert proba.shape[0] == 1

    monkeypatch.setattr(clf, "predict_proba", lambda _x: torch.tensor([[0.1, 0.9]]))
    pred = clf.predict(X_t)
    assert pred.shape[0] == 1


def test_lstm_scratch_cuda_branch(monkeypatch):
    X = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.int64)
    y = torch.tensor([0, 1], dtype=torch.int64)
    clf = TorchLSTMClassifier(batch_size=1, max_epochs=1, n_jobs=1, seed=0, vocab_size=10)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _boom(self, *_args, **_kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(torch.nn.Module, "to", _boom)

    with pytest.raises(RuntimeError, match="stop"):
        clf.fit(X, y)
