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
    assert isinstance(proba, torch.Tensor)
    assert proba.shape[0] == 1
    pred = clf.predict(np.array([[1, 2, 3]], dtype=np.int64))
    assert isinstance(pred, torch.Tensor)
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
    assert isinstance(proba, torch.Tensor)
    assert proba.shape[0] == 1

    monkeypatch.setattr(clf, "_scores", lambda _x: torch.tensor([[0.1, 0.9]]))
    pred = clf.predict(X_t)
    assert pred.shape[0] == 1


def test_lstm_scratch_cuda_branch(monkeypatch):
    X = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64)
    y = torch.tensor([0, 1], dtype=torch.int64)
    clf = TorchLSTMClassifier(batch_size=1, max_epochs=1, n_jobs=1, seed=0, vocab_size=10)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _boom(self, *_args, **_kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(torch.nn.Module, "to", _boom)

    with pytest.raises(RuntimeError, match="stop"):
        clf.fit(X, y)


def test_lstm_scratch_validation_and_score_error_paths(monkeypatch):
    clf = TorchLSTMClassifier(batch_size=1, max_epochs=1, n_jobs=0, seed=0, vocab_size=10)

    with pytest.raises(Exception, match="2D token ids"):
        clf._prepare_X(np.asarray([1, 2, 3], dtype=np.int64), torch)
    with pytest.raises(Exception, match="non-empty"):
        clf._prepare_X(np.zeros((0, 3), dtype=np.int64), torch)

    y_flat = clf._prepare_y(np.asarray([[0], [1]], dtype=np.int64), torch)
    assert y_flat.shape == (2,)
    with pytest.raises(Exception, match="non-empty"):
        clf._prepare_y(np.asarray([], dtype=np.int64), torch)

    x_cpu = torch.zeros((1, 3), dtype=torch.int64)
    y_meta = torch.empty((1,), dtype=torch.int64, device="meta")
    with pytest.raises(Exception, match="same device"):
        clf._fit_device(x_cpu, y_meta, torch)

    cuda_clf = TorchLSTMClassifier(batch_size=1, max_epochs=1, n_jobs=1, seed=0, vocab_size=10)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert (
        cuda_clf._fit_device(
            np.asarray([[1, 2, 3], [2, 3, 4]], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            torch,
        ).type
        == "cuda"
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(Exception, match="matching first dimension"):
        clf.fit(np.asarray([[1, 2, 3], [2, 3, 4]], dtype=np.int64), np.asarray([0], dtype=np.int64))

    with pytest.raises(RuntimeError, match="not fitted"):
        clf._scores(np.asarray([[1, 2, 3]], dtype=np.int64))

    fitted = TorchLSTMClassifier(batch_size=1, max_epochs=1, n_jobs=0, seed=0, vocab_size=10)
    X = np.asarray([[1, 2, 3], [2, 3, 4]], dtype=np.int64)
    y = np.asarray([0, 1], dtype=np.int64)
    fitted.fit(X, y)
    with pytest.raises(Exception, match="same device as the model"):
        fitted._scores(torch.empty((2, 3), dtype=torch.int64, device="meta"))
