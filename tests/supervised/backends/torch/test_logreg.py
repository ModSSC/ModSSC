from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on optional torch install
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.supervised.backends.torch.logreg import TorchLogRegClassifier
from modssc.supervised.errors import SupervisedValidationError


def _toy_data():
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    return X, y


def test_torch_logreg_fit_predict() -> None:
    X, y = _toy_data()
    clf = TorchLogRegClassifier(max_epochs=1, batch_size=2, lr=1e-2)
    fit = clf.fit(X, y)

    assert fit.n_samples == int(X.shape[0])
    assert clf.supports_proba

    scores = clf.predict_scores(X)
    assert scores.shape == (int(X.shape[0]), int(torch.unique(y).numel()))

    proba = clf.predict_proba(X)
    assert torch.allclose(scores, proba)

    pred = clf.predict(X)
    assert pred.shape[0] == int(X.shape[0])


def test_torch_logreg_fit_accepts_1d_and_2d_labels() -> None:
    X = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([[0], [1]], dtype=torch.int64)

    clf = TorchLogRegClassifier(max_epochs=1, batch_size=1, lr=1e-2)
    fit = clf.fit(X, y)
    assert fit.n_samples == 2


def test_torch_logreg_fit_rejects_scalar_input() -> None:
    X = torch.tensor(1.0, dtype=torch.float32)
    y = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(SupervisedValidationError, match="X must be 2D"):
        TorchLogRegClassifier(max_epochs=1).fit(X, y)


def test_torch_logreg_fit_requires_tensor_inputs() -> None:
    X, y = _toy_data()
    clf = TorchLogRegClassifier(max_epochs=1)

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor X"):
        clf.fit(np.zeros((2, 2), dtype=np.float32), y)

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor y"):
        clf.fit(X, np.zeros((2,), dtype=np.int64))


def test_torch_logreg_fit_invalid_shapes_and_params() -> None:
    X, y = _toy_data()

    with pytest.raises(SupervisedValidationError, match="matching first dimension"):
        TorchLogRegClassifier(max_epochs=1).fit(X[:2], y[:3])

    with pytest.raises(SupervisedValidationError, match="non-empty"):
        TorchLogRegClassifier(max_epochs=1).fit(
            torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
        )

    with pytest.raises(SupervisedValidationError, match="batch_size must be >= 1"):
        TorchLogRegClassifier(batch_size=0).fit(X, y)

    with pytest.raises(SupervisedValidationError, match="max_epochs must be >= 1"):
        TorchLogRegClassifier(max_epochs=0).fit(X, y)


def test_torch_logreg_predict_requires_fit() -> None:
    X, _ = _toy_data()
    clf = TorchLogRegClassifier(max_epochs=1)

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_scores(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(X)


def test_torch_logreg_accepts_3d_inputs() -> None:
    X, y = _toy_data()
    X3 = X.reshape(int(X.shape[0]), 1, int(X.shape[1]))
    clf = TorchLogRegClassifier(max_epochs=1, batch_size=2, lr=1e-2)
    clf.fit(X3, y)

    scores = clf.predict_scores(X3)
    assert scores.shape[0] == int(X.shape[0])


def test_torch_logreg_scores_input_validation() -> None:
    X, y = _toy_data()
    clf = TorchLogRegClassifier(max_epochs=1, batch_size=2, lr=1e-2)
    clf.fit(X, y)

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor input"):
        clf.predict_scores(np.zeros((2, 2), dtype=np.float32))

    X1 = torch.tensor([0.0, 1.0], dtype=torch.float32)
    y1 = torch.tensor([0, 1], dtype=torch.int64)
    clf1 = TorchLogRegClassifier(max_epochs=1, batch_size=2, lr=1e-2)
    clf1.fit(X1, y1)
    scores = clf1.predict_scores(X1)
    assert scores.shape == (2, 2)

    with pytest.raises(SupervisedValidationError, match="X must be 2D"):
        clf.predict_scores(torch.tensor(1.0))

    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    X_mismatch = torch.empty((2, 2), dtype=torch.float32, device=mismatch_device)
    with pytest.raises(SupervisedValidationError, match="same device"):
        clf.predict_scores(X_mismatch)


def test_torch_logreg_device_mismatch() -> None:
    X, y = _toy_data()
    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    y_mismatch = torch.empty((int(y.shape[0]),), dtype=torch.int64, device=mismatch_device)

    with pytest.raises(SupervisedValidationError, match="same device"):
        TorchLogRegClassifier(max_epochs=1).fit(X, y_mismatch)
