from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on optional torch install
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.supervised.backends.torch.text_cnn import TorchTextCNNClassifier, _make_activation
from modssc.supervised.errors import SupervisedValidationError


def _toy_data(n: int = 4, length: int = 6, dim: int = 3):
    X = torch.randn(n, length, dim, dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1][:n], dtype=torch.int64)
    return X, y


def test_text_cnn_fit_predict_channels_last() -> None:
    X, y = _toy_data()
    clf = TorchTextCNNClassifier(
        kernel_sizes=(2, 3),
        num_filters=4,
        dropout=0.0,
        max_epochs=1,
        batch_size=2,
        input_layout="channels_last",
    )
    fit = clf.fit(X, y)
    assert fit.n_samples == int(X.shape[0])

    scores = clf.predict_scores(X)
    assert scores.shape == (int(X.shape[0]), int(torch.unique(y).numel()))


def test_text_cnn_fit_predict_2d_and_proba() -> None:
    X = torch.randn(4, 6, dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    clf = TorchTextCNNClassifier(kernel_sizes=(2,), num_filters=4, max_epochs=1, batch_size=2)
    assert clf.supports_proba

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(X)

    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape[0] == int(X.shape[0])

    pred = clf.predict(X)
    assert pred.shape[0] == int(X.shape[0])


def test_text_cnn_fit_predict_channels_first() -> None:
    X, y = _toy_data()
    X_first = X.transpose(1, 2)
    clf = TorchTextCNNClassifier(
        kernel_sizes=(2,),
        num_filters=4,
        dropout=0.0,
        max_epochs=1,
        batch_size=2,
        input_layout="channels_first",
    )
    clf.fit(X_first, y)
    scores = clf.predict_scores(X_first)
    assert scores.shape[0] == int(X.shape[0])


def test_text_cnn_invalid_input_layout() -> None:
    X, y = _toy_data()
    with pytest.raises(SupervisedValidationError, match="input_layout must be"):
        TorchTextCNNClassifier(input_layout="bad", max_epochs=1).fit(X, y)


def test_text_cnn_prepare_errors() -> None:
    clf = TorchTextCNNClassifier(max_epochs=1)
    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor X"):
        clf._prepare_X(np.zeros((2, 2), dtype=np.float32), torch)
    with pytest.raises(SupervisedValidationError, match="input_layout must be"):
        TorchTextCNNClassifier(input_layout="bad")._prepare_X(torch.randn(2, 3, 4), torch)
    with pytest.raises(SupervisedValidationError, match="2D or 3D"):
        clf._prepare_X(torch.randn(1, 1, 1, 1), torch)


def test_text_cnn_make_activation() -> None:
    assert isinstance(_make_activation("relu", torch), torch.nn.ReLU)
    assert isinstance(_make_activation("gelu", torch), torch.nn.GELU)
    assert isinstance(_make_activation("tanh", torch), torch.nn.Tanh)
    with pytest.raises(SupervisedValidationError, match="Unknown activation"):
        _make_activation("bad", torch)


def test_text_cnn_kernel_sizes_too_large() -> None:
    X = torch.randn(2, 2, 3, dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.int64)
    with pytest.raises(SupervisedValidationError, match="All kernel_sizes are larger"):
        TorchTextCNNClassifier(kernel_sizes=(3,), max_epochs=1).fit(X, y)


def test_text_cnn_invalid_params() -> None:
    X, y = _toy_data()
    with pytest.raises(SupervisedValidationError, match="kernel_sizes must be non-empty"):
        TorchTextCNNClassifier(kernel_sizes=(), max_epochs=1).fit(X, y)

    with pytest.raises(SupervisedValidationError, match="num_filters must be >= 1"):
        TorchTextCNNClassifier(num_filters=0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="batch_size must be >= 1"):
        TorchTextCNNClassifier(batch_size=0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="max_epochs must be >= 1"):
        TorchTextCNNClassifier(max_epochs=0).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="lr must be > 0"):
        TorchTextCNNClassifier(lr=0.0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="dropout must be in"):
        TorchTextCNNClassifier(dropout=2.0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="kernel_sizes must be positive"):
        TorchTextCNNClassifier(kernel_sizes=(-1,), max_epochs=1).fit(X, y)


def test_text_cnn_predict_sequence_too_short() -> None:
    X, y = _toy_data(length=5)
    clf = TorchTextCNNClassifier(kernel_sizes=(2, 3), num_filters=4, max_epochs=1, dropout=0.0)
    clf.fit(X, y)

    short = torch.randn(int(X.shape[0]), 2, int(X.shape[2]), dtype=torch.float32)
    with pytest.raises(SupervisedValidationError, match="Sequence length too short"):
        clf.predict_scores(short)


def test_text_cnn_predict_requires_fit() -> None:
    X, _ = _toy_data()
    clf = TorchTextCNNClassifier(max_epochs=1)

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_scores(X)


def test_text_cnn_fit_validation_errors() -> None:
    X, y = _toy_data()
    clf = TorchTextCNNClassifier(kernel_sizes=(2,), num_filters=4, max_epochs=1, batch_size=2)

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor y"):
        clf.fit(X, np.zeros((2,), dtype=np.int64))

    with pytest.raises(SupervisedValidationError, match="matching first dimension"):
        clf.fit(X[:2], y[:3])

    with pytest.raises(SupervisedValidationError, match="X must be non-empty"):
        clf.fit(torch.zeros((0, 2, 2)), torch.zeros((0,), dtype=torch.int64))

    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    y_mismatch = torch.empty((int(y.shape[0]),), dtype=torch.int64, device=mismatch_device)
    with pytest.raises(SupervisedValidationError, match="same device"):
        clf.fit(X, y_mismatch)

    y2 = y.view(-1, 1)
    clf.fit(X, y2)


def test_text_cnn_scores_validation() -> None:
    X, y = _toy_data()
    clf = TorchTextCNNClassifier(kernel_sizes=(2,), num_filters=4, max_epochs=1, batch_size=2)
    clf.fit(X, y)

    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    X_mismatch = torch.empty((2, 2, 2), dtype=torch.float32, device=mismatch_device)
    with pytest.raises(SupervisedValidationError, match="same device"):
        clf.predict_scores(X_mismatch)
