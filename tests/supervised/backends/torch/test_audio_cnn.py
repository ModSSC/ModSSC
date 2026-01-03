from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on optional torch install
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.supervised.backends.torch.audio_cnn import (
    TorchAudioCNNClassifier,
    _make_activation,
    _parse_input_shape,
)
from modssc.supervised.errors import SupervisedValidationError


def _toy_data(n: int = 4, c: int = 1, length: int = 8):
    X = torch.randn(n, c, length, dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1][:n], dtype=torch.int64)
    return X, y


def test_parse_input_shape() -> None:
    assert _parse_input_shape(None) is None
    assert _parse_input_shape((8,)) == (1, 8)
    assert _parse_input_shape((2, 8)) == (2, 8)
    with pytest.raises(SupervisedValidationError, match="input_shape must be"):
        _parse_input_shape("bad")
    with pytest.raises(SupervisedValidationError, match="input_shape must be"):
        _parse_input_shape((1, 2, 3))
    with pytest.raises(SupervisedValidationError, match="input_shape entries must be positive"):
        _parse_input_shape((0, 2))


def test_make_activation() -> None:
    assert isinstance(_make_activation("relu", torch), torch.nn.ReLU)
    assert isinstance(_make_activation("gelu", torch), torch.nn.GELU)
    assert isinstance(_make_activation("tanh", torch), torch.nn.Tanh)
    with pytest.raises(SupervisedValidationError, match="Unknown activation"):
        _make_activation("bad", torch)


def test_audio_cnn_fit_predict_3d() -> None:
    X, y = _toy_data()
    clf = TorchAudioCNNClassifier(
        conv_channels=(4,),
        fc_dim=8,
        dropout=0.0,
        max_epochs=1,
        batch_size=2,
    )
    fit = clf.fit(X, y)
    assert fit.n_samples == int(X.shape[0])
    assert clf.supports_proba

    scores = clf.predict_scores(X)
    assert scores.shape == (int(X.shape[0]), int(torch.unique(y).numel()))

    proba = clf.predict_proba(X)
    assert torch.allclose(scores, proba)

    pred = clf.predict(X)
    assert pred.shape[0] == int(X.shape[0])


def test_audio_cnn_fit_predict_1d() -> None:
    X = torch.randn(8, dtype=torch.float32)
    y = torch.tensor([1], dtype=torch.int64)
    clf = TorchAudioCNNClassifier(conv_channels=(4,), fc_dim=0, max_epochs=1, batch_size=1)
    clf.fit(X, y)
    scores = clf.predict_scores(X)
    assert scores.shape[0] == 1


def test_audio_cnn_fit_predict_2d() -> None:
    X = torch.randn(4, 8, dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    clf = TorchAudioCNNClassifier(
        conv_channels=(4,),
        fc_dim=0,
        dropout=0.0,
        max_epochs=1,
        batch_size=2,
        input_shape=(1, 8),
    )
    clf.fit(X, y)
    scores = clf.predict_scores(X)
    assert scores.shape[0] == int(X.shape[0])


def test_audio_cnn_fit_predict_2d_multi_channel() -> None:
    X = torch.randn(4, 16, dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    clf = TorchAudioCNNClassifier(
        conv_channels=(4,),
        fc_dim=0,
        dropout=0.0,
        max_epochs=1,
        batch_size=2,
        input_shape=(2, 8),
    )
    clf.fit(X, y)
    scores = clf.predict_scores(X)
    assert scores.shape[0] == int(X.shape[0])


def test_audio_cnn_fit_predict_2d_infers_shape() -> None:
    X = torch.randn(4, 8, dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    clf = TorchAudioCNNClassifier(conv_channels=(4,), fc_dim=0, max_epochs=1, batch_size=2)
    clf.fit(X, y)
    scores = clf.predict_scores(X)
    assert scores.shape[0] == int(X.shape[0])


def test_audio_cnn_input_shape_mismatch() -> None:
    X, y = _toy_data()
    X2 = X.reshape(int(X.shape[0]), -1)
    clf = TorchAudioCNNClassifier(
        conv_channels=(4,),
        fc_dim=0,
        dropout=0.0,
        max_epochs=1,
        batch_size=2,
        input_shape=(2, int(X.shape[2])),
    )
    with pytest.raises(SupervisedValidationError, match="input_shape does not match"):
        clf.fit(X2, y)


def test_audio_cnn_fit_invalid_params() -> None:
    X, y = _toy_data()
    with pytest.raises(SupervisedValidationError, match="conv_channels must be non-empty"):
        TorchAudioCNNClassifier(conv_channels=(), max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="kernel_size must be >= 1"):
        TorchAudioCNNClassifier(kernel_size=0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="fc_dim must be >= 0"):
        TorchAudioCNNClassifier(fc_dim=-1, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="batch_size must be >= 1"):
        TorchAudioCNNClassifier(batch_size=0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="max_epochs must be >= 1"):
        TorchAudioCNNClassifier(max_epochs=0).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="lr must be > 0"):
        TorchAudioCNNClassifier(lr=0.0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="dropout must be in"):
        TorchAudioCNNClassifier(dropout=2.0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="conv_channels must be positive"):
        TorchAudioCNNClassifier(conv_channels=(-1,), max_epochs=1).fit(X, y)


def test_audio_cnn_prepare_errors_and_mismatch() -> None:
    X, y = _toy_data()
    clf = TorchAudioCNNClassifier(conv_channels=(4,), max_epochs=1, batch_size=2)
    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor X"):
        clf._prepare_X(np.zeros((2, 2), dtype=np.float32), torch, allow_infer=True)
    with pytest.raises(SupervisedValidationError, match="1D, 2D, or 3D"):
        clf._prepare_X(torch.zeros((1, 1, 1, 1)), torch, allow_infer=True)

    clf.fit(X, y)
    with pytest.raises(SupervisedValidationError, match="X shape does not match"):
        clf.predict_scores(torch.randn(2, 2, 4, dtype=torch.float32))


def test_audio_cnn_predict_requires_fit() -> None:
    X, _ = _toy_data()
    clf = TorchAudioCNNClassifier(conv_channels=(4,), max_epochs=1, batch_size=2)

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(X)


def test_audio_cnn_fit_validation_errors() -> None:
    X, y = _toy_data()
    clf = TorchAudioCNNClassifier(conv_channels=(4,), max_epochs=1, batch_size=2)

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor y"):
        clf.fit(X, np.zeros((2,), dtype=np.int64))

    with pytest.raises(SupervisedValidationError, match="matching first dimension"):
        clf.fit(X[:2], y[:3])

    with pytest.raises(SupervisedValidationError, match="X must be non-empty"):
        clf.fit(torch.zeros((0, 1, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64))

    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    y_mismatch = torch.empty((int(y.shape[0]),), dtype=torch.int64, device=mismatch_device)
    with pytest.raises(SupervisedValidationError, match="same device"):
        clf.fit(X, y_mismatch)

    y2 = y.view(-1, 1)
    clf.fit(X, y2)


def test_audio_cnn_scores_validation() -> None:
    X, y = _toy_data()
    clf = TorchAudioCNNClassifier(conv_channels=(4,), max_epochs=1, batch_size=2)

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_scores(X)

    clf.fit(X, y)
    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    X_mismatch = torch.empty((2, 1, 8), dtype=torch.float32, device=mismatch_device)
    with pytest.raises(SupervisedValidationError, match="same device"):
        clf.predict_scores(X_mismatch)
