from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on optional torch install
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.supervised.backends.torch.image_cnn import (
    TorchImageCNNClassifier,
    _make_activation,
    _parse_input_shape,
)
from modssc.supervised.errors import SupervisedValidationError


def _toy_data(n: int = 4, c: int = 1, h: int = 4, w: int = 4):
    X = torch.randn(n, c, h, w, dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1][:n], dtype=torch.int64)
    return X, y


def test_parse_input_shape() -> None:
    assert _parse_input_shape(None) is None
    assert _parse_input_shape((4, 5)) == (1, 4, 5)
    assert _parse_input_shape((3, 4, 5)) == (3, 4, 5)
    with pytest.raises(SupervisedValidationError, match="input_shape must be"):
        _parse_input_shape("bad")
    with pytest.raises(SupervisedValidationError, match="input_shape must be"):
        _parse_input_shape((1, 2, 3, 4))
    with pytest.raises(SupervisedValidationError, match="input_shape entries must be positive"):
        _parse_input_shape((0, 2))


def test_make_activation() -> None:
    assert isinstance(_make_activation("relu", torch), torch.nn.ReLU)
    assert isinstance(_make_activation("gelu", torch), torch.nn.GELU)
    assert isinstance(_make_activation("tanh", torch), torch.nn.Tanh)
    with pytest.raises(SupervisedValidationError, match="Unknown activation"):
        _make_activation("bad", torch)


def test_image_cnn_fit_predict_4d() -> None:
    X, y = _toy_data()
    clf = TorchImageCNNClassifier(
        conv_channels=(4,),
        fc_dim=8,
        dropout=0.2,
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


def test_image_cnn_fit_predict_3d() -> None:
    X, y = _toy_data()
    X3 = X.squeeze(1)
    clf = TorchImageCNNClassifier(
        conv_channels=(4,),
        fc_dim=8,
        dropout=0.0,
        max_epochs=1,
        batch_size=2,
    )
    clf.fit(X3, y)
    scores = clf.predict_scores(X3)
    assert scores.shape[0] == int(X.shape[0])


def test_image_cnn_fit_predict_2d_with_input_shape() -> None:
    X, y = _toy_data()
    X2 = X.reshape(int(X.shape[0]), -1)
    clf = TorchImageCNNClassifier(
        conv_channels=(4,),
        fc_dim=0,
        dropout=0.0,
        max_epochs=1,
        batch_size=2,
        input_shape=(1, int(X.shape[2]), int(X.shape[3])),
    )
    clf.fit(X2, y)
    scores = clf.predict_scores(X2)
    assert scores.shape[0] == int(X.shape[0])


def test_image_cnn_input_shape_mismatch() -> None:
    X, y = _toy_data()
    X2 = X.reshape(int(X.shape[0]), -1)
    clf = TorchImageCNNClassifier(
        conv_channels=(4,),
        fc_dim=0,
        dropout=0.0,
        max_epochs=1,
        batch_size=2,
        input_shape=(1, int(X.shape[2]), int(X.shape[3])),
    )
    clf.fit(X2, y)

    bad = torch.zeros((int(X.shape[0]), int(X2.shape[1]) + 1), dtype=torch.float32)
    with pytest.raises(SupervisedValidationError, match="input_shape does not match"):
        clf.predict_scores(bad)


def test_image_cnn_2d_requires_input_shape() -> None:
    X, y = _toy_data()
    X2 = X.reshape(int(X.shape[0]), -1)
    clf = TorchImageCNNClassifier(conv_channels=(4,), max_epochs=1, batch_size=2)
    with pytest.raises(SupervisedValidationError, match="requires 3D/4D inputs"):
        clf.fit(X2, y)

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor X"):
        clf._prepare_X(np.zeros((2, 2), dtype=np.float32), torch, allow_infer=True)

    with pytest.raises(SupervisedValidationError, match="X must be 2D, 3D, or 4D"):
        clf._prepare_X(torch.zeros((1, 1, 1, 1, 1)), torch, allow_infer=True)


def test_image_cnn_fit_invalid_params() -> None:
    X, y = _toy_data()
    with pytest.raises(SupervisedValidationError, match="conv_channels must be non-empty"):
        TorchImageCNNClassifier(conv_channels=(), max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="kernel_size must be >= 1"):
        TorchImageCNNClassifier(kernel_size=0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="fc_dim must be >= 0"):
        TorchImageCNNClassifier(fc_dim=-1, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="batch_size must be >= 1"):
        TorchImageCNNClassifier(batch_size=0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="max_epochs must be >= 1"):
        TorchImageCNNClassifier(max_epochs=0).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="lr must be > 0"):
        TorchImageCNNClassifier(lr=0.0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="dropout must be in"):
        TorchImageCNNClassifier(dropout=2.0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="conv_channels must be positive"):
        TorchImageCNNClassifier(conv_channels=(-1,), max_epochs=1).fit(X, y)


def test_image_cnn_fit_validation_errors() -> None:
    X, y = _toy_data()
    clf = TorchImageCNNClassifier(conv_channels=(4,), max_epochs=1, batch_size=2)

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor y"):
        clf.fit(X, np.zeros((2,), dtype=np.int64))

    with pytest.raises(SupervisedValidationError, match="matching first dimension"):
        clf.fit(X[:2], y[:3])

    with pytest.raises(SupervisedValidationError, match="X must be non-empty"):
        clf.fit(torch.zeros((0, 1, 2, 2)), torch.zeros((0,), dtype=torch.int64))

    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    y_mismatch = torch.empty((int(y.shape[0]),), dtype=torch.int64, device=mismatch_device)
    with pytest.raises(SupervisedValidationError, match="same device"):
        clf.fit(X, y_mismatch)

    y2 = y.view(-1, 1)
    clf.fit(X, y2)


def test_image_cnn_scores_validation() -> None:
    X, y = _toy_data()
    clf = TorchImageCNNClassifier(conv_channels=(4,), max_epochs=1, batch_size=2)

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_scores(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(X)

    clf.fit(X, y)

    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    X_mismatch = torch.empty((2, 1, 4, 4), dtype=torch.float32, device=mismatch_device)
    with pytest.raises(SupervisedValidationError, match="same device"):
        clf.predict_scores(X_mismatch)


def test_image_cnn_shape_mismatch_after_fit() -> None:
    X, y = _toy_data()
    clf = TorchImageCNNClassifier(conv_channels=(4,), max_epochs=1, batch_size=2)
    clf.fit(X, y)

    bad = torch.randn(int(X.shape[0]), 1, 5, 5, dtype=torch.float32)
    with pytest.raises(SupervisedValidationError, match="X shape does not match"):
        clf.predict_scores(bad)
