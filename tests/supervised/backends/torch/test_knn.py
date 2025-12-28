from __future__ import annotations

import numpy as np
import pytest

from modssc.supervised.backends.torch.knn import TorchKNNClassifier
from modssc.supervised.errors import SupervisedValidationError

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on optional torch install
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)


def _toy_data():
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    return X, y


def test_torch_knn_fit_predict_euclidean_uniform() -> None:
    X, y = _toy_data()
    clf = TorchKNNClassifier(k=1, metric="euclidean", weights="uniform")
    fit = clf.fit(X, y)

    assert fit.n_samples == int(X.shape[0])
    assert clf.supports_proba
    assert clf.classes_t_ is not None

    scores = clf.predict_scores(X)
    assert scores.shape == (int(X.shape[0]), int(clf.classes_t_.numel()))

    proba = clf.predict_proba(X)
    assert torch.allclose(scores, proba)

    pred = clf.predict(X)
    assert pred.shape[0] == int(X.shape[0])


def test_torch_knn_distance_weights_euclidean() -> None:
    X, y = _toy_data()
    clf = TorchKNNClassifier(k=2, metric="euclidean", weights="distance")
    clf.fit(X, y)

    scores = clf.predict_scores(X)
    row_sum = scores.sum(dim=1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum))


def test_torch_knn_distance_weights_cosine() -> None:
    X, y = _toy_data()
    clf = TorchKNNClassifier(k=2, metric="cosine", weights="distance")
    clf.fit(X, y)

    scores = clf.predict_scores(X)
    assert scores.shape[0] == int(X.shape[0])


def test_torch_knn_fit_accepts_1d_and_2d_labels() -> None:
    X = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([[0], [1]], dtype=torch.int64)

    clf = TorchKNNClassifier(k=1)
    fit = clf.fit(X, y)
    assert fit.n_samples == 2


def test_torch_knn_fit_requires_tensor_inputs() -> None:
    X, y = _toy_data()
    clf = TorchKNNClassifier()

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor X"):
        clf.fit(np.zeros((2, 2), dtype=np.float32), y)

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor y"):
        clf.fit(X, np.zeros((2,), dtype=np.int64))


def test_torch_knn_fit_invalid_shapes_and_empty() -> None:
    X, y = _toy_data()
    clf = TorchKNNClassifier()

    with pytest.raises(SupervisedValidationError, match="X must be 2D"):
        clf.fit(torch.zeros((1, 2, 3), dtype=torch.float32), y)

    with pytest.raises(SupervisedValidationError, match="matching first dimension"):
        clf.fit(X[:2], y[:3])

    with pytest.raises(SupervisedValidationError, match="non-empty"):
        clf.fit(torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64))


def test_torch_knn_fit_device_mismatch() -> None:
    X, y = _toy_data()
    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    y_mismatch = torch.empty((int(y.shape[0]),), dtype=torch.int64, device=mismatch_device)

    clf = TorchKNNClassifier()
    with pytest.raises(SupervisedValidationError, match="same device"):
        clf.fit(X, y_mismatch)


def test_torch_knn_fit_invalid_dtype_and_params() -> None:
    X, y = _toy_data()

    with pytest.raises(SupervisedValidationError, match="integer tensor"):
        TorchKNNClassifier().fit(X, y.to(torch.float32))

    with pytest.raises(SupervisedValidationError, match="k must be >= 1"):
        TorchKNNClassifier(k=0).fit(X, y)

    with pytest.raises(SupervisedValidationError, match="Unknown metric"):
        TorchKNNClassifier(metric="bad").fit(X, y)

    with pytest.raises(SupervisedValidationError, match="Unknown weights"):
        TorchKNNClassifier(weights="bad").fit(X, y)


def test_torch_knn_pairwise_scores_requires_fit() -> None:
    clf = TorchKNNClassifier()
    with pytest.raises(RuntimeError, match="not fitted"):
        clf._pairwise_scores(torch.zeros((1, 1), dtype=torch.float32))


def test_torch_knn_predict_scores_errors() -> None:
    X, y = _toy_data()
    clf = TorchKNNClassifier()

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_scores(X)

    clf.fit(X, y)

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor input"):
        clf.predict_scores(np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(SupervisedValidationError, match="X must be 2D"):
        clf.predict_scores(torch.zeros((1, 2, 3), dtype=torch.float32))

    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    X_mismatch = torch.empty((2, 2), dtype=torch.float32, device=mismatch_device)
    with pytest.raises(SupervisedValidationError, match="same device"):
        clf.predict_scores(X_mismatch)


def test_torch_knn_predict_scores_accepts_1d() -> None:
    X = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([0, 1, 0], dtype=torch.int64)
    clf = TorchKNNClassifier()
    clf.fit(X, y)

    scores = clf.predict_scores(torch.tensor([0.0, 1.0], dtype=torch.float32))
    assert scores.shape[0] == 2
