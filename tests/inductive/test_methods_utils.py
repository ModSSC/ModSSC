from __future__ import annotations

import numpy as np
import pytest
import torch

from modssc.inductive.errors import InductiveValidationError, OptionalDependencyError
from modssc.inductive.methods import utils
from modssc.inductive.types import DeviceSpec


class _NumpyScores:
    def predict_scores(self, X):
        return np.tile(np.array([[0.2, 0.8]], dtype=np.float32), (X.shape[0], 1))


class _NumpyProba:
    def predict_proba(self, X):
        return np.tile(np.array([[0.3, 0.7]], dtype=np.float32), (X.shape[0], 1))


class _NumpyPredictOnly:
    def __init__(self, classes=None):
        if classes is not None:
            self.classes_ = np.asarray(classes)

    def predict(self, X):
        return np.zeros((X.shape[0],), dtype=int)


class _NoPredict:
    pass


class _TorchScores:
    def predict_scores(self, X):
        return torch.ones((X.shape[0], 2), device=X.device, dtype=torch.float32)


class _TorchProbaNonTensor:
    def predict_proba(self, X):
        return np.ones((X.shape[0], 2), dtype=np.float32)


class _TorchScoresWrongShape:
    def predict_scores(self, X):
        return torch.ones((X.shape[0],), device=X.device)


class _TorchScoresWrongDevice:
    def predict_scores(self, X):
        return torch.ones((X.shape[0], 2), device="meta")


def test_detect_backend_and_classifier_backend():
    assert utils.detect_backend(np.zeros((2, 2))) == "numpy"
    assert utils.detect_backend(torch.zeros((2, 2))) == "torch"
    assert utils.detect_backend({"x": torch.zeros((2, 2))}) == "torch"
    with pytest.raises(InductiveValidationError):
        utils.detect_backend([[1, 2]])

    spec = utils.BaseClassifierSpec(classifier_backend="numpy")
    utils.ensure_classifier_backend(spec, backend="numpy")
    with pytest.raises(InductiveValidationError):
        utils.ensure_classifier_backend(spec, backend="torch")

    spec_t = utils.BaseClassifierSpec(classifier_backend="torch")
    utils.ensure_classifier_backend(spec_t, backend="torch")
    with pytest.raises(InductiveValidationError):
        utils.ensure_classifier_backend(spec_t, backend="numpy")

    with pytest.raises(InductiveValidationError):
        utils.ensure_classifier_backend(spec_t, backend="bad")


def test_is_torch_tensor_optional_missing(monkeypatch):
    def _boom():
        raise OptionalDependencyError("torch", "inductive-torch")

    monkeypatch.setattr("modssc.inductive.methods.utils._torch", _boom)
    assert utils.is_torch_tensor(torch.tensor([1])) is False


def test_ensure_1d_labels_numpy():
    utils.ensure_1d_labels(np.array([0, 1], dtype=np.int64))
    with pytest.raises(InductiveValidationError):
        utils.ensure_1d_labels(np.array([[0, 1]]))
    with pytest.raises(InductiveValidationError):
        utils.ensure_1d_labels(np.array([], dtype=np.int64))
    with pytest.raises(InductiveValidationError):
        utils.ensure_1d_labels(np.array([0.1, 0.2]))


def test_ensure_1d_labels_torch():
    utils.ensure_1d_labels_torch(torch.tensor([0, 1], dtype=torch.int64))
    with pytest.raises(InductiveValidationError):
        utils.ensure_1d_labels_torch(np.array([0, 1]))
    with pytest.raises(InductiveValidationError):
        utils.ensure_1d_labels_torch(torch.tensor([[0, 1]], dtype=torch.int64))
    with pytest.raises(InductiveValidationError):
        utils.ensure_1d_labels_torch(torch.tensor([], dtype=torch.int64))
    with pytest.raises(InductiveValidationError):
        utils.ensure_1d_labels_torch(torch.tensor([0.1, 0.2]))


def test_build_classifier_and_ensure_cpu_device():
    spec = utils.BaseClassifierSpec(classifier_id="knn", classifier_backend="numpy")
    clf = utils.build_classifier(spec, seed=0)
    assert hasattr(clf, "fit")
    utils.ensure_cpu_device(DeviceSpec(device="cpu"))
    with pytest.raises(InductiveValidationError):
        utils.ensure_cpu_device(DeviceSpec(device="cuda"))


def test_predict_scores_numpy_paths():
    X = np.zeros((3, 2), dtype=np.float32)
    assert utils._predict_scores_numpy(_NumpyScores(), X).shape == (3, 2)
    assert utils._predict_scores_numpy(_NumpyProba(), X).shape == (3, 2)
    assert utils._predict_scores_numpy(_NumpyPredictOnly(), X).shape == (3, 1)
    assert utils._predict_scores_numpy(_NumpyPredictOnly(classes=[0, 1]), X).shape == (3, 2)

    with pytest.raises(InductiveValidationError):
        utils._predict_scores_numpy(_NoPredict(), X)
    with pytest.raises(InductiveValidationError):
        utils._predict_scores_numpy(_NumpyScores(), [[1, 2]])

    class _BadScores:
        def predict_scores(self, X):
            return np.array([1, 2, 3])

    with pytest.raises(InductiveValidationError):
        utils._predict_scores_numpy(_BadScores(), X)


def test_predict_scores_torch_paths():
    X = torch.zeros((3, 2))
    assert utils._predict_scores_torch(_TorchScores(), X).shape == (3, 2)
    with pytest.raises(InductiveValidationError):
        utils._predict_scores_torch(_TorchProbaNonTensor(), X)
    with pytest.raises(InductiveValidationError):
        utils._predict_scores_torch(_TorchScoresWrongShape(), X)
    with pytest.raises(InductiveValidationError):
        utils._predict_scores_torch(_TorchScoresWrongDevice(), X)
    with pytest.raises(InductiveValidationError):
        utils._predict_scores_torch(_TorchScores(), np.zeros((2, 2)))

    class _NoScores:
        pass

    with pytest.raises(InductiveValidationError):
        utils._predict_scores_torch(_NoScores(), X)


def test_predict_scores_dispatch():
    X = np.zeros((2, 2), dtype=np.float32)
    assert utils.predict_scores(_NumpyScores(), X, backend="numpy").shape == (2, 2)
    Xt = torch.zeros((2, 2))
    assert utils.predict_scores(_TorchScores(), Xt, backend="torch").shape == (2, 2)
    with pytest.raises(InductiveValidationError):
        utils.predict_scores(_NumpyScores(), X, backend="bad")


def test_flatten_if_numpy_3d():
    X_3d = np.zeros((2, 3, 4))
    flat = utils.flatten_if_numpy(X_3d)
    assert flat.ndim == 2
    assert flat.shape == (2, 12)
    assert utils.flatten_if_numpy([[1, 2]]) == [[1, 2]]  # No-op for list


def test_select_confident_and_top_per_class():
    scores = np.array([[0.2, 0.8], [0.9, 0.1], [0.6, 0.4]], dtype=np.float32)
    idx = utils.select_confident(scores, threshold=0.7, max_new=1)
    assert idx.size == 1
    idx2 = utils.select_confident(scores, threshold=None, max_new=None)
    assert idx2.size == 3

    idx3 = utils.select_top_per_class(scores, k_per_class=1, threshold=0.7)
    assert idx3.ndim == 1
    idx4 = utils.select_top_per_class(scores, k_per_class=1, threshold=0.99)
    assert idx4.size == 0
    idx5 = utils.select_top_per_class(scores, k_per_class=1, threshold=None)
    assert idx5.size >= 1


def test_select_confident_torch_and_top_per_class():
    scores = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.6, 0.4]])
    idx = utils.select_confident_torch(scores, threshold=0.7, max_new=1)
    assert int(idx.numel()) == 1
    idx2 = utils.select_confident_torch(scores, threshold=None, max_new=None)
    assert int(idx2.numel()) == 3

    idx3 = utils.select_top_per_class_torch(scores, k_per_class=1, threshold=0.7)
    assert int(idx3.numel()) >= 1
    idx4 = utils.select_top_per_class_torch(scores, k_per_class=1, threshold=0.99)
    assert int(idx4.numel()) == 0
    idx5 = utils.select_top_per_class_torch(scores, k_per_class=1, threshold=None)
    assert int(idx5.numel()) >= 1
