from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import modssc.inductive.methods.helpers.classifier_bridge as classifier_bridge
import modssc.inductive.methods.utils as legacy_utils
from modssc.inductive.errors import InductiveValidationError


def test_legacy_utils_module_aliases_classifier_bridge() -> None:
    assert legacy_utils is classifier_bridge
    assert legacy_utils.detect_backend is classifier_bridge.detect_backend
    assert legacy_utils.predict_scores is classifier_bridge.predict_scores


class _RecordingPredictor:
    batch_size = 2

    def __init__(self, *, backend: str, scores: bool = False) -> None:
        self.backend = backend
        self.scores = scores
        self.batch_sizes: list[int] = []

    def _labels(self, X):
        labels = X[:, 0] % 3
        return labels.to(dtype=torch.int64) if self.backend == "torch" else labels

    def predict(self, X):
        self.batch_sizes.append(int(X.shape[0]))
        return self._labels(X)

    def predict_scores(self, X):
        self.batch_sizes.append(int(X.shape[0]))
        labels = self._labels(X)
        if self.backend == "torch":
            return torch.nn.functional.one_hot(labels, num_classes=3).to(dtype=torch.float32)
        return np.eye(3, dtype=np.float32)[labels]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("scores", [False, True])
def test_batched_prediction_matches_full_prediction(backend, scores):
    X = torch.arange(15).reshape(5, 3) if backend == "torch" else np.arange(15).reshape(5, 3)
    clf = _RecordingPredictor(backend=backend, scores=scores)

    actual = (
        classifier_bridge.predict_scores_in_batches(clf, X, backend=backend)
        if scores
        else classifier_bridge.predict_in_batches(clf, X, backend=backend)
    )
    labels = (X[:, 0] % 3).to(dtype=torch.int64) if backend == "torch" else X[:, 0] % 3
    expected = (
        torch.nn.functional.one_hot(labels, num_classes=3).to(dtype=torch.float32)
        if backend == "torch" and scores
        else np.eye(3, dtype=np.float32)[labels]
        if scores
        else labels
    )

    if backend == "torch":
        assert torch.equal(actual, expected)
    else:
        assert np.array_equal(actual, expected)
    assert clf.batch_sizes == [2, 2, 1]
    assert max(clf.batch_sizes) <= clf.batch_size


def test_batched_prediction_enforces_hard_memory_bound():
    class _Predictor:
        batch_size = classifier_bridge.MAX_PREDICTION_BATCH_SIZE + 100

        def __init__(self):
            self.batch_sizes = []

        def predict(self, X):
            self.batch_sizes.append(int(X.shape[0]))
            return np.zeros((int(X.shape[0]),), dtype=np.int64)

    X = np.zeros((classifier_bridge.MAX_PREDICTION_BATCH_SIZE + 1, 1), dtype=np.float32)
    clf = _Predictor()

    pred = classifier_bridge.predict_in_batches(clf, X, backend="numpy")

    assert pred.shape == (X.shape[0],)
    assert clf.batch_sizes == [classifier_bridge.MAX_PREDICTION_BATCH_SIZE, 1]
    assert max(clf.batch_sizes) <= classifier_bridge.MAX_PREDICTION_BATCH_SIZE


@pytest.mark.parametrize("configured", [True, "invalid", 0])
def test_prediction_batch_size_falls_back_for_invalid_values(configured):
    clf = SimpleNamespace(batch_size=configured)
    assert (
        classifier_bridge.prediction_batch_size(clf) == classifier_bridge.MAX_PREDICTION_BATCH_SIZE
    )


def test_batched_prediction_preserves_graph_and_empty_semantics():
    class _Predictor:
        batch_size = 1

        def __init__(self):
            self.inputs = []

        def predict(self, X):
            self.inputs.append(X)
            n_samples = int(X["x"].shape[0]) if isinstance(X, dict) else int(X.shape[0])
            return torch.zeros((n_samples,), dtype=torch.int64)

    graph = {
        "x": torch.zeros((3, 2)),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.int64),
    }
    graph_clf = _Predictor()
    graph_pred = classifier_bridge.predict_in_batches(graph_clf, graph, backend="torch")
    assert len(graph_clf.inputs) == 1
    assert graph_clf.inputs[0] is graph
    assert graph_pred.shape == (3,)

    empty = np.zeros((0, 2), dtype=np.float32)
    empty_clf = _Predictor()
    empty_pred = classifier_bridge.predict_in_batches(empty_clf, empty, backend="numpy")
    assert len(empty_clf.inputs) == 1
    assert empty_clf.inputs[0] is empty
    assert empty_pred.shape == (0,)


def test_batched_prediction_rejects_invalid_contracts():
    clf = SimpleNamespace(predict=lambda X: np.zeros((int(X.shape[0]) + 1,), dtype=np.int64))
    with pytest.raises(InductiveValidationError, match="preserve the inference batch size"):
        classifier_bridge.predict_in_batches(clf, np.zeros((2, 1)), backend="numpy")
    with pytest.raises(InductiveValidationError, match="Unknown backend"):
        classifier_bridge.predict_in_batches(clf, np.zeros((2, 1)), backend="invalid")


def test_batched_prediction_rejects_inputs_incompatible_with_backend():
    clf = SimpleNamespace(predict=lambda X: X)

    with pytest.raises(InductiveValidationError, match="Numpy backend requires numpy.ndarray"):
        classifier_bridge.predict_in_batches(clf, [[1.0]], backend="numpy")

    with pytest.raises(InductiveValidationError, match="Torch backend requires torch.Tensor"):
        classifier_bridge.predict_in_batches(clf, [torch.tensor([1.0])], backend="torch")


def test_batched_prediction_slices_torch_feature_dictionaries():
    class _Predictor:
        batch_size = 2

        def __init__(self):
            self.batch_sizes = []

        def predict(self, X):
            self.batch_sizes.append(int(X["x"].shape[0]))
            return X["x"][:, 0].to(dtype=torch.int64)

    X = {
        "x": torch.arange(10, dtype=torch.float32).reshape(5, 2),
        "sample_weight": torch.arange(5, dtype=torch.float32),
    }
    clf = _Predictor()

    pred = classifier_bridge.predict_in_batches(clf, X, backend="torch")

    assert torch.equal(pred, torch.tensor([0, 2, 4, 6, 8]))
    assert clf.batch_sizes == [2, 2, 1]


def test_batched_prediction_rejects_non_tensor_torch_predictions():
    clf = SimpleNamespace(
        batch_size=2,
        predict=lambda X: np.zeros((int(X.shape[0]),), dtype=np.int64),
    )

    with pytest.raises(InductiveValidationError, match="must return torch.Tensor"):
        classifier_bridge.predict_in_batches(clf, torch.zeros((3, 1)), backend="torch")
