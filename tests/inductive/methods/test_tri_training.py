from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.tri_training import TriTrainingMethod, TriTrainingSpec
from modssc.inductive.types import DeviceSpec


def test_tri_training_torch_backend():
    X_l = torch.zeros(10, 5)
    y_l = torch.zeros(10, dtype=torch.long)
    X_u = torch.zeros(10, 5)

    data = SimpleNamespace(X_l=X_l, y_l=y_l, X_u=X_u, X_u_w=None, X_u_s=None, views=None, meta=None)

    spec = TriTrainingSpec(classifier_backend="torch", max_iter=1, bootstrap_ratio=1.0)

    model = TriTrainingMethod(spec)

    mock_clf = MagicMock()
    mock_clf.fit.return_value = None
    mock_clf.predict_proba.return_value = torch.tensor([[0.5, 0.5]] * 10)
    mock_clf.predict.return_value = torch.zeros(10)
    mock_clf.classes_ = np.array([0, 1])
    del mock_clf.predict_scores

    with patch("modssc.inductive.methods.tri_training.build_classifier", return_value=mock_clf):
        model.fit(data, device=DeviceSpec(device="cpu"))

        probs = model.predict_proba(X_l)
        assert torch.is_tensor(probs)


class MutableClassesClassifier:
    def __init__(self, responses, scores):
        self._responses = responses
        self._counter = 0
        self.scores = scores

    @property
    def classes_(self):
        val = self._responses[self._counter]
        if self._counter < len(self._responses) - 1:
            self._counter += 1
        return np.array(val)

    def predict_proba(self, X):
        return self.scores


def test_tri_training_alignment_branch_coverage_numpy():
    """Test for lines 310 (numpy) branch coverage where class is not in global map."""
    model = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    model._backend = "numpy"

    # Clf 1: stable classes [0, 1]
    clf1 = MagicMock()
    clf1.classes_ = np.array([0, 1])
    clf1.predict_proba.return_value = np.zeros((10, 2))

    # Clf 2: Unstable classes.
    # Call 1 (collection): [0, 1, 2] -> global map has {0, 1, 2}
    # Call 2 (alignment): [0, 1, 999] -> 999 not in map -> hits "else" (implicit) branch
    clf2 = MutableClassesClassifier(responses=[[0, 1, 2], [0, 1, 999]], scores=np.zeros((10, 3)))

    model._clfs = [clf1, clf2]

    with patch(
        "modssc.inductive.methods.tri_training.predict_scores",
        side_effect=lambda clf, X, backend: clf.predict_proba(X),
    ):
        X = np.zeros((10, 5))
        # trigger alignment
        _ = model.predict_proba(X)


def test_tri_training_alignment_branch_coverage_torch():
    """Test for line 323 (torch) branch coverage where class is not in global map."""
    model = TriTrainingMethod(TriTrainingSpec(classifier_backend="torch"))
    model._backend = "torch"

    # Clf 1: stable classes [0, 1]
    clf1 = MagicMock()
    clf1.classes_ = torch.tensor([0, 1])
    clf1.predict_proba.return_value = torch.zeros((10, 2))

    # Clf 2: Unstable classes.
    clf2 = MutableClassesClassifier(responses=[[0, 1, 2], [0, 1, 999]], scores=torch.zeros((10, 3)))

    model._clfs = [clf1, clf2]

    with patch(
        "modssc.inductive.methods.tri_training.predict_scores",
        side_effect=lambda clf, X, backend: clf.predict_proba(X),
    ):
        X = torch.zeros(10, 5)
        # trigger alignment
        _ = model.predict_proba(X)


def test_tri_training_valid_alignment():
    """Test that TriTraining correctly aligns scores when classifiers have different class counts."""
    model = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    model._backend = "numpy"

    clf1 = MagicMock()
    clf1.classes_ = np.array([0, 1])
    # Returns [0.8, 0.2] for class 0 and 1
    clf1.predict_proba.return_value = np.array([[0.8, 0.2]])

    clf2 = MagicMock()
    clf2.classes_ = np.array([0, 1, 2])
    # Returns [0.1, 0.1, 0.8] for class 0, 1, 2
    clf2.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])

    model._clfs = [clf1, clf2]

    with patch(
        "modssc.inductive.methods.tri_training.predict_scores",
        side_effect=lambda clf, X, backend: clf.predict_proba(X),
    ):
        probs = model.predict_proba(np.array([[0]]))

    assert probs.shape == (1, 3)
    # Expected alignment:
    # Clf1: [0.8, 0.2] -> [0.8, 0.2, 0.0]
    # Clf2: [0.1, 0.1, 0.8] -> [0.1, 0.1, 0.8]
    # Avg: [0.45, 0.15, 0.4]
    np.testing.assert_allclose(probs, [[0.45, 0.15, 0.4]])


def test_tri_training_validation_error_shape_mismatch():
    """Test validation error when one classifier returned shape doesn't match its classes."""
    model = TriTrainingMethod(TriTrainingSpec(classifier_backend="numpy"))
    model._backend = "numpy"

    clf1 = MagicMock()
    clf1.classes_ = np.array([0, 1])
    # Incorrect shape: 3 columns for 2 classes
    clf1.predict_proba.return_value = np.zeros((10, 3))

    clf2 = MagicMock()
    clf2.classes_ = np.array([0, 1, 2])
    clf2.predict_proba.return_value = np.zeros((10, 3))

    model._clfs = [clf1, clf2]

    with (
        patch(
            "modssc.inductive.methods.tri_training.predict_scores",
            side_effect=lambda clf, X, backend: clf.predict_proba(X),
        ),
        pytest.raises(
            InductiveValidationError, match="TriTraining classifiers disagree on class counts"
        ),
    ):
        model.predict_proba(np.zeros((10, 5)))


from ._tri_training_coverage import *  # noqa: E402,F401,F403
