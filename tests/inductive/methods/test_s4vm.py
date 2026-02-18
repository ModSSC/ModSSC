from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from modssc.inductive.methods.s4vm import S4VMMethod, S4VMSpec
from modssc.inductive.types import DeviceSpec


def test_s4vm_torch_backend():
    X_l = torch.zeros(10, 5)
    y_l = torch.zeros(10, dtype=torch.long)
    y_l[:5] = 1  # Ensure binary classification (2 classes)
    X_u = torch.zeros(10, 5)

    data = SimpleNamespace(X_l=X_l, y_l=y_l, X_u=X_u, X_u_w=None, X_u_s=None, views=None, meta=None)

    spec = S4VMSpec(classifier_backend="torch", k_candidates=1)

    model = S4VMMethod(spec)

    mock_clf = MagicMock()
    mock_clf.fit.return_value = None
    mock_clf.predict.return_value = torch.zeros(10)  # For both X_l and X_u
    mock_clf.predict_proba.return_value = torch.tensor([[0.5, 0.5]] * 10)
    # Ensure predict_scores falls back to predict_proba
    del mock_clf.predict_scores

    with patch("modssc.inductive.methods.s4vm.build_classifier", return_value=mock_clf):
        model.fit(data, device=DeviceSpec(device="cpu"))

        # Verify predict_proba runs torch path
        probs = model.predict_proba(X_l)
        assert torch.is_tensor(probs)
        assert probs.shape == (10, 2)


def test_s4vm_predict_proba_zero_sum_numpy():
    """Test S4VM predict_proba handling of zero-sum scores in numpy backend."""
    model = S4VMMethod(S4VMSpec(classifier_backend="numpy"))
    model._backend = "numpy"

    mock_clf = MagicMock()
    model._clf = mock_clf

    with patch("modssc.inductive.methods.s4vm.predict_scores") as mock_predict_scores:
        # 2 samples, 2 classes. First sample [0, 0], second [0.5, 0.5]
        mock_predict_scores.return_value = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32)

        X = np.zeros((2, 5))
        probs = model.predict_proba(X)

        # Row 0 sum is 0, should become 1.0 (to avoid div by zero)
        # 0/1 -> 0.
        assert np.all(probs[0] == 0.0)
        assert np.allclose(probs[1], [0.5, 0.5])


def test_s4vm_predict_proba_numpy_normal():
    """Test S4VM predict_proba with Numpy backend and normal probabilities (sum > 0)."""
    model = S4VMMethod(S4VMSpec(classifier_backend="numpy"))
    model._backend = "numpy"

    mock_clf = MagicMock()
    model._clf = mock_clf

    with patch("modssc.inductive.methods.s4vm.predict_scores") as mock_predict_scores:
        # Normal probabilities
        mock_predict_scores.return_value = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)

        X = np.zeros((2, 5))
        probs = model.predict_proba(X)

        assert isinstance(probs, np.ndarray)
        assert np.allclose(probs.sum(axis=1), 1.0)


def test_s4vm_predict_coverage():
    """Test S4VM predict (wrapper around clf.predict)."""
    model = S4VMMethod(S4VMSpec(classifier_backend="numpy"))
    model._backend = "numpy"
    model._clf = MagicMock()
    model._clf.predict.return_value = np.array([0, 1])

    X = np.zeros((2, 5))
    preds = model.predict(X)
    assert np.array_equal(preds, np.array([0, 1]))


def test_s4vm_predict_proba_backend_mismatch():
    """Test S4VM predict_proba raises InductiveValidationError on backend mismatch."""
    from modssc.inductive.errors import InductiveValidationError

    model = S4VMMethod(S4VMSpec(classifier_backend="numpy"))
    model._backend = "numpy"
    # Mock _clf so checking it is fitted passes
    model._clf = MagicMock()

    # If detect_backend returns 'torch' while model._backend is 'numpy', it should raise.
    with (
        patch("modssc.inductive.methods.s4vm.detect_backend", return_value="torch"),
        pytest.raises(InductiveValidationError, match="predict_proba input backend mismatch"),
    ):
        model.predict_proba(np.zeros((2, 2)))
