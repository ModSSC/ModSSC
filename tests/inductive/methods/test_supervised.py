from types import SimpleNamespace

import numpy as np
import pytest

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.supervised import SupervisedMethod, SupervisedSpec
from modssc.inductive.types import DeviceSpec


def test_supervised_validation_error_empty_numpy():
    """Test validation error when X_l is empty (numpy)."""
    spec = SupervisedSpec(classifier_id="knn", classifier_backend="numpy")
    method = SupervisedMethod(spec=spec)
    # 0 samples, 2 features to satisfy 2D check
    X_l = np.zeros((0, 2))
    y_l = np.array([])
    X_u = np.array([[1, 2]])
    data = SimpleNamespace(X_l=X_l, y_l=y_l, X_u=X_u, X_u_w=None, X_u_s=None)

    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        method.fit(data, device=DeviceSpec("cpu"))


def test_supervised_validation_error_empty_torch():
    """Test validation error when X_l is empty (torch)."""
    pytest.importorskip("torch")
    import torch

    spec = SupervisedSpec(classifier_id="knn", classifier_backend="torch")
    method = SupervisedMethod(spec=spec)
    X_l = torch.zeros((0, 2))
    y_l = torch.tensor([])
    X_u = torch.tensor([[1, 2]])
    data = SimpleNamespace(X_l=X_l, y_l=y_l, X_u=X_u, X_u_w=None, X_u_s=None)

    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        method.fit(data, device=DeviceSpec("cpu"))


def test_supervised_predict_proba_not_fitted():
    """Test runtime error when predict_proba is called before fit."""
    spec = SupervisedSpec(classifier_id="knn", classifier_backend="numpy")
    method = SupervisedMethod(spec=spec)
    with pytest.raises(RuntimeError, match="SupervisedMethod is not fitted yet"):
        method.predict_proba(np.array([[1, 2]]))


def test_supervised_predict_backend_mismatch():
    """Test validation error when predict input backend differs from fit backend."""
    pytest.importorskip("torch")
    import torch

    spec = SupervisedSpec(classifier_id="knn", classifier_backend="numpy")
    method = SupervisedMethod(spec=spec)
    X_l = np.array([[1, 2], [3, 4]])
    y_l = np.array([0, 1])
    data = SimpleNamespace(X_l=X_l, y_l=y_l, X_u=None, X_u_w=None, X_u_s=None)
    method.fit(data, device=DeviceSpec("cpu"))

    # Passing torch tensor to predict after fitting with numpy
    with pytest.raises(InductiveValidationError, match="predict_proba input backend mismatch"):
        method.predict_proba(torch.tensor([[1, 2]]))


def test_supervised_predict_proba_torch_row_sum():
    """Test predict_proba normalization with torch backend."""
    pytest.importorskip("torch")
    import torch

    class MockClf:
        def predict_proba(self, X):
            return torch.zeros((X.shape[0], 2))

        def fit(self, X, y):
            pass

        def __call__(self, X):
            return self.predict_proba(X)

    spec = SupervisedSpec(classifier_id="mock", classifier_backend="torch")
    method = SupervisedMethod(spec=spec)
    method._clf = MockClf()
    method._backend = "torch"

    X = torch.tensor([[1.0, 2.0]])
    probs = method.predict_proba(X)
    assert torch.all(probs == 0)


from ._supervised_method import *  # noqa: E402,F401,F403
