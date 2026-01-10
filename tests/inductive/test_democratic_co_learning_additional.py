from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

import modssc.inductive.methods.democratic_co_learning as dcl
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.democratic_co_learning import (
    DemocraticCoLearningMethod,
    DemocraticCoLearningSpec,
)
from modssc.inductive.types import DeviceSpec

from .conftest import DummyDataset


class _FixedPredictor:
    def __init__(self, y_l, y_u, *, classes=None):
        self._y_l = np.asarray(y_l)
        self._y_u = np.asarray(y_u)
        if classes is not None:
            self.classes_ = np.asarray(classes)

    def fit(self, _x, _y):
        return self

    def predict(self, x):
        n = int(x.shape[0])
        if n == int(self._y_l.shape[0]):
            return self._y_l.copy()
        return self._y_u[:n].copy()


class _TorchPredictor:
    def __init__(self, y_l, y_u, *, classes_t=None):
        self._y_l = y_l
        self._y_u = y_u
        if classes_t is not None:
            self.classes_t_ = classes_t

    def fit(self, _x, _y):
        return self

    def predict(self, x):
        n = int(x.shape[0])
        if n == int(self._y_l.shape[0]):
            return self._y_l.clone()
        return self._y_u[:n].clone()


class _FixedPredictorWithLabeled:
    def __init__(self, pred_l, pred_u, *, classes=None):
        self._pred_l = np.asarray(pred_l)
        self._pred_u = np.asarray(pred_u)
        if classes is not None:
            self.classes_ = np.asarray(classes)

    def fit(self, _x, _y):
        return self

    def predict(self, x):
        n = int(x.shape[0])
        if n == int(self._pred_l.shape[0]):
            return self._pred_l.copy()
        return self._pred_u[:n].copy()


class _TorchPredictorWithLabeled:
    def __init__(self, pred_l, pred_u, *, classes_t=None):
        self._pred_l = pred_l
        self._pred_u = pred_u
        if classes_t is not None:
            self.classes_t_ = classes_t

    def fit(self, _x, _y):
        return self

    def predict(self, x):
        n = int(x.shape[0])
        if n == int(self._pred_l.shape[0]):
            return self._pred_l.clone()
        return self._pred_u[:n].clone()


def test_democratic_helper_validation_errors():
    with pytest.raises(InductiveValidationError, match="confidence_level"):
        dcl._z_value(1.0)
    with pytest.raises(InductiveValidationError, match="total=0"):
        dcl._accuracy_confidence_interval(0, 0, confidence_level=0.95)


def test_democratic_classifier_spec_resolution_errors():
    spec = DemocraticCoLearningSpec(classifier_specs=(dcl.BaseClassifierSpec(),) * 2)
    with pytest.raises(InductiveValidationError, match="at least 3 learners"):
        dcl._resolve_classifier_specs(spec)
    spec2 = DemocraticCoLearningSpec(n_learners=2)
    with pytest.raises(InductiveValidationError, match="n_learners must be"):
        dcl._resolve_classifier_specs(spec2)


def test_democratic_classifier_spec_resolution_valid():
    specs = (
        dcl.BaseClassifierSpec(),
        dcl.BaseClassifierSpec(),
        dcl.BaseClassifierSpec(),
    )
    spec = DemocraticCoLearningSpec(classifier_specs=specs)
    out = dcl._resolve_classifier_specs(spec)
    assert len(out) == 3


def test_democratic_resolve_classes_numpy_paths():
    y_l = np.array([0, 1, 1], dtype=np.int64)
    clfs = [_FixedPredictor(y_l, y_l), _FixedPredictor(y_l, y_l)]
    classes = dcl._resolve_classes_numpy(clfs, y_l)
    assert np.array_equal(classes, np.unique(y_l))

    class _WithClasses:
        def __init__(self, classes):
            self.classes_ = np.asarray(classes)

    with pytest.raises(InductiveValidationError, match="disagree on class labels"):
        dcl._resolve_classes_numpy([_WithClasses([0, 1]), _WithClasses([0, 2])], y_l)


def test_democratic_resolve_classes_torch_mismatch_classes_t():
    y_l = torch.tensor([0, 1], dtype=torch.int64)

    class _Clf:
        classes_t_ = torch.tensor([0, 2], dtype=torch.int64)

    with pytest.raises(InductiveValidationError, match="disagree on class labels"):
        dcl._resolve_classes_torch([_Clf()], y_l)


def test_democratic_resolve_classes_torch_mismatch_numpy_classes():
    y_l = torch.tensor([0, 1], dtype=torch.int64)

    class _Clf:
        def __init__(self, classes):
            self.classes_t_ = torch.tensor([0, 1], dtype=torch.int64)
            self.classes_ = np.asarray(classes)

    with pytest.raises(InductiveValidationError, match="disagree on class labels"):
        dcl._resolve_classes_torch([_Clf([0, 1]), _Clf([0, 2])], y_l)


def test_democratic_resolve_classes_torch_mismatch_torch_and_numpy():
    y_l = torch.tensor([0, 1], dtype=torch.int64)

    class _Clf:
        def __init__(self):
            self.classes_ = np.asarray([1, 2])

    with pytest.raises(InductiveValidationError, match="disagree on class labels"):
        dcl._resolve_classes_torch([_Clf()], y_l)


def test_democratic_resolve_classes_torch_skips_missing_numpy_classes():
    y_l = torch.tensor([0, 1], dtype=torch.int64)

    class _Clf:
        classes_t_ = torch.tensor([0, 1], dtype=torch.int64)

    class _ClfWithClasses:
        classes_t_ = torch.tensor([0, 1], dtype=torch.int64)
        classes_ = np.asarray([0, 1])

    out = dcl._resolve_classes_torch([_ClfWithClasses(), _Clf()], y_l)
    assert torch.equal(out, torch.tensor([0, 1], dtype=torch.int64))


def test_democratic_encode_predictions_torch_casts():
    classes_t = torch.tensor([0, 1], dtype=torch.int64)
    preds = [torch.tensor([[0], [1]], dtype=torch.int32)]
    encoded = dcl._encode_predictions_torch(preds, classes_t)
    assert encoded.shape == (1, 2)


def test_democratic_weighted_majority_single_class():
    preds_idx = np.zeros((3, 2), dtype=np.int64)
    weights = np.ones((3,), dtype=np.float64)
    idx, ok = dcl._weighted_majority_numpy(preds_idx, weights, n_classes=1)
    assert ok.all()
    preds_t = torch.zeros((3, 2), dtype=torch.int64)
    weights_t = torch.ones((3,), dtype=torch.float32)
    _idx_t, ok_t = dcl._weighted_majority_torch(preds_t, weights_t, n_classes=1)
    assert bool(ok_t.all())


def test_democratic_combine_scores_numpy_eligibility():
    preds_idx = np.array([[0, 1], [1, 0]], dtype=np.int64)
    weights_low = np.array([0.1, 0.2], dtype=np.float64)
    scores = dcl._combine_scores_numpy(preds_idx, weights_low, n_classes=2, min_confidence=0.9)
    assert scores.shape == (2, 2)

    weights_mix = np.array([0.1, 0.9], dtype=np.float64)
    scores_mix = dcl._combine_scores_numpy(preds_idx, weights_mix, n_classes=2, min_confidence=0.5)
    assert scores_mix.shape == (2, 2)


def test_democratic_combine_scores_torch_eligibility():
    preds_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([1.0, 0.2], dtype=torch.float32)
    scores = dcl._combine_scores_torch(preds_idx, weights, n_classes=2, min_confidence=0.1)
    assert scores.shape == (2, 2)


def test_democratic_combine_scores_torch_all_ineligible():
    preds_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([0.0, 0.0], dtype=torch.float32)
    scores = dcl._combine_scores_torch(preds_idx, weights, n_classes=2, min_confidence=10.0)
    assert scores.shape == (2, 2)


def test_democratic_fit_requires_data():
    method = DemocraticCoLearningMethod()
    with pytest.raises(InductiveValidationError, match="data must not be None"):
        method.fit(None, device=DeviceSpec(device="cpu"), seed=0)


def test_democratic_fit_numpy_empty_labeled(monkeypatch):
    bad = DummyDataset(
        X_l=np.zeros((0, 2), dtype=np.float32),
        y_l=np.array([0], dtype=np.int64),
        X_u=None,
    )
    monkeypatch.setattr(dcl, "ensure_numpy_data", lambda _data: bad)
    method = DemocraticCoLearningMethod(DemocraticCoLearningSpec(n_learners=3))
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        method.fit(bad, device=DeviceSpec(device="cpu"), seed=0)


def test_democratic_fit_torch_empty_labeled(monkeypatch):
    bad = DummyDataset(
        X_l=torch.zeros((0, 2)),
        y_l=torch.tensor([0], dtype=torch.int64),
        X_u=torch.zeros((2, 2)),
    )
    monkeypatch.setattr(dcl, "ensure_torch_data", lambda _data, device: bad)
    method = DemocraticCoLearningMethod(
        DemocraticCoLearningSpec(n_learners=3, classifier_backend="torch")
    )
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        method.fit(bad, device=DeviceSpec(device="cpu"), seed=0)


def test_democratic_fit_torch_without_unlabeled():
    data = DummyDataset(
        X_l=torch.tensor([[0.0], [1.0]]),
        y_l=torch.tensor([0, 1], dtype=torch.int64),
        X_u=None,
    )
    method = DemocraticCoLearningMethod(
        DemocraticCoLearningSpec(max_iter=1, n_learners=3, classifier_backend="torch")
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    assert method._backend == "torch"


def test_democratic_fit_numpy_updates_labels(monkeypatch):
    X_l = np.array([[0.0], [1.0]], dtype=np.float32)
    y_l = np.array([0, 1], dtype=np.int64)
    X_u = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
    data = DummyDataset(X_l=X_l, y_l=y_l, X_u=X_u)
    preds_u = [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([1, 1, 1])]
    created = []

    def _build(_spec, seed=0):
        idx = len(created)
        clf = _FixedPredictor(y_l, preds_u[idx], classes=[0, 1])
        created.append(clf)
        return clf

    monkeypatch.setattr(dcl, "build_classifier", _build)
    method = DemocraticCoLearningMethod(DemocraticCoLearningSpec(max_iter=1, n_learners=3))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    assert method._backend == "numpy"


def test_democratic_fit_numpy_no_label_updates(monkeypatch):
    X_l = np.array([[0.0], [1.0]], dtype=np.float32)
    y_l = np.array([0, 1], dtype=np.int64)
    X_u = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
    data = DummyDataset(X_l=X_l, y_l=y_l, X_u=X_u)
    preds_u = [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([1, 1, 1])]
    pred_l = np.array([0, 0], dtype=np.int64)
    created = []

    def _build(_spec, seed=0):
        idx = len(created)
        clf = _FixedPredictorWithLabeled(pred_l, preds_u[idx], classes=[0, 1])
        created.append(clf)
        return clf

    monkeypatch.setattr(dcl, "build_classifier", _build)
    method = DemocraticCoLearningMethod(DemocraticCoLearningSpec(max_iter=1, n_learners=3))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    assert method._backend == "numpy"


def test_democratic_fit_torch_updates_labels(monkeypatch):
    X_l = torch.tensor([[0.0], [1.0]])
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    X_u = torch.tensor([[2.0], [3.0], [4.0]])
    data = DummyDataset(X_l=X_l, y_l=y_l, X_u=X_u)
    preds_u = [
        torch.tensor([0, 0, 0], dtype=torch.int64),
        torch.tensor([0, 0, 0], dtype=torch.int64),
        torch.tensor([1, 1, 1], dtype=torch.int64),
    ]
    created = []

    def _build(_spec, seed=0):
        idx = len(created)
        clf = _TorchPredictor(y_l, preds_u[idx], classes_t=torch.tensor([0, 1]))
        created.append(clf)
        return clf

    monkeypatch.setattr(dcl, "build_classifier", _build)
    method = DemocraticCoLearningMethod(
        DemocraticCoLearningSpec(max_iter=1, n_learners=3, classifier_backend="torch")
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    assert method._backend == "torch"


def test_democratic_fit_torch_no_label_updates(monkeypatch):
    X_l = torch.tensor([[0.0], [1.0]])
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    X_u = torch.tensor([[2.0], [3.0], [4.0]])
    data = DummyDataset(X_l=X_l, y_l=y_l, X_u=X_u)
    preds_u = [
        torch.tensor([0, 0, 0], dtype=torch.int64),
        torch.tensor([0, 0, 0], dtype=torch.int64),
        torch.tensor([1, 1, 1], dtype=torch.int64),
    ]
    pred_l = torch.tensor([0, 0], dtype=torch.int64)
    created = []

    def _build(_spec, seed=0):
        idx = len(created)
        clf = _TorchPredictorWithLabeled(pred_l, preds_u[idx], classes_t=torch.tensor([0, 1]))
        created.append(clf)
        return clf

    monkeypatch.setattr(dcl, "build_classifier", _build)
    method = DemocraticCoLearningMethod(
        DemocraticCoLearningSpec(max_iter=1, n_learners=3, classifier_backend="torch")
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    assert method._backend == "torch"


def test_democratic_predict_proba_error_paths():
    method = DemocraticCoLearningMethod()
    with pytest.raises(RuntimeError, match="not fitted"):
        method.predict_proba(np.zeros((1, 2), dtype=np.float32))

    method._clfs = [_FixedPredictor([0], [0])]
    method._weights = np.ones((1,), dtype=np.float64)
    method._backend = ""
    with pytest.raises(InductiveValidationError, match="backend mismatch"):
        method.predict_proba(torch.zeros((1, 2)))

    method2 = DemocraticCoLearningMethod()
    method2._clfs = [_FixedPredictor([0], [0])]
    with pytest.raises(RuntimeError, match="missing weights"):
        method2.predict_proba(np.zeros((1, 2), dtype=np.float32))

    method3 = DemocraticCoLearningMethod()
    method3._clfs = [_FixedPredictor([0], [0])]
    method3._weights = np.ones((1,), dtype=np.float64)
    method3._backend = "numpy"
    with pytest.raises(RuntimeError, match="missing classes"):
        method3.predict_proba(np.zeros((1, 2), dtype=np.float32))

    method4 = DemocraticCoLearningMethod()
    method4._clfs = [_TorchPredictor(torch.tensor([0]), torch.tensor([0]))]
    method4._weights = np.ones((1,), dtype=np.float64)
    method4._backend = "torch"
    with pytest.raises(RuntimeError, match="missing classes"):
        method4.predict_proba(torch.zeros((1, 2)))


def test_democratic_predict_returns_idx_when_classes_missing(monkeypatch):
    method = DemocraticCoLearningMethod()
    method._backend = "numpy"
    monkeypatch.setattr(
        method,
        "predict_proba",
        lambda _x: np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32),
    )
    pred = method.predict(np.zeros((2, 2), dtype=np.float32))
    assert pred.shape == (2,)

    method2 = DemocraticCoLearningMethod()
    method2._backend = "torch"
    monkeypatch.setattr(method2, "predict_proba", lambda _x: torch.tensor([[0.2, 0.8], [0.9, 0.1]]))
    pred2 = method2.predict(torch.zeros((2, 2)))
    assert int(pred2.shape[0]) == 2
