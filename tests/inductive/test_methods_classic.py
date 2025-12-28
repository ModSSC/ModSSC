from __future__ import annotations

import numpy as np
import pytest
import torch

from modssc.evaluation import accuracy as accuracy_score
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.co_training import (
    CoTrainingMethod,
    CoTrainingSpec,
    _view_payload_numpy,
    _view_payload_torch,
    _view_predict_payload_numpy,
    _view_predict_payload_torch,
)
from modssc.inductive.methods.pseudo_label import PseudoLabelMethod, PseudoLabelSpec
from modssc.inductive.methods.s4vm import S4VMMethod, S4VMSpec
from modssc.inductive.methods.self_training import SelfTrainingMethod, SelfTrainingSpec
from modssc.inductive.methods.tri_training import TriTrainingMethod, TriTrainingSpec
from modssc.inductive.methods.tsvm import (
    TSVMMethod,
    TSVMSpec,
    _batch_indices,
    _encode_binary,
    _LinearSVM,
)
from modssc.inductive.types import DeviceSpec, InductiveDataset

from .conftest import DummyDataset, make_numpy_dataset, make_torch_dataset


@pytest.mark.parametrize(
    "method_cls,spec_cls",
    [(PseudoLabelMethod, PseudoLabelSpec), (SelfTrainingMethod, SelfTrainingSpec)],
)
def test_classic_numpy_methods_fit_predict(method_cls, spec_cls):
    data = make_numpy_dataset()
    spec = spec_cls(max_iter=1, confidence_threshold=0.0, max_new_labels=2, min_new_labels=1)
    method = method_cls(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    assert proba.shape[0] == data.X_l.shape[0]
    pred = method.predict(data.X_l)
    assert pred.shape[0] == data.X_l.shape[0]

    data_none = DummyDataset(X_l=data.X_l, y_l=data.y_l, X_u=None)
    method2 = method_cls(spec)
    method2.fit(data_none, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    "method_cls,spec_cls",
    [(PseudoLabelMethod, PseudoLabelSpec), (SelfTrainingMethod, SelfTrainingSpec)],
)
def test_classic_torch_methods_fit_predict(method_cls, spec_cls):
    data = make_torch_dataset()
    spec = spec_cls(
        max_iter=1,
        confidence_threshold=0.0,
        max_new_labels=2,
        min_new_labels=1,
        classifier_backend="torch",
    )
    method = method_cls(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])
    pred = method.predict(data.X_l)
    assert int(pred.shape[0]) == int(data.X_l.shape[0])


def test_classic_methods_errors_and_backend_mismatch():
    data = make_numpy_dataset()
    method = PseudoLabelMethod(PseudoLabelSpec())
    with pytest.raises(RuntimeError):
        method.predict_proba(data.X_l)

    PseudoLabelMethod(PseudoLabelSpec(min_new_labels=10)).fit(
        data, device=DeviceSpec(device="cpu"), seed=0
    )
    SelfTrainingMethod(SelfTrainingSpec(min_new_labels=10)).fit(
        data, device=DeviceSpec(device="cpu"), seed=0
    )

    data_empty = DummyDataset(
        X_l=np.zeros((0, 2), dtype=np.float32), y_l=np.array([], dtype=np.int64), X_u=data.X_u
    )
    with pytest.raises(InductiveValidationError):
        SelfTrainingMethod(SelfTrainingSpec()).fit(
            data_empty, device=DeviceSpec(device="cpu"), seed=0
        )

    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    method._backend = ""
    with pytest.raises(InductiveValidationError):
        method.predict_proba(torch.tensor([[0.0, 1.0]]))
    with pytest.raises(InductiveValidationError):
        method.predict(torch.tensor([[0.0, 1.0]]))

    with pytest.raises(RuntimeError):
        PseudoLabelMethod(PseudoLabelSpec()).predict(data.X_l)

    st = SelfTrainingMethod(SelfTrainingSpec())
    with pytest.raises(RuntimeError):
        st.predict(data.X_l)
    with pytest.raises(RuntimeError):
        st.predict_proba(data.X_l)
    st.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    st._backend = ""
    with pytest.raises(InductiveValidationError):
        st.predict_proba(torch.tensor([[0.0, 1.0]]))
    with pytest.raises(InductiveValidationError):
        st.predict(torch.tensor([[0.0, 1.0]]))


def _make_views_numpy():
    data = make_numpy_dataset()
    v1 = {"X_l": data.X_l, "X_u": data.X_u}
    v2 = {"X_l": data.X_l.copy(), "X_u": data.X_u.copy()}
    views = {"v1": v1, "v2": v2}
    return data, views


def _make_views_torch():
    data = make_torch_dataset()
    v1 = {"X_l": data.X_l, "X_u": data.X_u}
    v2 = {"X_l": data.X_l.clone(), "X_u": data.X_u.clone()}
    views = {"v1": v1, "v2": v2}
    return data, views


def test_co_training_helpers_and_errors():
    data, views = _make_views_numpy()
    _view_payload_numpy(views["v1"], name="v1")
    _view_predict_payload_numpy({"X": data.X_l}, name="v1")
    with pytest.raises(InductiveValidationError):
        _view_payload_numpy({"X_l": data.X_l}, name="v1")
    with pytest.raises(InductiveValidationError):
        _view_payload_numpy((data.X_l,), name="v1")
    with pytest.raises(InductiveValidationError):
        _view_payload_numpy((data.X_l, [1, 2, 3]), name="v1")
    with pytest.raises(InductiveValidationError):
        _view_payload_numpy((data.X_l.reshape(-1), data.X_u), name="v1")

    with pytest.raises(InductiveValidationError):
        _view_predict_payload_numpy({"bad": data.X_l}, name="v1")
    _view_predict_payload_numpy({"X_u": data.X_l}, name="v1")
    _view_predict_payload_numpy({"X_l": data.X_l}, name="v1")
    with pytest.raises(InductiveValidationError):
        _view_predict_payload_numpy([1, 2, 3], name="v1")
    with pytest.raises(InductiveValidationError):
        _view_predict_payload_numpy(np.array([1, 2, 3]), name="v1")

    data_t, views_t = _make_views_torch()
    _view_payload_torch(views_t["v1"], name="v1")
    _view_predict_payload_torch({"X": data_t.X_l}, name="v1")
    with pytest.raises(InductiveValidationError):
        _view_payload_torch({"X_l": data_t.X_l}, name="v1")
    with pytest.raises(InductiveValidationError):
        _view_payload_torch((data_t.X_l,), name="v1")
    with pytest.raises(InductiveValidationError):
        _view_payload_torch((data_t.X_l, np.zeros((2, 2))), name="v1")
    with pytest.raises(InductiveValidationError):
        _view_payload_torch((data_t.X_l.reshape(-1), data_t.X_u), name="v1")
    with pytest.raises(InductiveValidationError):
        _view_payload_torch((data_t.X_l, torch.zeros_like(data_t.X_u, device="meta")), name="v1")

    with pytest.raises(InductiveValidationError):
        _view_predict_payload_torch({"bad": data_t.X_l}, name="v1")
    _view_predict_payload_torch({"X_u": data_t.X_l}, name="v1")
    _view_predict_payload_torch({"X_l": data_t.X_l}, name="v1")
    with pytest.raises(InductiveValidationError):
        _view_predict_payload_torch(np.zeros((2, 2)), name="v1")
    with pytest.raises(InductiveValidationError):
        _view_predict_payload_torch(torch.zeros((2,)), name="v1")


def test_co_training_numpy_fit_predict():
    data, views = _make_views_numpy()
    ds = DummyDataset(X_l=data.X_l, y_l=data.y_l, views=views)
    spec = CoTrainingSpec(max_iter=1, k_per_class=1, confidence_threshold=0.0)
    method = CoTrainingMethod(spec)
    method.fit(ds, device=DeviceSpec(device="cpu"), seed=0)

    pred_data = DummyDataset(
        X_l=data.X_l, y_l=data.y_l, views={"v1": {"X": data.X_l}, "v2": {"X": data.X_l}}
    )
    proba = method.predict_proba(pred_data)
    assert proba.shape[0] == data.X_l.shape[0]
    pred = method.predict(pred_data)
    assert pred.shape[0] == data.X_l.shape[0]

    with pytest.raises(InductiveValidationError):
        method.predict_proba(DummyDataset(X_l=data.X_l, y_l=data.y_l, views=None))
    with pytest.raises(InductiveValidationError):
        method.predict_proba(
            DummyDataset(X_l=data.X_l, y_l=data.y_l, views={"v1": {"X": data.X_l}})
        )
    with pytest.raises(RuntimeError):
        CoTrainingMethod(CoTrainingSpec()).predict_proba(pred_data)


def test_co_training_torch_fit_predict():
    data, views = _make_views_torch()
    ds = DummyDataset(X_l=data.X_l, y_l=data.y_l, views=views)
    spec = CoTrainingSpec(
        max_iter=1, k_per_class=1, confidence_threshold=1.1, classifier_backend="torch"
    )
    method = CoTrainingMethod(spec)
    method.fit(ds, device=DeviceSpec(device="cpu"), seed=0)

    pred_views = {"v1": {"X": data.X_l}, "v2": {"X": data.X_l}}
    proba = method.predict_proba(DummyDataset(X_l=data.X_l, y_l=data.y_l, views=pred_views))
    assert int(proba.shape[0]) == int(data.X_l.shape[0])


def test_co_training_errors_and_predict_scores_pair():
    data, views = _make_views_numpy()
    method = CoTrainingMethod(CoTrainingSpec())
    with pytest.raises(InductiveValidationError):
        method.fit(None, device=DeviceSpec(device="cpu"), seed=0)
    with pytest.raises(InductiveValidationError):
        method.fit(
            DummyDataset(X_l=data.X_l, y_l=data.y_l, views=None),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )
    with pytest.raises(InductiveValidationError):
        method.fit(
            DummyDataset(X_l=data.X_l, y_l=data.y_l, views={"v1": views["v1"]}),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )
    with pytest.raises(InductiveValidationError):
        method.fit(
            DummyDataset(X_l=data.X_l.tolist(), y_l=data.y_l, views=views),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )
    with pytest.raises(InductiveValidationError):
        method.fit(
            DummyDataset(X_l=data.X_l, y_l=data.y_l.tolist(), views=views),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    bad_views = {
        "v1": {"X_l": data.X_l[:1], "X_u": data.X_u},
        "v2": {"X_l": data.X_l, "X_u": data.X_u},
    }
    with pytest.raises(InductiveValidationError):
        method.fit(
            DummyDataset(X_l=data.X_l, y_l=data.y_l, views=bad_views),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    bad_views_u = {
        "v1": {"X_l": data.X_l, "X_u": data.X_u[:1]},
        "v2": {"X_l": data.X_l, "X_u": data.X_u},
    }
    with pytest.raises(InductiveValidationError):
        method.fit(
            DummyDataset(X_l=data.X_l, y_l=data.y_l, views=bad_views_u),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    class _Dummy:
        def __init__(self, scores, classes=None):
            self._scores = scores
            self.classes_ = classes

        def predict_scores(self, X):
            return self._scores

    method._clf1 = _Dummy(np.ones((2, 2), dtype=np.float32), classes=np.array([0, 1]))
    method._clf2 = _Dummy(np.ones((2, 3), dtype=np.float32), classes=np.array([0, 1]))
    method._backend = "numpy"
    with pytest.raises(InductiveValidationError):
        method._predict_scores_pair(np.zeros((2, 2)), np.zeros((2, 2)))

    method._clf2 = _Dummy(np.ones((2, 2), dtype=np.float32), classes=np.array([1, 2]))
    with pytest.raises(InductiveValidationError):
        method._predict_scores_pair(np.zeros((2, 2)), np.zeros((2, 2)))

    method._clf1 = _Dummy(np.ones((data.X_l.shape[0], 2), dtype=np.float32))
    method._clf2 = _Dummy(np.ones((data.X_l.shape[0], 2), dtype=np.float32))
    method._backend = "numpy"
    method._view_keys = ("v1", "v2")
    pred_data = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        views={"v1": {"X": data.X_l}, "v2": {"X": data.X_l}},
    )
    out = method.predict(pred_data)
    assert out.shape[0] == data.X_l.shape[0]

    method._clf1.classes_ = None
    out2 = method.predict(pred_data)
    assert out2.shape[0] == data.X_l.shape[0]

    data_t, views_t = _make_views_torch()

    class _DummyT:
        def __init__(self):
            self.classes_t_ = None

        def predict_scores(self, X):
            return torch.ones((int(X.shape[0]), 2))

    method._clf1 = _DummyT()
    method._clf2 = _DummyT()
    method._backend = "torch"
    method._view_keys = ("v1", "v2")
    pred_t = method.predict(
        DummyDataset(
            X_l=data_t.X_l, y_l=data_t.y_l, views={"v1": {"X": data_t.X_l}, "v2": {"X": data_t.X_l}}
        )
    )
    assert int(pred_t.shape[0]) == int(data_t.X_l.shape[0])


def test_co_training_type_checks_and_backend_mismatch(monkeypatch):
    data, views = _make_views_numpy()
    method = CoTrainingMethod(CoTrainingSpec())

    monkeypatch.setattr("modssc.inductive.methods.co_training.detect_backend", lambda _x: "numpy")
    with pytest.raises(InductiveValidationError):
        method.fit(
            DummyDataset(X_l=[[1.0, 2.0]], y_l=data.y_l, views=views),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    monkeypatch.setattr("modssc.inductive.methods.co_training.detect_backend", lambda _x: "torch")
    with pytest.raises(InductiveValidationError):
        method.fit(
            DummyDataset(X_l=torch.zeros((2, 2)), y_l=data.y_l, views=views),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    class _Dummy:
        def __init__(self, scores):
            self._scores = scores

        def predict_scores(self, X):
            return self._scores

    method._clf1 = _Dummy(np.ones((2, 2), dtype=np.float32))
    method._clf2 = _Dummy(np.ones((2, 2), dtype=np.float32))
    method._backend = ""
    with pytest.raises(InductiveValidationError):
        method._predict_scores_pair(np.zeros((2, 2)), np.zeros((2, 2)))

    method._clf1 = None
    with pytest.raises(RuntimeError):
        method.predict(DummyDataset(X_l=data.X_l, y_l=data.y_l, views=views))


def test_co_training_selection_branches_numpy(monkeypatch):
    data, views = _make_views_numpy()
    ds = DummyDataset(X_l=data.X_l, y_l=data.y_l, views=views)

    calls = {"n": 0}

    def fake_select(_scores, *, k_per_class, threshold):
        calls["n"] += 1
        return np.array([0], dtype=np.int64) if calls["n"] == 1 else np.array([], dtype=np.int64)

    monkeypatch.setattr("modssc.inductive.methods.co_training.select_top_per_class", fake_select)
    CoTrainingMethod(CoTrainingSpec(max_iter=1, k_per_class=1)).fit(
        ds, device=DeviceSpec(device="cpu"), seed=0
    )

    def fake_select2(_scores, *, k_per_class, threshold):
        return np.array([], dtype=np.int64)

    monkeypatch.setattr("modssc.inductive.methods.co_training.select_top_per_class", fake_select2)
    CoTrainingMethod(CoTrainingSpec(max_iter=1, k_per_class=1)).fit(
        ds, device=DeviceSpec(device="cpu"), seed=0
    )


def test_co_training_selection_branches_torch(monkeypatch):
    data, views = _make_views_torch()
    ds = DummyDataset(X_l=data.X_l, y_l=data.y_l, views=views)

    calls = {"n": 0}

    def fake_select(_scores, *, k_per_class, threshold):
        calls["n"] += 1
        return (
            torch.tensor([0], dtype=torch.long)
            if calls["n"] == 1
            else torch.tensor([], dtype=torch.long)
        )

    monkeypatch.setattr(
        "modssc.inductive.methods.co_training.select_top_per_class_torch", fake_select
    )
    CoTrainingMethod(CoTrainingSpec(max_iter=1, k_per_class=1, classifier_backend="torch")).fit(
        ds, device=DeviceSpec(device="cpu"), seed=0
    )


def test_co_training_empty_unlabeled_break():
    data = make_numpy_dataset()
    empty_u = np.zeros((0, data.X_l.shape[1]), dtype=np.float32)
    views = {"v1": {"X_l": data.X_l, "X_u": empty_u}, "v2": {"X_l": data.X_l, "X_u": empty_u}}
    ds = DummyDataset(X_l=data.X_l, y_l=data.y_l, views=views)
    CoTrainingMethod(CoTrainingSpec(max_iter=1, k_per_class=1)).fit(
        ds, device=DeviceSpec(device="cpu"), seed=0
    )


def test_co_training_torch_type_checks(monkeypatch):
    data, views = _make_views_numpy()

    monkeypatch.setattr("modssc.inductive.methods.co_training.detect_backend", lambda _x: "torch")
    spec = CoTrainingSpec(classifier_backend="torch")
    method = CoTrainingMethod(spec)
    with pytest.raises(InductiveValidationError):
        method.fit(
            DummyDataset(X_l=data.X_l, y_l=torch.tensor([0, 1]), views=views),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    with pytest.raises(InductiveValidationError):
        method.fit(
            DummyDataset(X_l=torch.zeros((2, 2)), y_l=data.y_l, views=views),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )


def test_co_training_view_keys_spec():
    data, views = _make_views_numpy()
    ds = DummyDataset(X_l=data.X_l, y_l=data.y_l, views=views)
    spec = CoTrainingSpec(view_keys=("v1", "v2"))
    CoTrainingMethod(spec).fit(ds, device=DeviceSpec(device="cpu"), seed=0)


def test_co_training_numpy_idx2_only(monkeypatch):
    data, views = _make_views_numpy()
    ds = DummyDataset(X_l=data.X_l, y_l=data.y_l, views=views)

    calls = {"n": 0}

    def fake_select(_scores, *, k_per_class, threshold):
        calls["n"] += 1
        return np.array([], dtype=np.int64) if calls["n"] == 1 else np.array([0], dtype=np.int64)

    monkeypatch.setattr("modssc.inductive.methods.co_training.select_top_per_class", fake_select)
    CoTrainingMethod(CoTrainingSpec(max_iter=1, k_per_class=1)).fit(
        ds, device=DeviceSpec(device="cpu"), seed=0
    )


def test_co_training_torch_idx2_only(monkeypatch):
    data, views = _make_views_torch()
    ds = DummyDataset(X_l=data.X_l, y_l=data.y_l, views=views)

    calls = {"n": 0}

    def fake_select(_scores, *, k_per_class, threshold):
        calls["n"] += 1
        return (
            torch.tensor([], dtype=torch.long)
            if calls["n"] == 1
            else torch.tensor([0], dtype=torch.long)
        )

    monkeypatch.setattr(
        "modssc.inductive.methods.co_training.select_top_per_class_torch", fake_select
    )
    CoTrainingMethod(CoTrainingSpec(max_iter=1, k_per_class=1, classifier_backend="torch")).fit(
        ds, device=DeviceSpec(device="cpu"), seed=0
    )


def test_co_training_predict_scores_pair_not_fitted():
    data = make_numpy_dataset()
    method = CoTrainingMethod(CoTrainingSpec())
    with pytest.raises(RuntimeError):
        method._predict_scores_pair(data.X_l, data.X_l)


def test_co_training_predict_proba_backend_mismatch():
    data, views = _make_views_torch()
    ds = DummyDataset(X_l=data.X_l, y_l=data.y_l, views=views)
    spec = CoTrainingSpec(max_iter=1, k_per_class=1, classifier_backend="torch")
    method = CoTrainingMethod(spec)
    method.fit(ds, device=DeviceSpec(device="cpu"), seed=0)
    method._backend = ""
    with pytest.raises(InductiveValidationError):
        method.predict_proba(ds)


def test_co_training_predict_torch_classes():
    data, views = _make_views_torch()

    class _Dummy:
        def __init__(self):
            self.classes_t_ = torch.tensor([10, 11])

        def predict_scores(self, X):
            return torch.ones((int(X.shape[0]), 2), device=X.device)

    method = CoTrainingMethod(CoTrainingSpec(classifier_backend="torch"))
    method._clf1 = _Dummy()
    method._clf2 = _Dummy()
    method._backend = "torch"
    method._view_keys = ("v1", "v2")
    pred = method.predict(
        DummyDataset(
            X_l=data.X_l, y_l=data.y_l, views={"v1": {"X": data.X_l}, "v2": {"X": data.X_l}}
        )
    )
    assert int(pred.shape[0]) == int(data.X_l.shape[0])


def test_tri_training_numpy_and_torch():
    data = make_numpy_dataset()
    method = TriTrainingMethod(
        TriTrainingSpec(max_iter=1, confidence_threshold=1.1, max_new_labels=1)
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    class _Dummy:
        def __init__(self, scores):
            self._scores = scores

        def predict_scores(self, X):
            return self._scores

    method._clfs = [_Dummy(np.ones((data.X_l.shape[0], 2), dtype=np.float32)) for _ in range(3)]
    method._backend = "numpy"
    proba = method.predict_proba(data.X_l)
    assert proba.shape[0] == data.X_l.shape[0]

    data_t = make_torch_dataset()
    method_t = TriTrainingMethod(
        TriTrainingSpec(
            max_iter=1, confidence_threshold=1.1, max_new_labels=1, classifier_backend="torch"
        )
    )
    method_t.fit(data_t, device=DeviceSpec(device="cpu"), seed=0)
    method_t._clfs = [_Dummy(torch.ones((int(data_t.X_l.shape[0]), 2))) for _ in range(3)]
    method_t._backend = "torch"
    proba_t = method_t.predict_proba(data_t.X_l)
    assert int(proba_t.shape[0]) == int(data_t.X_l.shape[0])


def test_tri_training_errors_and_predict():
    data = make_numpy_dataset()
    method = TriTrainingMethod(TriTrainingSpec(max_iter=1))
    with pytest.raises(InductiveValidationError):
        method.fit(
            DummyDataset(X_l=data.X_l, y_l=data.y_l, X_u=None),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    with pytest.raises(RuntimeError):
        TriTrainingMethod(TriTrainingSpec()).predict_proba(data.X_l)

    class _Dummy:
        def __init__(self, scores):
            self._scores = scores

        def predict_scores(self, X):
            return self._scores

    method._clfs = [_Dummy(np.ones((2, 2))), _Dummy(np.ones((2, 3)))]
    method._backend = "numpy"
    with pytest.raises(InductiveValidationError):
        method.predict_proba(np.zeros((2, 2)))

    method._clfs = [_Dummy(np.ones((2, 2)))]
    method._backend = "numpy"
    method._clfs[0].classes_ = None
    pred = method.predict(np.zeros((2, 2)))
    assert pred.shape[0] == 2


def test_tri_training_branching_numpy(monkeypatch):
    data = make_numpy_dataset()

    class _DummyClf:
        def fit(self, X, y):
            return self

    monkeypatch.setattr(
        "modssc.inductive.methods.tri_training.build_classifier", lambda spec, seed: _DummyClf()
    )

    calls = {"n": 0}

    def scores_agree(_model, X, *, backend):
        return np.tile(np.array([[0.9, 0.1]], dtype=np.float32), (X.shape[0], 1))

    def scores_no_agree(_model, X, *, backend):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            return np.tile(np.array([[0.9, 0.1]], dtype=np.float32), (X.shape[0], 1))
        return np.tile(np.array([[0.1, 0.9]], dtype=np.float32), (X.shape[0], 1))

    monkeypatch.setattr("modssc.inductive.methods.tri_training.predict_scores", scores_agree)
    TriTrainingMethod(TriTrainingSpec(max_iter=1, max_new_labels=1)).fit(
        data, device=DeviceSpec(device="cpu"), seed=0
    )

    monkeypatch.setattr("modssc.inductive.methods.tri_training.predict_scores", scores_no_agree)
    TriTrainingMethod(TriTrainingSpec(max_iter=1, confidence_threshold=0.5)).fit(
        data, device=DeviceSpec(device="cpu"), seed=0
    )

    monkeypatch.setattr("modssc.inductive.methods.tri_training.predict_scores", scores_agree)
    TriTrainingMethod(TriTrainingSpec(max_iter=2, max_new_labels=1)).fit(
        data, device=DeviceSpec(device="cpu"), seed=0
    )


def test_tri_training_branching_torch(monkeypatch):
    data = make_torch_dataset()

    class _DummyClf:
        def fit(self, X, y):
            return self

    monkeypatch.setattr(
        "modssc.inductive.methods.tri_training.build_classifier", lambda spec, seed: _DummyClf()
    )

    calls = {"n": 0}

    def scores_agree(_model, X, *, backend):
        return torch.ones((int(X.shape[0]), 2))

    def scores_no_agree(_model, X, *, backend):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            return torch.tensor([[1.0, 0.0]]).repeat(int(X.shape[0]), 1)
        return torch.tensor([[0.0, 1.0]]).repeat(int(X.shape[0]), 1)

    monkeypatch.setattr("modssc.inductive.methods.tri_training.predict_scores", scores_agree)
    TriTrainingMethod(
        TriTrainingSpec(max_iter=1, max_new_labels=1, classifier_backend="torch")
    ).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    monkeypatch.setattr("modssc.inductive.methods.tri_training.predict_scores", scores_no_agree)
    TriTrainingMethod(
        TriTrainingSpec(max_iter=1, confidence_threshold=0.5, classifier_backend="torch")
    ).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_tri_training_empty_xl_and_missing_u(monkeypatch):
    data = make_numpy_dataset()
    empty_np = DummyDataset(
        X_l=np.zeros((0, data.X_l.shape[1]), dtype=np.float32),
        y_l=np.array([0], dtype=np.int64),
        X_u=data.X_u,
    )
    monkeypatch.setattr(
        "modssc.inductive.methods.tri_training.ensure_numpy_data",
        lambda data: data,
    )
    with pytest.raises(InductiveValidationError):
        TriTrainingMethod(TriTrainingSpec()).fit(empty_np, device=DeviceSpec(device="cpu"), seed=0)

    data_t = make_torch_dataset()
    empty_t = DummyDataset(
        X_l=torch.zeros((0, data_t.X_l.shape[1])),
        y_l=torch.zeros((0,), dtype=torch.int64),
        X_u=data_t.X_u,
    )
    monkeypatch.setattr(
        "modssc.inductive.methods.tri_training.ensure_1d_labels_torch",
        lambda y, name="y_l": y,
    )
    with pytest.raises(InductiveValidationError):
        TriTrainingMethod(TriTrainingSpec(classifier_backend="torch")).fit(
            empty_t, device=DeviceSpec(device="cpu"), seed=0
        )

    missing_u = DummyDataset(X_l=data_t.X_l, y_l=data_t.y_l, X_u=None)
    with pytest.raises(InductiveValidationError):
        TriTrainingMethod(TriTrainingSpec(classifier_backend="torch")).fit(
            missing_u, device=DeviceSpec(device="cpu"), seed=0
        )


def test_tri_training_max_new_labels_numpy(monkeypatch):
    data = make_numpy_dataset()

    class _DummyClf:
        def fit(self, X, y):
            return self

    monkeypatch.setattr(
        "modssc.inductive.methods.tri_training.build_classifier",
        lambda spec, seed: _DummyClf(),
    )

    def scores(_model, X, *, backend):
        return np.tile(np.array([[0.9, 0.1]], dtype=np.float32), (X.shape[0], 1))

    monkeypatch.setattr("modssc.inductive.methods.tri_training.predict_scores", scores)
    TriTrainingMethod(TriTrainingSpec(max_iter=1, max_new_labels=1)).fit(
        data, device=DeviceSpec(device="cpu"), seed=0
    )


def test_tri_training_repeat_agreement_numpy(monkeypatch):
    data = make_numpy_dataset()

    class _DummyClf:
        def fit(self, X, y):
            return self

    monkeypatch.setattr(
        "modssc.inductive.methods.tri_training.build_classifier",
        lambda spec, seed: _DummyClf(),
    )

    def scores(_model, X, *, backend):
        return np.tile(np.array([[0.9, 0.1]], dtype=np.float32), (X.shape[0], 1))

    monkeypatch.setattr("modssc.inductive.methods.tri_training.predict_scores", scores)
    TriTrainingMethod(TriTrainingSpec(max_iter=2, max_new_labels=None)).fit(
        data, device=DeviceSpec(device="cpu"), seed=0
    )


def test_tri_training_max_new_labels_torch(monkeypatch):
    data = make_torch_dataset()

    class _DummyClf:
        def fit(self, X, y):
            return self

    monkeypatch.setattr(
        "modssc.inductive.methods.tri_training.build_classifier",
        lambda spec, seed: _DummyClf(),
    )

    def scores(_model, X, *, backend):
        return torch.tensor([[0.9, 0.1]], device=X.device).repeat(int(X.shape[0]), 1)

    monkeypatch.setattr("modssc.inductive.methods.tri_training.predict_scores", scores)
    TriTrainingMethod(
        TriTrainingSpec(max_iter=1, max_new_labels=1, classifier_backend="torch")
    ).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_tri_training_repeat_agreement_torch(monkeypatch):
    data = make_torch_dataset()

    class _DummyClf:
        def fit(self, X, y):
            return self

    monkeypatch.setattr(
        "modssc.inductive.methods.tri_training.build_classifier",
        lambda spec, seed: _DummyClf(),
    )

    def scores(_model, X, *, backend):
        return torch.tensor([[0.9, 0.1]], device=X.device).repeat(int(X.shape[0]), 1)

    monkeypatch.setattr("modssc.inductive.methods.tri_training.predict_scores", scores)
    TriTrainingMethod(
        TriTrainingSpec(max_iter=2, max_new_labels=None, classifier_backend="torch")
    ).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_tri_training_predict_backend_mismatch_and_classes_t():
    data_t = make_torch_dataset()

    class _Dummy:
        def __init__(self, scores):
            self._scores = scores
            self.classes_t_ = torch.tensor([10, 11])

        def predict_scores(self, X):
            return self._scores

    method = TriTrainingMethod(TriTrainingSpec())
    method._clfs = [_Dummy(np.ones((2, 2), dtype=np.float32)) for _ in range(3)]
    method._backend = ""
    with pytest.raises(InductiveValidationError):
        method.predict_proba(data_t.X_l)

    method_t = TriTrainingMethod(TriTrainingSpec())
    method_t._clfs = [_Dummy(torch.ones((int(data_t.X_l.shape[0]), 2))) for _ in range(3)]
    method_t._backend = "torch"
    pred = method_t.predict(data_t.X_l)
    assert int(pred.shape[0]) == int(data_t.X_l.shape[0])


def test_tri_training_predict_classes_numpy_and_torch():
    class _Dummy:
        def __init__(self):
            self.classes_ = np.array([1, 2])

        def predict_scores(self, X):
            return np.tile(np.array([[0.9, 0.1]], dtype=np.float32), (X.shape[0], 1))

    method = TriTrainingMethod(TriTrainingSpec())
    method._clfs = [_Dummy(), _Dummy(), _Dummy()]
    method._backend = "numpy"
    pred = method.predict(np.zeros((2, 2), dtype=np.float32))
    assert pred.shape[0] == 2

    data_t = make_torch_dataset()

    class _DummyT:
        def predict_scores(self, X):
            return torch.ones((int(X.shape[0]), 2), device=X.device)

    method_t = TriTrainingMethod(TriTrainingSpec())
    method_t._clfs = [_DummyT(), _DummyT(), _DummyT()]
    method_t._backend = "torch"
    pred_t = method_t.predict(data_t.X_l)
    assert int(pred_t.shape[0]) == int(data_t.X_l.shape[0])


def test_pseudo_label_additional_branches_numpy_torch():
    data = make_numpy_dataset()
    empty = DummyDataset(
        X_l=np.zeros((0, 2), dtype=np.float32),
        y_l=np.array([0], dtype=np.int64),
        X_u=data.X_u,
    )
    with pytest.raises(InductiveValidationError):
        PseudoLabelMethod(PseudoLabelSpec()).fit(empty, device=DeviceSpec(device="cpu"), seed=0)

    spec = PseudoLabelSpec(max_iter=2, confidence_threshold=0.0, min_new_labels=1)
    PseudoLabelMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    data_t = make_torch_dataset()
    spec_t = PseudoLabelSpec(
        max_iter=2,
        confidence_threshold=0.0,
        min_new_labels=1,
        classifier_backend="torch",
    )
    PseudoLabelMethod(spec_t).fit(data_t, device=DeviceSpec(device="cpu"), seed=0)

    data_t_none = DummyDataset(X_l=data_t.X_l, y_l=data_t.y_l, X_u=None)
    PseudoLabelMethod(PseudoLabelSpec(classifier_backend="torch")).fit(
        data_t_none, device=DeviceSpec(device="cpu"), seed=0
    )

    data_empty_t = DummyDataset(
        X_l=torch.zeros((0, 2)), y_l=torch.zeros((0,), dtype=torch.int64), X_u=data_t.X_u
    )
    with pytest.raises(InductiveValidationError):
        PseudoLabelMethod(PseudoLabelSpec(classifier_backend="torch")).fit(
            data_empty_t, device=DeviceSpec(device="cpu"), seed=0
        )

    spec_break = PseudoLabelSpec(
        max_iter=1, confidence_threshold=0.0, min_new_labels=10, classifier_backend="torch"
    )
    PseudoLabelMethod(spec_break).fit(data_t, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_additional_branches_numpy_torch():
    data = make_numpy_dataset()
    empty = DummyDataset(
        X_l=np.zeros((0, 2), dtype=np.float32),
        y_l=np.array([0], dtype=np.int64),
        X_u=data.X_u,
    )
    with pytest.raises(InductiveValidationError):
        SelfTrainingMethod(SelfTrainingSpec()).fit(empty, device=DeviceSpec(device="cpu"), seed=0)

    spec = SelfTrainingSpec(max_iter=2, confidence_threshold=0.0, min_new_labels=1)
    SelfTrainingMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    data_t = make_torch_dataset()
    spec_t = SelfTrainingSpec(
        max_iter=2,
        confidence_threshold=0.0,
        min_new_labels=1,
        classifier_backend="torch",
    )
    SelfTrainingMethod(spec_t).fit(data_t, device=DeviceSpec(device="cpu"), seed=0)

    data_t_none = DummyDataset(X_l=data_t.X_l, y_l=data_t.y_l, X_u=None)
    SelfTrainingMethod(SelfTrainingSpec(classifier_backend="torch")).fit(
        data_t_none, device=DeviceSpec(device="cpu"), seed=0
    )

    data_empty_t = DummyDataset(
        X_l=torch.zeros((0, 2)), y_l=torch.zeros((0,), dtype=torch.int64), X_u=data_t.X_u
    )
    with pytest.raises(InductiveValidationError):
        SelfTrainingMethod(SelfTrainingSpec(classifier_backend="torch")).fit(
            data_empty_t, device=DeviceSpec(device="cpu"), seed=0
        )

    spec_break = SelfTrainingSpec(
        max_iter=1, confidence_threshold=0.0, min_new_labels=10, classifier_backend="torch"
    )
    SelfTrainingMethod(spec_break).fit(data_t, device=DeviceSpec(device="cpu"), seed=0)


def test_pseudo_label_torch_empty_xl_hits_check(monkeypatch):
    data_t = make_torch_dataset()
    empty = DummyDataset(
        X_l=torch.zeros((0, 2)), y_l=torch.zeros((0,), dtype=torch.int64), X_u=data_t.X_u
    )

    monkeypatch.setattr(
        "modssc.inductive.methods.pseudo_label.ensure_1d_labels_torch",
        lambda y, name="y_l": y,
    )
    with pytest.raises(InductiveValidationError):
        PseudoLabelMethod(PseudoLabelSpec(classifier_backend="torch")).fit(
            empty, device=DeviceSpec(device="cpu"), seed=0
        )


def test_self_training_torch_empty_xl_hits_check(monkeypatch):
    data_t = make_torch_dataset()
    empty = DummyDataset(
        X_l=torch.zeros((0, 2)), y_l=torch.zeros((0,), dtype=torch.int64), X_u=data_t.X_u
    )

    monkeypatch.setattr(
        "modssc.inductive.methods.self_training.ensure_1d_labels_torch",
        lambda y, name="y_l": y,
    )
    with pytest.raises(InductiveValidationError):
        SelfTrainingMethod(SelfTrainingSpec(classifier_backend="torch")).fit(
            empty, device=DeviceSpec(device="cpu"), seed=0
        )


def test_pseudo_label_numpy_empty_xl_hits_check(monkeypatch):
    data = make_numpy_dataset()
    empty = DummyDataset(
        X_l=np.zeros((0, 2), dtype=np.float32),
        y_l=np.array([0], dtype=np.int64),
        X_u=data.X_u,
    )
    monkeypatch.setattr(
        "modssc.inductive.methods.pseudo_label.ensure_numpy_data",
        lambda data: data,
    )
    with pytest.raises(InductiveValidationError):
        PseudoLabelMethod(PseudoLabelSpec()).fit(empty, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_numpy_empty_xl_hits_check(monkeypatch):
    data = make_numpy_dataset()
    empty = DummyDataset(
        X_l=np.zeros((0, 2), dtype=np.float32),
        y_l=np.array([0], dtype=np.int64),
        X_u=data.X_u,
    )
    monkeypatch.setattr(
        "modssc.inductive.methods.self_training.ensure_numpy_data",
        lambda data: data,
    )
    with pytest.raises(InductiveValidationError):
        SelfTrainingMethod(SelfTrainingSpec()).fit(empty, device=DeviceSpec(device="cpu"), seed=0)


def test_tsvm_helpers_and_fit_predict_numpy():
    y_enc, classes = _encode_binary(np.array([0, 1]))
    assert classes.shape[0] == 2
    with pytest.raises(InductiveValidationError):
        _encode_binary(np.array([0, 1, 2]))

    rng = np.random.default_rng(0)
    assert list(_batch_indices(rng, np.array([], dtype=np.int64), 2)) == []

    svm = _LinearSVM(n_features=2, seed=0)
    svm.w[:] = 10.0
    svm.fit_sgd(
        np.array([[1.0, 1.0]], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        epochs=1,
        batch_size=1,
        lr=0.1,
        C=1.0,
        l2=1.0,
        rng=rng,
    )

    data = InductiveDataset(
        X_l=np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32),
        y_l=np.array([0, 1], dtype=np.int64),
        X_u=np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    spec = TSVMSpec(max_iter=1, epochs_per_iter=1, batch_size=2, balance=True)
    method = TSVMMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    assert proba.shape[0] == data.X_l.shape[0]
    pred = method.predict(data.X_l)
    assert pred.shape[0] == data.X_l.shape[0]


def test_tsvm_numpy_additional_branches():
    data = make_numpy_dataset()
    with pytest.raises(InductiveValidationError):
        TSVMMethod(TSVMSpec()).fit(
            DummyDataset(X_l=data.X_l, y_l=data.y_l, X_u=None),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    empty = DummyDataset(
        X_l=np.zeros((0, data.X_l.shape[1]), dtype=np.float32),
        y_l=np.array([], dtype=np.int64),
        X_u=data.X_u,
    )
    with pytest.raises(InductiveValidationError):
        TSVMMethod(TSVMSpec()).fit(empty, device=DeviceSpec(device="cpu"), seed=0)

    empty_u = DummyDataset(
        X_l=data.X_l.astype(np.float32),
        y_l=data.y_l,
        X_u=np.zeros((0, data.X_l.shape[1]), dtype=np.float32),
    )
    TSVMMethod(TSVMSpec(max_iter=1)).fit(empty_u, device=DeviceSpec(device="cpu"), seed=0)

    TSVMMethod(TSVMSpec(max_iter=1, balance=False, C_l=10.0)).fit(
        data, device=DeviceSpec(device="cpu"), seed=0
    )


def test_tsvm_torch_and_errors():
    data = InductiveDataset(
        X_l=torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
        y_l=torch.tensor([0, 1], dtype=torch.int64),
        X_u=torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
    )
    spec = TSVMSpec(max_iter=1, epochs_per_iter=1, batch_size=2, balance=False)
    method = TSVMMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])

    with pytest.raises(InductiveValidationError):
        TSVMMethod(TSVMSpec()).fit(
            InductiveDataset(
                X_l=torch.ones((2, 2), dtype=torch.int64),
                y_l=torch.tensor([0, 1], dtype=torch.int64),
                X_u=torch.ones((2, 2), dtype=torch.int64),
            ),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    empty_u = InductiveDataset(
        X_l=torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
        y_l=torch.tensor([0, 1], dtype=torch.int64),
        X_u=torch.zeros((0, 2)),
    )
    TSVMMethod(TSVMSpec(max_iter=1)).fit(empty_u, device=DeviceSpec(device="cpu"), seed=0)

    with pytest.raises(InductiveValidationError):
        TSVMMethod(TSVMSpec()).fit(
            DummyDataset(X_l=data.X_l, y_l=data.y_l, X_u=None),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    with pytest.raises(InductiveValidationError):
        TSVMMethod(TSVMSpec()).fit(
            InductiveDataset(
                X_l=torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
                y_l=torch.tensor([0, 1, 2], dtype=torch.int64),
                X_u=torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
            ),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    method._backend = "torch"
    with pytest.raises(InductiveValidationError):
        method.predict_proba(np.zeros((2, 2), dtype=np.float32))

    method._backend = "numpy"
    with pytest.raises(InductiveValidationError):
        method.predict_proba([[1.0, 2.0]])
    method._classes = None
    with pytest.raises(RuntimeError):
        method.predict_proba(data.X_l.cpu().numpy())


def test_tsvm_numpy_empty_xl(monkeypatch):
    data = make_numpy_dataset()
    empty = DummyDataset(
        X_l=np.zeros((0, data.X_l.shape[1]), dtype=np.float32),
        y_l=np.array([0], dtype=np.int64),
        X_u=data.X_u,
    )
    monkeypatch.setattr(
        "modssc.inductive.methods.tsvm.ensure_numpy_data",
        lambda data: data,
    )
    with pytest.raises(InductiveValidationError):
        TSVMMethod(TSVMSpec()).fit(empty, device=DeviceSpec(device="cpu"), seed=0)


def test_tsvm_numpy_balance_and_rep_false(monkeypatch):
    data = make_numpy_dataset()

    def _decision(self, X):
        return np.zeros((X.shape[0],), dtype=np.float32)

    monkeypatch.setattr("modssc.inductive.methods.tsvm._LinearSVM.decision_function", _decision)
    spec = TSVMSpec(max_iter=1, epochs_per_iter=1, batch_size=1, balance=True, C_l=1e-6)
    TSVMMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_tsvm_numpy_balance_false_branch(monkeypatch):
    data = make_numpy_dataset()

    def _decision(self, X):
        scores = np.ones((X.shape[0],), dtype=np.float32)
        scores[::2] = -1.0
        return scores

    monkeypatch.setattr("modssc.inductive.methods.tsvm._LinearSVM.decision_function", _decision)
    spec = TSVMSpec(max_iter=1, epochs_per_iter=1, batch_size=1, balance=True)
    TSVMMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_tsvm_torch_empty_xl(monkeypatch):
    data = make_torch_dataset()
    empty = InductiveDataset(
        X_l=torch.zeros((0, data.X_l.shape[1])),
        y_l=torch.zeros((0,), dtype=torch.int64),
        X_u=data.X_u,
    )
    monkeypatch.setattr(
        "modssc.inductive.methods.tsvm.ensure_1d_labels_torch",
        lambda y, name="y_l": y,
    )
    with pytest.raises(InductiveValidationError):
        TSVMMethod(TSVMSpec()).fit(empty, device=DeviceSpec(device="cpu"), seed=0)


def test_tsvm_torch_fit_no_active_and_balance(monkeypatch):
    import torch

    def _randn(*shape, **kwargs):
        return torch.ones(
            *shape, device=kwargs.get("device"), dtype=kwargs.get("dtype", torch.float32)
        )

    monkeypatch.setattr(torch, "randn", _randn)

    X_l = torch.tensor([[-100.0, -100.0], [100.0, 100.0]])
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    X_u = torch.tensor([[100.0, 100.0], [100.0, 100.0]])
    data = InductiveDataset(X_l=X_l, y_l=y_l, X_u=X_u)
    spec = TSVMSpec(max_iter=1, epochs_per_iter=1, batch_size=1, balance=True, C_l=1e-6)
    method = TSVMMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    pred = method.predict(X_l)
    assert int(pred.shape[0]) == int(X_l.shape[0])

    with pytest.raises(InductiveValidationError):
        method.predict_proba(torch.ones((2, 2), dtype=torch.int64))


def test_tsvm_torch_balance_false_branch(monkeypatch):
    import torch

    def _randn(*shape, **kwargs):
        return torch.ones(
            *shape, device=kwargs.get("device"), dtype=kwargs.get("dtype", torch.float32)
        )

    monkeypatch.setattr(torch, "randn", _randn)

    X_l = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    X_u = torch.tensor([[100.0, 100.0], [-100.0, -100.0]])
    data = InductiveDataset(X_l=X_l, y_l=y_l, X_u=X_u)
    spec = TSVMSpec(max_iter=1, epochs_per_iter=1, batch_size=1, balance=True, C_l=1.0)
    TSVMMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_tsvm_predict_proba_not_fitted_and_mismatch():
    with pytest.raises(RuntimeError):
        TSVMMethod(TSVMSpec()).predict_proba(np.zeros((2, 2), dtype=np.float32))

    data = make_torch_dataset()
    method = TSVMMethod(TSVMSpec(max_iter=1, epochs_per_iter=1, batch_size=1, balance=False))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    method._backend = ""
    with pytest.raises(InductiveValidationError):
        method.predict_proba(data.X_l)


def test_s4vm_numpy_torch_and_errors():
    data = make_numpy_dataset()
    spec = S4VMSpec(k_candidates=2, flip_rate=0.6)
    method = S4VMMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    pred = method._clf.predict(data.X_l)
    assert accuracy_score(data.y_l, pred) >= 0.0

    spec2 = S4VMSpec(k_candidates=1, flip_rate=0.01)
    method2 = S4VMMethod(spec2)
    method2.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    proba = method2.predict_proba(data.X_l)
    assert proba.shape[0] == data.X_l.shape[0]

    data_t = make_torch_dataset()
    method_t = S4VMMethod(S4VMSpec(k_candidates=1, flip_rate=0.6, classifier_backend="torch"))
    method_t.fit(data_t, device=DeviceSpec(device="cpu"), seed=0)
    proba_t = method_t.predict_proba(data_t.X_l)
    assert int(proba_t.shape[0]) == int(data_t.X_l.shape[0])
    pred_t = method_t.predict(data_t.X_l)
    assert int(pred_t.shape[0]) == int(data_t.X_l.shape[0])

    method_t2 = S4VMMethod(S4VMSpec(k_candidates=1, flip_rate=0.01, classifier_backend="torch"))
    method_t2.fit(data_t, device=DeviceSpec(device="cpu"), seed=0)

    with pytest.raises(InductiveValidationError):
        S4VMMethod(S4VMSpec()).fit(
            DummyDataset(X_l=data.X_l, y_l=data.y_l, X_u=None),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    data_multi = DummyDataset(X_l=data.X_l, y_l=np.array([0, 1, 2, 2]), X_u=data.X_u)
    with pytest.raises(InductiveValidationError):
        S4VMMethod(S4VMSpec()).fit(data_multi, device=DeviceSpec(device="cpu"), seed=0)

    with pytest.raises(InductiveValidationError):
        S4VMMethod(S4VMSpec(classifier_backend="torch")).fit(
            DummyDataset(X_l=data_t.X_l, y_l=data_t.y_l, X_u=None),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    bad_t = DummyDataset(
        X_l=data_t.X_l,
        y_l=torch.tensor([0, 1, 2, 2], dtype=torch.int64),
        X_u=data_t.X_u,
    )
    with pytest.raises(InductiveValidationError):
        S4VMMethod(S4VMSpec(classifier_backend="torch")).fit(
            bad_t, device=DeviceSpec(device="cpu"), seed=0
        )

    with pytest.raises(RuntimeError):
        S4VMMethod(S4VMSpec()).predict(data.X_l)
    with pytest.raises(RuntimeError):
        S4VMMethod(S4VMSpec()).predict_proba(data.X_l)

    method_t2._backend = ""
    with pytest.raises(InductiveValidationError):
        method_t2.predict_proba(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(InductiveValidationError):
        method_t2.predict(np.zeros((2, 2), dtype=np.float32))
