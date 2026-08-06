from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import modssc.inductive.methods.democratic_co_learning as dcl
from bench.orchestrators.evaluation import _views_for_split
from bench.orchestrators.method_inductive import _should_auto_to_torch
from bench.schema import DeviceConfig, MethodConfig, ModelConfig
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
from modssc.inductive.types import DeviceSpec

from ..conftest import DummyDataset, make_numpy_dataset, make_torch_dataset


def test_should_auto_to_torch_respects_method_backend_contract() -> None:
    device = DeviceConfig(device="auto", dtype="float32")

    co_training_cfg = MethodConfig(
        kind="inductive",
        method_id="co_training",
        device=device,
        params={"classifier_backend": "sklearn"},
    )
    assert _should_auto_to_torch(co_training_cfg, requires_torch=False) is False

    pseudo_label_cfg = MethodConfig(
        kind="inductive",
        method_id="pseudo_label",
        device=device,
        params={"classifier_backend": "torch"},
    )
    assert _should_auto_to_torch(pseudo_label_cfg, requires_torch=False) is True

    softmatch_cfg = MethodConfig(
        kind="inductive",
        method_id="softmatch",
        device=device,
        model=ModelConfig(classifier_backend="torch"),
    )
    assert _should_auto_to_torch(softmatch_cfg, requires_torch=False) is True
    assert _should_auto_to_torch(co_training_cfg, requires_torch=True) is True


def test_views_for_split_keeps_numpy_views_for_numpy_backend() -> None:
    data, views = _make_views_numpy()
    views_result = SimpleNamespace(
        views={
            name: SimpleNamespace(train=SimpleNamespace(X=view["X_l"]), test=None)
            for name, view in views.items()
        }
    )

    class _Sampling:
        indices = {"val": np.array([0, 1], dtype=np.int64)}
        refs = {"val": "train"}

    payload = _views_for_split(
        views=views_result,
        split="val",
        sampling=_Sampling(),
        backend_ref=data.X_l,
        strict=False,
    )

    assert isinstance(payload["v1"]["X"], np.ndarray)
    assert isinstance(payload["v2"]["X"], np.ndarray)


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

    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    method._backend = ""
    with pytest.raises(InductiveValidationError):
        method.predict_proba(torch.tensor([[0.0, 1.0]]))
    with pytest.raises(InductiveValidationError):
        method.predict(torch.tensor([[0.0, 1.0]]))

    with pytest.raises(RuntimeError):
        PseudoLabelMethod(PseudoLabelSpec()).predict(data.X_l)


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


def test_co_training_torch_device_mismatch():
    data, views = _make_views_torch()
    y_l = torch.zeros_like(data.y_l, device="meta")
    ds = DummyDataset(X_l=data.X_l, y_l=y_l, views=views)
    method = CoTrainingMethod(CoTrainingSpec(classifier_backend="torch"))
    with pytest.raises(InductiveValidationError, match="same device"):
        method.fit(ds, device=DeviceSpec(device="cpu"), seed=0)


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


def test_democratic_co_learning_numpy_fit_predict():
    data = make_numpy_dataset()
    spec = dcl.DemocraticCoLearningSpec(max_iter=1, n_learners=3)
    method = dcl.DemocraticCoLearningMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    proba = method.predict_proba(data.X_l)
    assert proba.shape[0] == data.X_l.shape[0]
    pred = method.predict(data.X_l)
    assert pred.shape[0] == data.X_l.shape[0]

    data_none = DummyDataset(X_l=data.X_l, y_l=data.y_l, X_u=None)
    method2 = dcl.DemocraticCoLearningMethod(spec)
    method2.fit(data_none, device=DeviceSpec(device="cpu"), seed=0)
    assert method2.n_iter_ == 0
    assert method2.changed_rounds_ == 0
    assert method2.converged_ is True
    assert method2.pseudo_labels_added_per_learner_ == (0, 0, 0)
    assert method2.diagnostics_ == {
        "n_iter": 0,
        "changed_rounds": 0,
        "converged": True,
        "pseudo_labels_added_per_learner": [0, 0, 0],
        "pseudo_labels_added_total": 0,
    }


def test_democratic_co_learning_torch_fit_predict():
    data = make_torch_dataset()
    spec = dcl.DemocraticCoLearningSpec(max_iter=1, n_learners=3, classifier_backend="torch")
    method = dcl.DemocraticCoLearningMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    proba = method.predict_proba(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])
    pred = method.predict(data.X_l)
    assert int(pred.shape[0]) == int(data.X_l.shape[0])


def test_classic_methods_share_bounded_torch_inference_with_prediction_parity():
    class _Classifier:
        batch_size = 2
        classes_ = np.array([0, 1], dtype=np.int64)
        classes_t_ = torch.tensor([0, 1], dtype=torch.int64)

        def __init__(self):
            self.predict_batches = []
            self.score_batches = []

        def predict(self, X):
            self.predict_batches.append(int(X.shape[0]))
            return (X[:, 0] % 2).to(dtype=torch.int64)

        def predict_scores(self, X):
            self.score_batches.append(int(X.shape[0]))
            labels = (X[:, 0] % 2).to(dtype=torch.int64)
            return torch.nn.functional.one_hot(labels, num_classes=2).to(dtype=torch.float32)

    X = torch.arange(10).reshape(5, 2)
    labels = (X[:, 0] % 2).to(dtype=torch.int64)
    expected_scores = torch.nn.functional.one_hot(labels, num_classes=2).to(dtype=torch.float32)

    democratic_classifiers = [_Classifier() for _ in range(3)]
    democratic = dcl.DemocraticCoLearningMethod()
    democratic._clfs = democratic_classifiers
    democratic._backend = "torch"
    democratic._weights = np.ones((3,), dtype=np.float64)
    democratic._classes = np.array([0, 1], dtype=np.int64)
    democratic._classes_t = torch.tensor([0, 1], dtype=torch.int64)
    assert torch.equal(democratic.predict_proba(X), expected_scores)
    assert all(clf.predict_batches == [5] for clf in democratic_classifiers)

    pseudo_classifier = _Classifier()
    pseudo = PseudoLabelMethod()
    pseudo._clf = pseudo_classifier
    pseudo._backend = "torch"
    assert torch.equal(pseudo.predict_proba(X), expected_scores)
    assert torch.equal(pseudo.predict(X), labels)
    assert pseudo_classifier.score_batches == [2, 2, 1]
    assert pseudo_classifier.predict_batches == [2, 2, 1]

    tri_classifiers = [_Classifier() for _ in range(3)]
    tri = TriTrainingMethod()
    tri._clfs = tri_classifiers
    tri._backend = "torch"
    tri.classes_ = np.array([0, 1], dtype=np.int64)
    assert torch.equal(tri.predict_proba(X), expected_scores)
    assert all(clf.score_batches == [2, 2, 1] for clf in tri_classifiers)


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


def test_s4vm_numpy_torch_and_errors():
    data = make_numpy_dataset()
    spec = S4VMSpec(k_candidates=2, flip_rate=0.6)
    method = S4VMMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    pred = method.predict(data.X_l)
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
        method_t2.predict(np.zeros((2, 2), dtype=np.float32))


def test_tri_training_alignment_mismatch_classes():
    # Setup 3 classifiers with different classes
    class _Clf:
        def __init__(self, classes, scores):
            self.classes_ = np.asarray(classes)
            self._scores = scores

        def predict_scores(self, X):
            return self._scores

    # Clf 1: classes [0, 1], scores shape (N, 2)
    # Clf 2: classes [0, 1, 2], scores shape (N, 3)
    # Clf 3: classes [0, 2], scores shape (N, 2)
    # Union classes: [0, 1, 2] -> 3 classes
    scores_2 = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float32)
    # Note: scores_3 columns match [0, 1, 2]
    scores_3 = np.array([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]], dtype=np.float32)

    clf1 = _Clf([0, 1], scores_2)
    clf2 = _Clf([0, 1, 2], scores_3)
    clf3 = _Clf([0, 2], scores_2)

    method = TriTrainingMethod(TriTrainingSpec())
    method._clfs = [clf1, clf2, clf3]
    method._backend = "numpy"

    X = np.zeros((2, 2), dtype=np.float32)
    # predict_proba should return (N, 3)
    proba = method.predict_proba(X)
    assert proba.shape == (2, 3)

    # Check values manually to be sure
    # Row 0:
    # clf1 (0,1): [0.8, 0.2] -> global [0.8, 0.2, 0.0] (class 2 missing)
    # clf2 (0,1,2): [0.7, 0.2, 0.1] -> global [0.7, 0.2, 0.1]
    # clf3 (0,2): [0.8, 0.2] -> global [0.8, 0.0, 0.2] (class 1 missing)
    # Avg: [(0.8+0.7+0.8)/3, (0.2+0.2+0)/3, (0.0+0.1+0.2)/3]
    #    = [2.3/3, 0.4/3, 0.3/3] = [0.766, 0.133, 0.1]
    # Sum = 1.0.

    expected_0 = np.array([2.3 / 3, 0.4 / 3, 0.3 / 3], dtype=np.float32)
    # Check if we normalize. The code does:
    # row_sum = avg.sum(axis=1, keepdims=True)
    # return avg / row_sum
    # Since scores were normalized (mostly), avg sum might be close to 1 but not exact if 0s were added.
    # 2.3+0.4+0.3 = 3.0. 3.0/3 = 1.0. So it is normalized.

    assert np.allclose(proba[0], expected_0, atol=1e-5)

    # Torch path
    class _ClfTorch:
        def __init__(self, classes, scores):
            self.classes_ = np.asarray(classes)
            self._scores = scores

        def predict_scores(self, X):
            return self._scores

    scores_2_t = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    scores_3_t = torch.tensor([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]])

    clf1_t = _ClfTorch([0, 1], scores_2_t)
    clf2_t = _ClfTorch([0, 1, 2], scores_3_t)
    clf3_t = _ClfTorch([0, 2], scores_2_t)

    method_t = TriTrainingMethod(TriTrainingSpec(classifier_backend="torch"))
    method_t._clfs = [clf1_t, clf2_t, clf3_t]
    method_t._backend = "torch"

    X_t = torch.zeros((2, 2))
    proba_t = method_t.predict_proba(X_t)
    assert proba_t.shape == (2, 3)
    assert torch.allclose(proba_t[0], torch.tensor(expected_0), atol=1e-5)


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
    spec = dcl.DemocraticCoLearningSpec(classifier_specs=(dcl.BaseClassifierSpec(),) * 2)
    with pytest.raises(InductiveValidationError, match="at least 3 learners"):
        dcl._resolve_classifier_specs(spec)
    spec2 = dcl.DemocraticCoLearningSpec(n_learners=2)
    with pytest.raises(InductiveValidationError, match="n_learners must be"):
        dcl._resolve_classifier_specs(spec2)


def test_democratic_classifier_spec_resolution_valid():
    specs = (
        dcl.BaseClassifierSpec(),
        dcl.BaseClassifierSpec(),
        dcl.BaseClassifierSpec(),
    )
    spec = dcl.DemocraticCoLearningSpec(classifier_specs=specs)
    out = dcl._resolve_classifier_specs(spec)
    assert len(out) == 3


def test_democratic_classifier_spec_resolution_from_config_mappings():
    spec = dcl.DemocraticCoLearningSpec(
        classifier_specs=(
            {"classifier_id": "gaussian_nb", "classifier_backend": "sklearn"},
            {
                "classifier_id": "decision_tree",
                "classifier_backend": "sklearn",
                "classifier_params": {"criterion": "entropy"},
            },
            {
                "classifier_id": "knn",
                "classifier_backend": "sklearn",
                "classifier_params": {"k": 3},
            },
        )
    )

    out = dcl._resolve_classifier_specs(spec)

    assert [item.classifier_id for item in out] == ["gaussian_nb", "decision_tree", "knn"]
    assert all(item.classifier_backend == "sklearn" for item in out)
    assert out[1].classifier_params == {"criterion": "entropy"}
    assert out[2].classifier_params == {"k": 3}


def test_democratic_classifier_spec_resolution_rejects_invalid_mapping():
    spec = dcl.DemocraticCoLearningSpec(
        classifier_specs=(
            {},
            {},
            {"classifier_id": "knn", "unknown": True},
        )
    )

    with pytest.raises(InductiveValidationError, match=r"classifier_specs\[2\] is invalid"):
        dcl._resolve_classifier_specs(spec)

    non_mapping_spec = dcl.DemocraticCoLearningSpec(
        classifier_specs=(dcl.BaseClassifierSpec(), dcl.BaseClassifierSpec(), object())
    )
    with pytest.raises(InductiveValidationError, match="instances or mappings"):
        dcl._resolve_classifier_specs(non_mapping_spec)


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


@pytest.mark.parametrize(
    "preds_idx,weights,n_classes,expected_idx,expected_ok",
    [
        pytest.param(
            [[1], [0], [0]],
            [0.99, 0.30, 0.30],
            2,
            [0],
            [False],
            id="vote-majority-loses-confidence-comparison",
        ),
        pytest.param(
            [[1], [1], [0]],
            [0.60, 0.60, 0.99],
            2,
            [1],
            [True],
            id="vote-majority-wins-confidence-comparison",
        ),
        pytest.param(
            [[1], [1], [0], [0]],
            [0.90, 0.90, 0.20, 0.20],
            2,
            [0],
            [False],
            id="two-two-tie-is-not-a-strict-majority",
        ),
        pytest.param(
            [[0, 0], [0, 0], [0, 0]],
            [1.0, 1.0, 1.0],
            1,
            [0, 0],
            [True, True],
            id="single-class",
        ),
    ],
)
def test_democratic_weighted_majority_paper_rule_numpy_torch_parity(
    preds_idx, weights, n_classes, expected_idx, expected_ok
):
    preds_numpy = np.asarray(preds_idx, dtype=np.int64)
    weights_numpy = np.asarray(weights, dtype=np.float64)
    idx_numpy, ok_numpy = dcl._weighted_majority_numpy(
        preds_numpy, weights_numpy, n_classes=n_classes
    )

    idx_torch, ok_torch = dcl._weighted_majority_torch(
        torch.as_tensor(preds_idx, dtype=torch.int64),
        torch.as_tensor(weights, dtype=torch.float64),
        n_classes=n_classes,
    )

    assert idx_numpy.tolist() == expected_idx
    assert ok_numpy.tolist() == expected_ok
    assert idx_torch.tolist() == expected_idx
    assert ok_torch.tolist() == expected_ok


def test_democratic_proposal_error_uses_each_majority_group() -> None:
    preds = np.array(
        [
            [0, 1],
            [0, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    majority = np.array([0, 1], dtype=np.int64)
    proposed = np.array([0, 1], dtype=np.int64)
    lower = np.array([0.9, 0.7, 0.5], dtype=np.float64)

    # Sample 0 is supported by learners 0/1: error 1 - .8 = .2.
    # Sample 1 is supported by learners 0/2: error 1 - .7 = .3.
    expected = 0.5
    assert dcl._proposal_error_numpy(
        preds_idx=preds,
        majority_idx=majority,
        proposed_idx=proposed,
        lower_bounds=lower,
    ) == pytest.approx(expected)
    assert dcl._proposal_error_torch(
        preds_idx=torch.as_tensor(preds),
        majority_idx=torch.as_tensor(majority),
        proposed_idx=torch.as_tensor(proposed),
        lower_bounds=torch.as_tensor(lower),
    ) == pytest.approx(expected)


def test_democratic_proposal_error_empty_set() -> None:
    preds = np.zeros((3, 1), dtype=np.int64)
    assert (
        dcl._proposal_error_numpy(
            preds_idx=preds,
            majority_idx=np.zeros(1, dtype=np.int64),
            proposed_idx=np.array([], dtype=np.int64),
            lower_bounds=np.ones(3),
        )
        == 0.0
    )
    assert (
        dcl._proposal_error_torch(
            preds_idx=torch.as_tensor(preds),
            majority_idx=torch.zeros(1, dtype=torch.int64),
            proposed_idx=torch.tensor([], dtype=torch.int64),
            lower_bounds=torch.ones(3),
        )
        == 0.0
    )


def test_democratic_combine_scores_numpy_eligibility():
    preds_idx = np.array([[0, 1], [1, 0]], dtype=np.int64)
    weights_low = np.array([0.1, 0.2], dtype=np.float64)
    scores = dcl._combine_scores_numpy(preds_idx, weights_low, n_classes=2, min_confidence=0.9)
    assert np.all(scores.sum(axis=1) > 0.0)
    np.testing.assert_array_equal(scores.argmax(axis=1), np.array([1, 0], dtype=np.int64))

    weights_mix = np.array([0.1, 0.9], dtype=np.float64)
    scores_mix = dcl._combine_scores_numpy(preds_idx, weights_mix, n_classes=2, min_confidence=0.5)
    np.testing.assert_allclose(scores_mix, np.array([[0.0, 0.675], [0.675, 0.0]]))


def test_democratic_combine_scores_torch_eligibility():
    preds_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([1.0, 0.2], dtype=torch.float32)
    scores = dcl._combine_scores_torch(preds_idx, weights, n_classes=2, min_confidence=0.1)
    torch.testing.assert_close(
        scores,
        torch.tensor([[0.75, 0.15], [0.15, 0.75]], dtype=torch.float32),
    )


def test_democratic_combine_scores_torch_all_ineligible():
    preds_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([0.0, 0.0], dtype=torch.float32)
    scores = dcl._combine_scores_torch(preds_idx, weights, n_classes=2, min_confidence=10.0)
    torch.testing.assert_close(scores, torch.full((2, 2), 0.75, dtype=torch.float32))
    torch.testing.assert_close(scores.argmax(dim=1), torch.zeros((2,), dtype=torch.int64))


def test_democratic_all_ineligible_falls_back_to_complete_ensemble() -> None:
    X_numpy = np.zeros((2, 1), dtype=np.float32)
    numpy_method = dcl.DemocraticCoLearningMethod()
    numpy_method._clfs = [
        _FixedPredictor([1, 0], [1, 0], classes=[0, 1]),
        _FixedPredictor([0, 1], [0, 1], classes=[0, 1]),
    ]
    numpy_method._backend = "numpy"
    numpy_method._weights = np.array([0.5, 0.2], dtype=np.float64)
    numpy_method._classes = np.array([0, 1], dtype=np.int64)

    numpy_probabilities = numpy_method.predict_proba(X_numpy)
    np.testing.assert_allclose(numpy_probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(numpy_method.predict(X_numpy), np.array([1, 0]))

    X_torch = torch.zeros((2, 1), dtype=torch.float32)
    torch_method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(classifier_backend="torch")
    )
    torch_method._clfs = [
        _TorchPredictor(
            torch.tensor([1, 0]),
            torch.tensor([1, 0]),
            classes_t=torch.tensor([0, 1]),
        ),
        _TorchPredictor(
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            classes_t=torch.tensor([0, 1]),
        ),
    ]
    torch_method._backend = "torch"
    torch_method._weights = np.array([0.5, 0.2], dtype=np.float64)
    torch_method._classes_t = torch.tensor([0, 1], dtype=torch.int64)

    torch_probabilities = torch_method.predict_proba(X_torch)
    torch.testing.assert_close(torch_probabilities.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(
        torch_method.predict(X_torch),
        torch.tensor([1, 0], dtype=torch.int64),
    )


def test_democratic_fit_requires_data():
    method = dcl.DemocraticCoLearningMethod()
    with pytest.raises(InductiveValidationError, match="data must not be None"):
        method.fit(None, device=DeviceSpec(device="cpu"), seed=0)


def test_democratic_fit_numpy_empty_labeled(monkeypatch):
    bad = DummyDataset(
        X_l=np.zeros((0, 2), dtype=np.float32),
        y_l=np.array([0], dtype=np.int64),
        X_u=None,
    )
    monkeypatch.setattr(dcl, "ensure_numpy_data", lambda _data: bad)
    method = dcl.DemocraticCoLearningMethod(dcl.DemocraticCoLearningSpec(n_learners=3))
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        method.fit(bad, device=DeviceSpec(device="cpu"), seed=0)


def test_democratic_fit_torch_empty_labeled(monkeypatch):
    bad = DummyDataset(
        X_l=torch.zeros((0, 2)),
        y_l=torch.tensor([0], dtype=torch.int64),
        X_u=torch.zeros((2, 2)),
    )
    monkeypatch.setattr(dcl, "ensure_torch_data", lambda _data, device: bad)
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(n_learners=3, classifier_backend="torch")
    )
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        method.fit(bad, device=DeviceSpec(device="cpu"), seed=0)


def test_democratic_fit_torch_without_unlabeled():
    data = DummyDataset(
        X_l=torch.tensor([[0.0], [1.0]]),
        y_l=torch.tensor([0, 1], dtype=torch.int64),
        X_u=None,
    )
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(max_iter=1, n_learners=3, classifier_backend="torch")
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
    method = dcl.DemocraticCoLearningMethod(dcl.DemocraticCoLearningSpec(max_iter=1, n_learners=3))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    assert method._backend == "numpy"
    assert method.n_iter_ == 1
    assert method.changed_rounds_ == 1
    assert method.converged_ is False
    assert method.pseudo_labels_added_per_learner_ == (0, 0, 3)
    np.testing.assert_array_equal(method._weights, np.ones(3, dtype=np.float64))
    np.testing.assert_allclose(
        method.predict_proba(X_u),
        np.tile(np.array([[10.0 / 19.0, 9.0 / 19.0]], dtype=np.float32), (3, 1)),
    )
    assert method._initial_clfs == []
    assert method.round_trace_ == []
    assert method.diagnostics_ == {
        "n_iter": 1,
        "changed_rounds": 1,
        "converged": False,
        "pseudo_labels_added_per_learner": [0, 0, 3],
        "pseudo_labels_added_total": 3,
    }


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
    method = dcl.DemocraticCoLearningMethod(dcl.DemocraticCoLearningSpec(max_iter=1, n_learners=3))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    assert method._backend == "numpy"
    assert method.n_iter_ == 0
    assert method.changed_rounds_ == 0
    assert method.converged_ is True
    assert method.pseudo_labels_added_per_learner_ == (0, 0, 0)
    assert method.diagnostics_ == {
        "n_iter": 0,
        "changed_rounds": 0,
        "converged": True,
        "pseudo_labels_added_per_learner": [0, 0, 0],
        "pseudo_labels_added_total": 0,
    }


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
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(max_iter=1, n_learners=3, classifier_backend="torch")
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    assert method._backend == "torch"
    assert method.n_iter_ == 1
    assert method.changed_rounds_ == 1
    assert method.converged_ is False
    assert method.pseudo_labels_added_per_learner_ == (0, 0, 3)


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
    method = dcl.DemocraticCoLearningMethod(
        dcl.DemocraticCoLearningSpec(max_iter=1, n_learners=3, classifier_backend="torch")
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    assert method._backend == "torch"
    assert method.n_iter_ == 0
    assert method.changed_rounds_ == 0
    assert method.converged_ is True
    assert method.pseudo_labels_added_per_learner_ == (0, 0, 0)


def test_democratic_predict_proba_error_paths():
    method = dcl.DemocraticCoLearningMethod()
    with pytest.raises(RuntimeError, match="not fitted"):
        method.predict_proba(np.zeros((1, 2), dtype=np.float32))

    method._clfs = [_FixedPredictor([0], [0])]
    method._weights = np.ones((1,), dtype=np.float64)
    method._backend = ""
    with pytest.raises(InductiveValidationError, match="backend mismatch"):
        method.predict_proba(torch.zeros((1, 2)))

    method2 = dcl.DemocraticCoLearningMethod()
    method2._clfs = [_FixedPredictor([0], [0])]
    with pytest.raises(RuntimeError, match="missing weights"):
        method2.predict_proba(np.zeros((1, 2), dtype=np.float32))

    method3 = dcl.DemocraticCoLearningMethod()
    method3._clfs = [_FixedPredictor([0], [0])]
    method3._weights = np.ones((1,), dtype=np.float64)
    method3._backend = "numpy"
    with pytest.raises(RuntimeError, match="missing classes"):
        method3.predict_proba(np.zeros((1, 2), dtype=np.float32))

    method4 = dcl.DemocraticCoLearningMethod()
    method4._clfs = [_TorchPredictor(torch.tensor([0]), torch.tensor([0]))]
    method4._weights = np.ones((1,), dtype=np.float64)
    method4._backend = "torch"
    with pytest.raises(RuntimeError, match="missing classes"):
        method4.predict_proba(torch.zeros((1, 2)))


def test_democratic_predict_returns_idx_when_classes_missing(monkeypatch):
    method = dcl.DemocraticCoLearningMethod()
    method._backend = "numpy"
    monkeypatch.setattr(
        method,
        "predict_proba",
        lambda _x: np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32),
    )
    pred = method.predict(np.zeros((2, 2), dtype=np.float32))
    assert pred.shape == (2,)

    method2 = dcl.DemocraticCoLearningMethod()
    method2._backend = "torch"
    monkeypatch.setattr(method2, "predict_proba", lambda _x: torch.tensor([[0.2, 0.8], [0.9, 0.1]]))
    pred2 = method2.predict(torch.zeros((2, 2)))
    assert int(pred2.shape[0]) == 2
