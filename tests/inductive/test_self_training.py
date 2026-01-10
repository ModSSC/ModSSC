from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods import self_training as self_training_mod
from modssc.inductive.methods.self_training import (
    SelfTrainingMethod,
    SelfTrainingSpec,
    _normalize_group_ids_numpy,
    _normalize_group_ids_torch,
    _resolve_group_ids,
    _select_candidates_numpy,
)
from modssc.inductive.types import DeviceSpec

from .conftest import DummyDataset, make_numpy_dataset, make_torch_dataset

torch = pytest.importorskip("torch")


def test_group_id_normalization_numpy_and_torch():
    arr = np.array([1, 2], dtype=np.int64)
    out = _normalize_group_ids_numpy(arr, n_expected=2, name="group")
    assert out.shape == (2,)
    with pytest.raises(InductiveValidationError):
        _normalize_group_ids_numpy(np.array([[1, 2]]), n_expected=2, name="group")
    with pytest.raises(InductiveValidationError):
        _normalize_group_ids_numpy(np.array([1, 2, 3]), n_expected=2, name="group")

    t = torch.tensor([1, 2], dtype=torch.int64)
    out_t = _normalize_group_ids_torch(t, n_expected=2, name="group")
    assert out_t is t
    with pytest.raises(InductiveValidationError):
        _normalize_group_ids_torch([1, 2], n_expected=2, name="group")
    with pytest.raises(InductiveValidationError):
        _normalize_group_ids_torch(torch.tensor([[1, 2]]), n_expected=2, name="group")
    with pytest.raises(InductiveValidationError):
        _normalize_group_ids_torch(torch.tensor([1, 2, 3]), n_expected=2, name="group")
    with pytest.raises(InductiveValidationError):
        _normalize_group_ids_torch(torch.tensor([1.0, 2.0]), n_expected=2, name="group")


def test_resolve_group_ids_paths():
    assert (
        _resolve_group_ids(
            None,
            group_key=None,
            n_expected=2,
            backend="numpy",
            name="group",
            key_candidates=("group_u",),
        )
        is None
    )
    with pytest.raises(InductiveValidationError):
        _resolve_group_ids(
            ["bad"],
            group_key=None,
            n_expected=2,
            backend="numpy",
            name="group",
            key_candidates=("group_u",),
        )
    with pytest.raises(InductiveValidationError):
        _resolve_group_ids(
            {"other": np.array([1, 2])},
            group_key="group_u",
            n_expected=2,
            backend="numpy",
            name="group",
            key_candidates=("group_u",),
        )

    meta = {"group_u": np.array([1, 2])}
    out = _resolve_group_ids(
        meta,
        group_key="group_u",
        n_expected=2,
        backend="numpy",
        name="group",
        key_candidates=("group_u",),
    )
    assert np.array_equal(out, meta["group_u"])

    meta2 = {"group_u": np.array([[1, 2]]), "groups": np.array([0, 1])}
    out2 = _resolve_group_ids(
        meta2,
        group_key=None,
        n_expected=2,
        backend="numpy",
        name="group",
        key_candidates=("group_u", "groups"),
    )
    assert np.array_equal(out2, meta2["groups"])

    meta3 = {"group_u": np.array([[1, 2]])}
    out3 = _resolve_group_ids(
        meta3,
        group_key=None,
        n_expected=2,
        backend="numpy",
        name="group",
        key_candidates=("group_u",),
    )
    assert out3 is None

    meta_t = {"group_u": torch.tensor([1, 2], dtype=torch.int64)}
    out_t = _resolve_group_ids(
        meta_t,
        group_key="group_u",
        n_expected=2,
        backend="torch",
        name="group",
        key_candidates=("group_u",),
    )
    assert out_t is meta_t["group_u"]


def test_select_candidates_numpy_group_add_and_truncate():
    scores = np.array(
        [
            [0.2, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.6, 0.4],
            [0.95, 0.05],
            [0.7, 0.3],
        ],
        dtype=np.float32,
    )
    pred = scores.argmax(axis=1)
    group_u = np.array([1, 2, 2, 3, 4, 4])
    group_l = np.array([4])
    y_l = np.array([0], dtype=np.int64)
    idx, labels, direct_count, group_added = _select_candidates_numpy(
        scores,
        pred,
        threshold=0.8,
        max_new=2,
        use_group=True,
        group_u=group_u,
        group_l=group_l,
        y_l=y_l,
        group_min_count=2,
        group_min_fraction=0.6,
        group_conf_threshold=0.5,
    )
    assert direct_count == 3
    assert group_added == 1
    assert idx.size == 2
    assert labels.size == 2


def test_select_candidates_numpy_conf_thresholds_and_empty():
    scores = np.array(
        [
            [0.6, 0.4],
            [0.6, 0.4],
            [0.95, 0.05],
            [0.95, 0.05],
        ],
        dtype=np.float32,
    )
    pred = scores.argmax(axis=1)
    group_u = np.array([1, 1, 2, 2])
    idx, labels, direct_count, group_added = _select_candidates_numpy(
        scores,
        pred,
        threshold=0.5,
        max_new=None,
        use_group=True,
        group_u=group_u,
        group_l=None,
        y_l=None,
        group_min_count=2,
        group_min_fraction=0.5,
        group_conf_threshold=0.7,
    )
    assert direct_count == 4
    assert group_added == 0
    assert idx.size == 4
    assert labels.size == 4

    scores2 = np.array([[0.2, 0.8], [0.3, 0.7]], dtype=np.float32)
    pred2 = scores2.argmax(axis=1)
    idx2, labels2, direct_count2, group_added2 = _select_candidates_numpy(
        scores2,
        pred2,
        threshold=None,
        max_new=None,
        use_group=True,
        group_u=np.array([1, 1]),
        group_l=None,
        y_l=None,
        group_min_count=1,
        group_min_fraction=0.5,
        group_conf_threshold=None,
    )
    assert direct_count2 == 2
    assert group_added2 == 0
    assert idx2.size == 2
    assert labels2.size == 2

    idx3, labels3, _, _ = _select_candidates_numpy(
        scores2,
        pred2,
        threshold=1.1,
        max_new=None,
        use_group=False,
        group_u=None,
        group_l=None,
        y_l=None,
        group_min_count=1,
        group_min_fraction=0.5,
        group_conf_threshold=None,
    )
    assert idx3.size == 0
    assert labels3.size == 0


def test_select_candidates_numpy_empty_group_idx(monkeypatch):
    scores = np.array([[0.9, 0.1]], dtype=np.float32)
    pred = scores.argmax(axis=1)
    group_u = np.array([1])
    orig_unique = self_training_mod.np.unique

    def fake_unique(arr, *args, **kwargs):
        if arr is group_u:
            return np.array([999], dtype=arr.dtype)
        return orig_unique(arr, *args, **kwargs)

    monkeypatch.setattr(self_training_mod.np, "unique", fake_unique)
    idx, labels, direct_count, group_added = _select_candidates_numpy(
        scores,
        pred,
        threshold=0.5,
        max_new=None,
        use_group=True,
        group_u=group_u,
        group_l=None,
        y_l=None,
        group_min_count=1,
        group_min_fraction=0.5,
        group_conf_threshold=0.5,
    )
    assert direct_count == 1
    assert group_added == 0
    assert idx.size == 1
    assert labels.size == 1


def test_self_training_errors_and_predict_mismatch():
    data = make_numpy_dataset()
    with pytest.raises(InductiveValidationError):
        SelfTrainingMethod(SelfTrainingSpec(group_min_count=0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        SelfTrainingMethod(SelfTrainingSpec(group_min_fraction=1.1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        SelfTrainingMethod(SelfTrainingSpec(use_group_propagation=True)).fit(
            DummyDataset(X_l=data.X_l, y_l=data.y_l, X_u=data.X_u, meta={}),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )

    method = SelfTrainingMethod()
    with pytest.raises(RuntimeError):
        method.predict_proba(np.zeros((1, 2)))
    with pytest.raises(RuntimeError):
        method.predict(np.zeros((1, 2)))

    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    method._backend = ""
    with pytest.raises(InductiveValidationError):
        method.predict_proba(torch.tensor([[0.0, 1.0]]))
    with pytest.raises(InductiveValidationError):
        method.predict(torch.tensor([[0.0, 1.0]]))


def test_self_training_numpy_group_and_breaks():
    data = make_numpy_dataset(n_l=4, n_u=2)
    ds = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        meta={"group_u": np.array([0, 1], dtype=np.int64)},
    )
    spec = SelfTrainingSpec(max_iter=2, confidence_threshold=0.0, min_new_labels=1)
    method = SelfTrainingMethod(spec)
    method.fit(ds, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    assert proba.shape[0] == data.X_l.shape[0]


def test_self_training_numpy_min_new_labels_break():
    data = make_numpy_dataset(n_l=4, n_u=1)
    spec = SelfTrainingSpec(
        max_iter=1,
        confidence_threshold=0.0,
        min_new_labels=2,
        use_group_propagation=False,
    )
    method = SelfTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_numpy_no_unlabeled_and_skip_loop():
    data = make_numpy_dataset(n_l=4, n_u=0)
    method = SelfTrainingMethod(SelfTrainingSpec())
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    data_loop = make_numpy_dataset(n_l=4, n_u=2)
    spec = SelfTrainingSpec(max_iter=0, use_group_propagation=False)
    method2 = SelfTrainingMethod(spec)
    method2.fit(data_loop, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_numpy_no_group_update_branch():
    data = make_numpy_dataset(n_l=4, n_u=1)
    spec = SelfTrainingSpec(
        max_iter=1,
        confidence_threshold=0.0,
        min_new_labels=1,
        use_group_propagation=False,
    )
    method = SelfTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    pred = method.predict(data.X_l)
    assert pred.shape[0] == data.X_l.shape[0]


def test_self_training_numpy_empty_xl_error(monkeypatch):
    def fake_ensure_numpy_data(_data):
        return SimpleNamespace(
            X_l=np.empty((0, 2), dtype=np.float32),
            y_l=np.array([0], dtype=np.int64),
            X_u=np.array([[0.1, 0.2]], dtype=np.float32),
            meta=None,
        )

    monkeypatch.setattr(self_training_mod, "ensure_numpy_data", fake_ensure_numpy_data)
    data = DummyDataset(
        X_l=np.array([[0.0, 1.0]], dtype=np.float32),
        y_l=np.array([0], dtype=np.int64),
        X_u=np.array([[0.1, 0.2]], dtype=np.float32),
    )
    method = SelfTrainingMethod()
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_torch_early_return_and_predict_proba():
    X_l = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    ds = DummyDataset(X_l=X_l, y_l=y_l, X_u=None)
    method = SelfTrainingMethod(SelfTrainingSpec(classifier_backend="torch"))
    method.fit(ds, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(X_l)
    assert int(proba.shape[0]) == int(X_l.shape[0])


def test_self_training_torch_group_flow_and_breaks():
    data = make_torch_dataset(n_l=4, n_u=2)
    ds = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        meta={"group_u": torch.tensor([0, 1], dtype=torch.int64)},
    )
    spec = SelfTrainingSpec(
        classifier_backend="torch",
        max_iter=2,
        confidence_threshold=0.0,
        min_new_labels=1,
    )
    method = SelfTrainingMethod(spec)
    method.fit(ds, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_torch_min_new_labels_break():
    data = make_torch_dataset(n_l=4, n_u=1)
    spec = SelfTrainingSpec(
        classifier_backend="torch",
        max_iter=1,
        confidence_threshold=0.0,
        min_new_labels=2,
    )
    method = SelfTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_torch_group_missing_error():
    data = make_torch_dataset(n_l=4, n_u=1)
    spec = SelfTrainingSpec(classifier_backend="torch", use_group_propagation=True)
    method = SelfTrainingMethod(spec)
    with pytest.raises(InductiveValidationError):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_torch_skip_loop_and_no_group_update():
    data = make_torch_dataset(n_l=4, n_u=2)
    spec = SelfTrainingSpec(
        classifier_backend="torch",
        max_iter=0,
        use_group_propagation=False,
    )
    method = SelfTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_torch_no_group_update_branch():
    data = make_torch_dataset(n_l=4, n_u=1)
    spec = SelfTrainingSpec(
        classifier_backend="torch",
        max_iter=1,
        confidence_threshold=0.0,
        min_new_labels=1,
        use_group_propagation=False,
    )
    method = SelfTrainingMethod(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_self_training_torch_empty_xl_error(monkeypatch):
    def fake_ensure_torch_data(_data, *, device):
        return SimpleNamespace(
            X_l=torch.zeros((0, 2)),
            y_l=torch.tensor([0], dtype=torch.int64),
            X_u=torch.tensor([[0.1, 0.2]]),
            meta=None,
        )

    monkeypatch.setattr(self_training_mod, "ensure_torch_data", fake_ensure_torch_data)
    data = DummyDataset(
        X_l=torch.tensor([[0.0, 1.0]]),
        y_l=torch.tensor([0], dtype=torch.int64),
        X_u=torch.tensor([[0.1, 0.2]]),
    )
    method = SelfTrainingMethod(SelfTrainingSpec(classifier_backend="torch"))
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
