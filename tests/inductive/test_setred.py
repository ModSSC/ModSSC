from __future__ import annotations

import numpy as np
import pytest
import torch

import modssc.inductive.methods.setred as setred
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.setred import (
    SetredMethod,
    SetredSpec,
    _allocate_per_class,
    _build_graph_numpy,
    _build_knn_graph_numpy,
    _build_rng_graph_numpy,
    _edge_weights_numpy,
    _filter_setred_numpy,
    _pairwise_distances_numpy,
    _select_candidates_by_class,
)
from modssc.inductive.types import DeviceSpec

from .conftest import DummyDataset, make_numpy_dataset, make_torch_dataset


def test_setred_helpers_graph_and_selection():
    with pytest.raises(InductiveValidationError):
        _pairwise_distances_numpy(np.array([1.0, 2.0], dtype=np.float32))

    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    neighbors, dist = _build_rng_graph_numpy(X)
    assert 1 in neighbors[0]
    assert 2 not in neighbors[0]
    assert 0 in neighbors[1]
    assert 2 in neighbors[1]
    assert dist.shape == (3, 3)

    knn_neighbors, _ = _build_knn_graph_numpy(X, k=10)
    assert knn_neighbors[0].size >= 1

    _build_graph_numpy(X, graph_type="rng", knn_k=1)
    _build_graph_numpy(X, graph_type="knn", knn_k=1)
    with pytest.raises(InductiveValidationError):
        _build_graph_numpy(X, graph_type="bad", knn_k=1)

    dist2 = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32)
    assert np.allclose(
        _edge_weights_numpy(dist2, mode="uniform", eps=1e-12),
        np.ones_like(dist2),
    )
    assert np.allclose(_edge_weights_numpy(dist2, mode="distance", eps=1e-12), dist2)
    inv = _edge_weights_numpy(dist2, mode="inverse_distance", eps=0.5)
    assert inv[0, 1] == pytest.approx(1.0 / 2.5)
    with pytest.raises(InductiveValidationError):
        _edge_weights_numpy(dist2, mode="bad", eps=1e-12)

    assert _allocate_per_class(0, np.array([1, 2])).sum() == 0
    assert _allocate_per_class(2, np.array([0, 0])).sum() == 0
    alloc = _allocate_per_class(3, np.array([1, 1]))
    assert alloc.sum() == 3
    alloc_more = _allocate_per_class(5, np.array([1, 1, 1]))
    assert alloc_more.sum() == 5

    scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
    pred = np.array([0, 0, 1])
    labels = np.array([0, 1])
    counts = np.array([2, 2])

    assert (
        _select_candidates_by_class(scores, pred, labels, counts, max_new=0, threshold=None).size
        == 0
    )
    one = _select_candidates_by_class(scores, pred, labels, counts, max_new=1, threshold=None)
    assert one.size == 1
    empty = _select_candidates_by_class(scores, pred, labels, counts, max_new=2, threshold=0.95)
    assert empty.size == 0

    pred_single = np.array([0, 0, 0])
    _select_candidates_by_class(scores, pred_single, labels, counts, max_new=2, threshold=None)


def test_setred_allocate_per_class_empty_order(monkeypatch):
    monkeypatch.setattr(setred.np, "argsort", lambda _arr: np.array([], dtype=np.int64))
    alloc = _allocate_per_class(3, np.array([1, 1], dtype=np.int64))
    assert alloc.sum() == 2


def test_setred_filter_branches():
    X_all = np.array([[0.0, 0.0]], dtype=np.float32)
    y_all = np.array([0], dtype=np.int64)
    keep = _filter_setred_numpy(
        X_all,
        y_all,
        np.array([0], dtype=np.int64),
        class_probs={0: 0.5},
        theta=0.5,
        graph_type="knn",
        knn_k=1,
        edge_weight="uniform",
        eps=1e-12,
    )
    assert keep.tolist() == [True]

    X_all = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    y_all = np.array([0, 1], dtype=np.int64)
    idx_l0 = np.array([1], dtype=np.int64)
    keep_sigma0 = _filter_setred_numpy(
        X_all,
        y_all,
        idx_l0,
        class_probs={1: 0.0},
        theta=0.5,
        graph_type="knn",
        knn_k=1,
        edge_weight="uniform",
        eps=1e-12,
    )
    assert keep_sigma0.tolist() == [True]

    keep_sigma = _filter_setred_numpy(
        X_all,
        y_all,
        idx_l0,
        class_probs={1: 0.5},
        theta=0.5,
        graph_type="knn",
        knn_k=1,
        edge_weight="uniform",
        eps=1e-12,
    )
    assert keep_sigma.tolist() == [False]


def test_setred_method_numpy_validation_and_predict():
    data = make_numpy_dataset()
    with pytest.raises(InductiveValidationError):
        SetredMethod(SetredSpec(theta=0.0)).fit(data, device=DeviceSpec(device="cpu"), seed=0)
    with pytest.raises(InductiveValidationError):
        SetredMethod(SetredSpec(min_new_labels=-1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        SetredMethod(SetredSpec(knn_k=0)).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    data_none = DummyDataset(X_l=data.X_l, y_l=data.y_l, X_u=None)
    method = SetredMethod(SetredSpec(max_iter=1))
    method.fit(data_none, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    assert proba.shape[0] == data.X_l.shape[0]
    pred = method.predict(data.X_l)
    assert pred.shape[0] == data.X_l.shape[0]

    method._backend = ""
    with pytest.raises(InductiveValidationError):
        method.predict_proba(torch.tensor([[0.0, 1.0]]))
    with pytest.raises(InductiveValidationError):
        method.predict(torch.tensor([[0.0, 1.0]]))

    method2 = SetredMethod(SetredSpec())
    with pytest.raises(RuntimeError):
        method2.predict_proba(data.X_l)
    with pytest.raises(RuntimeError):
        method2.predict(data.X_l)


def test_setred_method_numpy_empty_labeled(monkeypatch):
    data = make_numpy_dataset()

    class _Dummy:
        X_l = np.zeros((0, 2), dtype=np.float32)
        y_l = np.array([0], dtype=np.int64)
        X_u = data.X_u

    monkeypatch.setattr(setred, "ensure_numpy_data", lambda _data: _Dummy())
    with pytest.raises(InductiveValidationError):
        SetredMethod(SetredSpec()).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_setred_method_numpy_loop_branches(monkeypatch):
    data = make_numpy_dataset()

    method = SetredMethod(SetredSpec(max_iter=1, pool_size=0))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    def _select_one(_scores, *_args, **_kwargs):
        return np.array([0], dtype=np.int64)

    def _select_all(scores, *_args, **_kwargs):
        return np.arange(scores.shape[0], dtype=np.int64)

    def _keep_all(_x_all, _y_all, idx_l0, *_args, **_kwargs):
        return np.ones((int(idx_l0.shape[0]),), dtype=bool)

    def _keep_none(_x_all, _y_all, idx_l0, *_args, **_kwargs):
        return np.zeros((int(idx_l0.shape[0]),), dtype=bool)

    monkeypatch.setattr(setred, "_select_candidates_by_class", _select_one)
    monkeypatch.setattr(setred, "_filter_setred_numpy", _keep_all)
    method = SetredMethod(SetredSpec(max_iter=1, pool_size=1, max_new_labels=1, min_new_labels=1))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    monkeypatch.setattr(setred, "_select_candidates_by_class", _select_all)
    method = SetredMethod(SetredSpec(max_iter=2, pool_size=None, min_new_labels=1))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    monkeypatch.setattr(
        setred,
        "_select_candidates_by_class",
        lambda *_args, **_kwargs: np.empty((0,), dtype=np.int64),
    )
    method = SetredMethod(SetredSpec(max_iter=1, pool_size=None))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    monkeypatch.setattr(setred, "_select_candidates_by_class", _select_one)
    monkeypatch.setattr(setred, "_filter_setred_numpy", _keep_none)
    method = SetredMethod(SetredSpec(max_iter=1, pool_size=1, min_new_labels=1))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_setred_method_torch_paths(monkeypatch):
    data = make_torch_dataset()

    data_none = DummyDataset(X_l=data.X_l, y_l=data.y_l, X_u=None)
    method = SetredMethod(SetredSpec(max_iter=1, classifier_backend="torch"))
    method.fit(data_none, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])
    pred = method.predict(data.X_l)
    assert int(pred.shape[0]) == int(data.X_l.shape[0])

    method = SetredMethod(SetredSpec(max_iter=1, pool_size=0, classifier_backend="torch"))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    def _select_one(_scores, *_args, **_kwargs):
        return np.array([0], dtype=np.int64)

    def _select_all(scores, *_args, **_kwargs):
        return np.arange(scores.shape[0], dtype=np.int64)

    def _keep_all(_x_all, _y_all, idx_l0, *_args, **_kwargs):
        return np.ones((int(idx_l0.shape[0]),), dtype=bool)

    def _keep_none(_x_all, _y_all, idx_l0, *_args, **_kwargs):
        return np.zeros((int(idx_l0.shape[0]),), dtype=bool)

    monkeypatch.setattr(setred, "_select_candidates_by_class", _select_one)
    monkeypatch.setattr(setred, "_filter_setred_numpy", _keep_all)
    method = SetredMethod(
        SetredSpec(max_iter=1, pool_size=1, max_new_labels=1, classifier_backend="torch")
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    monkeypatch.setattr(setred, "_select_candidates_by_class", _select_all)
    method = SetredMethod(SetredSpec(max_iter=2, pool_size=None, classifier_backend="torch"))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    monkeypatch.setattr(
        setred,
        "_select_candidates_by_class",
        lambda *_args, **_kwargs: np.empty((0,), dtype=np.int64),
    )
    method = SetredMethod(SetredSpec(max_iter=1, pool_size=None, classifier_backend="torch"))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    monkeypatch.setattr(setred, "_select_candidates_by_class", _select_one)
    monkeypatch.setattr(setred, "_filter_setred_numpy", _keep_none)
    method = SetredMethod(
        SetredSpec(max_iter=1, pool_size=1, min_new_labels=1, classifier_backend="torch")
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_setred_method_torch_empty_accepts_no_indices(monkeypatch):
    data = make_torch_dataset()

    def _select_one(_scores, *_args, **_kwargs):
        return np.array([0], dtype=np.int64)

    def _keep_none(_x_all, _y_all, idx_l0, *_args, **_kwargs):
        return np.zeros((int(idx_l0.shape[0]),), dtype=bool)

    monkeypatch.setattr(setred, "_select_candidates_by_class", _select_one)
    monkeypatch.setattr(setred, "_filter_setred_numpy", _keep_none)
    method = SetredMethod(
        SetredSpec(max_iter=1, pool_size=1, min_new_labels=0, classifier_backend="torch")
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_setred_method_torch_empty_labeled(monkeypatch):
    data = make_torch_dataset()

    class _Dummy:
        X_l = data.X_l[:0]
        y_l = data.y_l[:1]
        X_u = data.X_u

    monkeypatch.setattr(setred, "ensure_torch_data", lambda _data, device: _Dummy())
    with pytest.raises(InductiveValidationError):
        SetredMethod(SetredSpec(classifier_backend="torch")).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
