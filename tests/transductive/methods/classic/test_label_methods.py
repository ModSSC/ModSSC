from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

try:
    import torch
except Exception:
    torch = None

from modssc.transductive.methods.classic.dynamic_label_propagation import (
    DynamicLabelPropagationMethod,
    DynamicLabelPropagationSpec,
    dynamic_label_propagation,
    dynamic_label_propagation_numpy,
)
from modssc.transductive.methods.classic.dynamic_label_propagation import (
    _infer_num_classes as _dlp_infer_num_classes,
)
from modssc.transductive.methods.classic.dynamic_label_propagation import (
    _knn_matrix_numpy as _dlp_knn_numpy,
)
from modssc.transductive.methods.classic.label_propagation import (
    LabelPropagationMethod,
    LabelPropagationSpec,
    label_propagation,
    label_propagation_numpy,
    label_propagation_torch,
)
from modssc.transductive.methods.classic.label_spreading import (
    LabelSpreadingMethod,
    LabelSpreadingSpec,
    label_spreading,
    label_spreading_numpy,
    label_spreading_torch,
)
from modssc.transductive.methods.classic.laplace_learning import (
    LaplaceLearningMethod,
    LaplaceLearningSpec,
    laplace_learning,
    laplace_learning_numpy,
)
from modssc.transductive.methods.classic.laplace_learning import (
    _infer_num_classes as _laplace_infer_num_classes,
)
from modssc.transductive.methods.classic.lazy_random_walk import (
    LazyRandomWalkMethod,
    LazyRandomWalkSpec,
    lazy_random_walk,
    lazy_random_walk_numpy,
)
from modssc.transductive.methods.classic.lazy_random_walk import (
    _encode_binary as _lrw_encode_binary,
)

if torch is not None:
    from modssc.transductive.methods.classic.dynamic_label_propagation import (
        _knn_matrix_torch as _dlp_knn_torch,
    )
    from modssc.transductive.methods.classic.dynamic_label_propagation import (
        dynamic_label_propagation_torch,
    )
    from modssc.transductive.methods.classic.laplace_learning import laplace_learning_torch
    from modssc.transductive.methods.classic.lazy_random_walk import lazy_random_walk_torch


@dataclass(frozen=True)
class DummyGraph:
    edge_index: Any
    edge_weight: Any | None = None


@dataclass(frozen=True)
class DummyNodeDataset:
    X: Any
    y: Any
    graph: DummyGraph
    masks: Mapping[str, Any] | None = None
    meta: Mapping[str, Any] | None = None


def _two_cluster_graph() -> tuple[int, np.ndarray, np.ndarray]:
    n = 6
    edges = []
    for cluster in ((0, 1, 2), (3, 4, 5)):
        for i in cluster:
            for j in cluster:
                if i == j:
                    continue
                edges.append((j, i))
    edge_index = np.array(edges, dtype=np.int64).T
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    return n, edge_index, edge_weight


def _simple_graph(n_nodes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    edge_index = np.array(
        [[i for i in range(n_nodes)], [(i + 1) % n_nodes for i in range(n_nodes)]],
        dtype=np.int64,
    )
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    return edge_index, edge_weight


def _isolated_graph() -> tuple[np.ndarray, np.ndarray]:
    edge_index = np.empty((2, 0), dtype=np.int64)
    edge_weight = np.empty((0,), dtype=np.float32)
    return edge_index, edge_weight


def test_label_propagation_two_clusters_numpy():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labeled = np.zeros(n, dtype=bool)
    labeled[0] = True
    labeled[3] = True

    res = label_propagation(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled,
        spec=LabelPropagationSpec(max_iter=200, tol=1e-9, norm="rw"),
        backend="numpy",
    )
    pred = res.F.argmax(axis=1)
    assert pred.tolist() == y.tolist()

    assert np.allclose(res.F[labeled], np.eye(2, dtype=np.float32)[[0, 1]])


def test_label_spreading_two_clusters_numpy():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labeled = np.zeros(n, dtype=bool)
    labeled[0] = True
    labeled[3] = True

    res = label_spreading(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled,
        spec=LabelSpreadingSpec(alpha=0.99, max_iter=200, tol=1e-9, norm="sym"),
        backend="numpy",
    )
    pred = res.F.argmax(axis=1)
    assert pred.tolist() == y.tolist()
    assert res.n_iter >= 1
    assert res.residual >= 0.0


def test_laplace_learning_two_clusters_numpy():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labeled = np.zeros(n, dtype=bool)
    labeled[0] = True
    labeled[3] = True

    res = laplace_learning(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled,
        backend="numpy",
    )
    pred = res.F.argmax(axis=1)
    assert pred.tolist() == y.tolist()
    assert np.allclose(res.F[labeled], np.eye(2, dtype=np.float32)[[0, 1]])


def test_lazy_random_walk_two_clusters_numpy():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labeled = np.zeros(n, dtype=bool)
    labeled[0] = True
    labeled[3] = True

    res = lazy_random_walk(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled,
        spec=LazyRandomWalkSpec(alpha=0.9),
        backend="numpy",
    )
    pred = res.F.argmax(axis=1)
    assert pred.tolist() == y.tolist()


def test_dynamic_label_propagation_two_clusters_numpy():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labeled = np.zeros(n, dtype=bool)
    labeled[0] = True
    labeled[3] = True

    res = dynamic_label_propagation(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled,
        spec=DynamicLabelPropagationSpec(k_neighbors=2, alpha=0.05, lambda_value=0.1, max_iter=5),
        backend="numpy",
    )
    pred = res.F.argmax(axis=1)
    assert pred.tolist() == y.tolist()


def test_label_propagation_method_fit_predict(monkeypatch):
    import importlib

    lp_mod = importlib.import_module("modssc.transductive.methods.classic.label_propagation")
    monkeypatch.setattr(lp_mod, "torch", None)

    n, edge_index, edge_weight = _two_cluster_graph()
    X = np.zeros((n, 2), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[0] = True
    train_mask[3] = True

    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight),
        masks={"train_mask": train_mask},
    )

    method = lp_mod.LabelPropagationMethod(LabelPropagationSpec(max_iter=200, tol=1e-9, norm="rw"))
    method.fit(data)
    proba = method.predict_proba(data)

    assert proba.shape == (n, 2)
    assert proba.argmax(axis=1).tolist() == y.tolist()


def test_label_spreading_method_fit_predict(monkeypatch):
    import importlib

    ls_mod = importlib.import_module("modssc.transductive.methods.classic.label_spreading")
    monkeypatch.setattr(ls_mod, "torch", None)

    n, edge_index, edge_weight = _two_cluster_graph()
    X = np.zeros((n, 2), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[0] = True
    train_mask[3] = True

    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight),
        masks={"train_mask": train_mask},
    )

    method = ls_mod.LabelSpreadingMethod(LabelSpreadingSpec(alpha=0.99, tol=1e-9))
    method.fit(data)
    proba = method.predict_proba(data)

    assert proba.shape == (n, 2)
    assert proba.argmax(axis=1).tolist() == y.tolist()


def test_laplace_learning_method_fit_predict(monkeypatch):
    import importlib

    ll_mod = importlib.import_module("modssc.transductive.methods.classic.laplace_learning")
    monkeypatch.setattr(ll_mod, "torch", None)

    n, edge_index, edge_weight = _two_cluster_graph()
    X = np.zeros((n, 2), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[0] = True
    train_mask[3] = True

    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight),
        masks={"train_mask": train_mask},
    )

    method = ll_mod.LaplaceLearningMethod(LaplaceLearningSpec())
    method.fit(data)
    proba = method.predict_proba(data)

    assert proba.shape == (n, 2)
    assert proba.argmax(axis=1).tolist() == y.tolist()


def test_lazy_random_walk_method_fit_predict(monkeypatch):
    import importlib

    lrw_mod = importlib.import_module("modssc.transductive.methods.classic.lazy_random_walk")
    monkeypatch.setattr(lrw_mod, "torch", None)

    n, edge_index, edge_weight = _two_cluster_graph()
    X = np.zeros((n, 2), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[0] = True
    train_mask[3] = True

    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight),
        masks={"train_mask": train_mask},
    )

    method = lrw_mod.LazyRandomWalkMethod(LazyRandomWalkSpec(alpha=0.9))
    method.fit(data)
    proba = method.predict_proba(data)

    assert proba.shape == (n, 2)
    assert proba.argmax(axis=1).tolist() == y.tolist()


def test_dynamic_label_propagation_method_fit_predict(monkeypatch):
    import importlib

    dlp_mod = importlib.import_module(
        "modssc.transductive.methods.classic.dynamic_label_propagation"
    )
    monkeypatch.setattr(dlp_mod, "torch", None)

    n, edge_index, edge_weight = _two_cluster_graph()
    X = np.zeros((n, 2), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[0] = True
    train_mask[3] = True

    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight),
        masks={"train_mask": train_mask},
    )

    method = dlp_mod.DynamicLabelPropagationMethod(
        DynamicLabelPropagationSpec(k_neighbors=2, alpha=0.05, lambda_value=0.1, max_iter=5)
    )
    method.fit(data)
    proba = method.predict_proba(data)

    assert proba.shape == (n, 2)
    assert proba.argmax(axis=1).tolist() == y.tolist()


def test_label_methods_require_train_mask() -> None:
    n, edge_index, edge_weight = _two_cluster_graph()
    X = np.zeros((n, 2), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)

    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight),
        masks={},
    )
    with pytest.raises(ValueError, match="train_mask"):
        LabelPropagationMethod().fit(data)
    with pytest.raises(ValueError, match="train_mask"):
        LabelSpreadingMethod().fit(data)
    with pytest.raises(ValueError, match="train_mask"):
        LaplaceLearningMethod().fit(data)
    with pytest.raises(ValueError, match="train_mask"):
        LazyRandomWalkMethod().fit(data)
    with pytest.raises(ValueError, match="train_mask"):
        DynamicLabelPropagationMethod().fit(data)


def test_label_methods_predict_proba_requires_fit() -> None:
    n, edge_index, edge_weight = _two_cluster_graph()
    X = np.zeros((n, 2), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)

    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight),
        masks={"train_mask": np.zeros(n, dtype=bool)},
    )

    with pytest.raises(RuntimeError, match="not fitted"):
        LabelPropagationMethod().predict_proba(data)
    with pytest.raises(RuntimeError, match="not fitted"):
        LabelSpreadingMethod().predict_proba(data)
    with pytest.raises(RuntimeError, match="not fitted"):
        LaplaceLearningMethod().predict_proba(data)
    with pytest.raises(RuntimeError, match="not fitted"):
        LazyRandomWalkMethod().predict_proba(data)
    with pytest.raises(RuntimeError, match="not fitted"):
        DynamicLabelPropagationMethod().predict_proba(data)


def test_torch_backend_matches_numpy_cpu():
    try:
        import torch  # noqa: F401
    except Exception as exc:
        pytest.skip(f"torch unavailable: {exc}")

    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labeled = np.zeros(n, dtype=bool)
    labeled[0] = True
    labeled[3] = True

    spec_lp = LabelPropagationSpec(max_iter=200, tol=1e-9, norm="rw")
    res_np = label_propagation(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled,
        spec=spec_lp,
        backend="numpy",
    )
    res_th = label_propagation(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled,
        spec=spec_lp,
        backend="torch",
        device="cpu",
    )
    assert np.allclose(res_np.F, res_th.F, atol=1e-5, rtol=0)

    spec_ls = LabelSpreadingSpec(alpha=0.99, max_iter=200, tol=1e-9, norm="sym")
    res_np2 = label_spreading(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled,
        spec=spec_ls,
        backend="numpy",
    )
    res_th2 = label_spreading(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled,
        spec=spec_ls,
        backend="torch",
        device="cpu",
    )
    assert np.allclose(res_np2.F, res_th2.F, atol=1e-5, rtol=0)


def test_label_propagation_validation():
    n = 3
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([0, -1, 1])
    labeled_mask = np.array([True, False, True])
    spec = LabelPropagationSpec()

    with pytest.raises(ValueError, match="y must have shape"):
        label_propagation_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y[:-1],
            labeled_mask=labeled_mask,
            spec=spec,
        )

    with pytest.raises(ValueError, match="labeled_mask must have shape"):
        label_propagation_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask[:-1],
            spec=spec,
        )

    pass


def test_label_propagation_edge_cases():
    n = 3
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([0, -1, 1])
    labeled_mask = np.array([True, False, True])

    res = label_propagation_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelPropagationSpec(normalize_rows=False, max_iter=5),
    )
    assert res.n_iter > 0

    res = label_propagation_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelPropagationSpec(max_iter=1, tol=0.0),
    )
    assert res.n_iter == 1


def test_label_spreading_validation():
    n = 3
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([0, -1, 1])
    labeled_mask = np.array([True, False, True])

    with pytest.raises(ValueError, match="alpha must be in"):
        label_spreading_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=LabelSpreadingSpec(alpha=1.5),
        )


def test_dispatchers_extended():
    n = 3
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([0, -1, 1])
    labeled_mask = np.array([True, False, True])

    with pytest.raises(ValueError, match="backend must be one of"):
        label_propagation(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            backend="invalid",
        )

    if torch is not None:
        res = label_propagation(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            backend="torch",
        )
        assert isinstance(res.F, np.ndarray)


def test_missing_torch_mock():
    from unittest.mock import patch

    with patch("modssc.transductive.methods.classic.label_propagation.torch", None):
        n = 3
        edge_index = np.array([[0, 1], [1, 2]])
        edge_weight = np.array([1.0, 1.0])
        y = np.array([0, -1, 1])
        labeled_mask = np.array([True, False, True])

        with pytest.raises(ImportError, match="torch is not available"):
            label_propagation(
                n_nodes=n,
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=y,
                labeled_mask=labeled_mask,
                backend="torch",
            )

        res = label_propagation(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            backend="auto",
        )
        assert isinstance(res.F, np.ndarray)


def test_label_spreading_edge_cases():
    n = 3
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([0, -1, 1])
    labeled_mask = np.array([True, False, True])

    res = label_spreading_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelSpreadingSpec(normalize_rows=False, max_iter=5),
    )
    assert res.n_iter > 0

    res = label_spreading_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelSpreadingSpec(max_iter=1, tol=0.0),
    )
    assert res.n_iter == 1


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_label_spreading_torch_edge_cases():
    n = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
    y = torch.tensor([0, -1, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, False, True], dtype=torch.bool)

    F, n_iter, residual = label_spreading_torch(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelSpreadingSpec(normalize_rows=False, max_iter=5),
    )
    assert n_iter > 0

    F, n_iter, residual = label_spreading_torch(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelSpreadingSpec(max_iter=1, tol=0.0),
    )
    assert n_iter == 1


def test_label_spreading_dispatchers_extended():
    n = 3
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([0, -1, 1])
    labeled_mask = np.array([True, False, True])

    with pytest.raises(ValueError, match="backend must be one of"):
        label_spreading(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            backend="invalid",
        )

    if torch is not None:
        res = label_spreading(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            backend="torch",
        )
        assert isinstance(res.F, np.ndarray)


def test_label_spreading_missing_torch_mock():
    from unittest.mock import patch

    with patch("modssc.transductive.methods.classic.label_spreading.torch", None):
        n = 3
        edge_index = np.array([[0, 1], [1, 2]])
        edge_weight = np.array([1.0, 1.0])
        y = np.array([0, -1, 1])
        labeled_mask = np.array([True, False, True])

        with pytest.raises(ImportError, match="torch is not available"):
            label_spreading(
                n_nodes=n,
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=y,
                labeled_mask=labeled_mask,
                backend="torch",
            )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_label_propagation_torch_validation():
    n = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
    y = torch.tensor([0, -1, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, False, True], dtype=torch.bool)
    spec = LabelPropagationSpec()

    with pytest.raises(ValueError, match="edge_index must have shape"):
        label_propagation_torch(
            n_nodes=n,
            edge_index=torch.tensor([0, 1, 2]),
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
        )

    with pytest.raises(ValueError, match="edge_weight must have shape"):
        label_propagation_torch(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=torch.tensor([1.0]),
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
        )

    with pytest.raises(ValueError, match="y and labeled_mask must have shape"):
        label_propagation_torch(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y[:-1],
            labeled_mask=labeled_mask,
            spec=spec,
        )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_label_propagation_torch_edge_cases():
    n = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
    y = torch.tensor([0, -1, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, False, True], dtype=torch.bool)

    F, n_iter, residual = label_propagation_torch(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelPropagationSpec(normalize_rows=False, max_iter=5),
    )
    assert n_iter > 0

    F, n_iter, residual = label_propagation_torch(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelPropagationSpec(max_iter=1, tol=0.0),
    )
    assert n_iter == 1


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_label_spreading_torch_validation():
    n = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
    y = torch.tensor([0, -1, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, False, True], dtype=torch.bool)

    with pytest.raises(ValueError, match="alpha must be in"):
        label_spreading_torch(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=LabelSpreadingSpec(alpha=1.5),
        )


def test_label_spreading_validation_extended():
    n = 3
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([0, -1, 1])
    labeled_mask = np.array([True, False, True])
    spec = LabelSpreadingSpec()

    with pytest.raises(ValueError, match="y must have shape"):
        label_spreading_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y[:-1],
            labeled_mask=labeled_mask,
            spec=spec,
        )

    with pytest.raises(ValueError, match="labeled_mask must have shape"):
        label_spreading_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask[:-1],
            spec=spec,
        )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_label_spreading_torch_validation_extended():
    n = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
    y = torch.tensor([0, -1, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, False, True], dtype=torch.bool)
    spec = LabelSpreadingSpec()

    with pytest.raises(ValueError, match="edge_index must have shape"):
        label_spreading_torch(
            n_nodes=n,
            edge_index=torch.tensor([0, 1, 2]),
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
        )

    with pytest.raises(ValueError, match="edge_weight must have shape"):
        label_spreading_torch(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=torch.tensor([1.0]),
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
        )

    with pytest.raises(ValueError, match="y and labeled_mask must have shape"):
        label_spreading_torch(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y[:-1],
            labeled_mask=labeled_mask,
            spec=spec,
        )


def test_label_propagation_no_labeled_nodes():
    n = 3
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([-1, -1, -1])
    labeled_mask = np.array([False, False, False])

    res = label_propagation_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelPropagationSpec(max_iter=2),
    )

    assert np.allclose(res.F, 0.0)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_label_propagation_torch_no_labeled_nodes():
    n = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
    y = torch.tensor([-1, -1, -1], dtype=torch.long)
    labeled_mask = torch.tensor([False, False, False], dtype=torch.bool)

    F, n_iter, residual = label_propagation_torch(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelPropagationSpec(max_iter=2),
    )
    assert torch.allclose(F, torch.zeros_like(F))


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_label_propagation_torch_invalid_weight_shape():
    n = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([[1.0, 1.0]], dtype=torch.float)
    y = torch.tensor([0, -1, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, False, True], dtype=torch.bool)

    with pytest.raises(ValueError, match="edge_weight must have shape"):
        label_propagation_torch(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_label_spreading_torch_invalid_weight_shape():
    n = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([[1.0, 1.0]], dtype=torch.float)
    y = torch.tensor([0, -1, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, False, True], dtype=torch.bool)

    with pytest.raises(ValueError, match="edge_weight must have shape"):
        label_spreading_torch(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


def test_label_spreading_auto_fallback_no_torch():
    n = 3
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([0, -1, 1])
    labeled_mask = np.array([True, False, True])

    with patch("modssc.transductive.methods.classic.label_spreading.torch", None):
        res = label_spreading(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            backend="auto",
        )
        assert isinstance(res.F, np.ndarray)


def test_label_spreading_no_labeled_nodes_numpy():
    n = 3
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([-1, -1, -1])
    labeled_mask = np.array([False, False, False])

    res = label_spreading_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelSpreadingSpec(max_iter=2),
    )
    assert np.allclose(res.F, 0.0)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_label_spreading_no_labeled_nodes_torch():
    n = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float32)
    y = torch.tensor([-1, -1, -1], dtype=torch.long)
    labeled_mask = torch.tensor([False, False, False], dtype=torch.bool)

    F, _, _ = label_spreading_torch(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelSpreadingSpec(max_iter=2),
    )
    assert torch.allclose(F, torch.zeros_like(F))


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_label_propagation_torch_invalid_weight_length():
    n = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float)
    y = torch.tensor([0, -1, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, False, True], dtype=torch.bool)

    with pytest.raises(ValueError, match="edge_weight must have shape"):
        label_propagation_torch(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_label_spreading_torch_invalid_weight_length():
    n = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float)
    y = torch.tensor([0, -1, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, False, True], dtype=torch.bool)

    with pytest.raises(ValueError, match="edge_weight must have shape"):
        label_spreading_torch(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


def test_check_import_path():
    import modssc

    print(f"MODSSC PATH: {modssc.__file__}")


def test_label_propagation_numpy_max_iter():
    n = 3
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([0, 1, 0])
    labeled_mask = np.array([True, True, False])
    spec = LabelPropagationSpec(max_iter=1, tol=0.0)

    res = label_propagation_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=spec,
    )

    assert res.n_iter == 1


def test_label_propagation_numpy_max_iter_zero():
    n = 2
    edge_index = np.array([[0, 1], [1, 0]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([0, 1])
    labeled_mask = np.array([True, True])
    spec = LabelPropagationSpec(max_iter=0)

    res = label_propagation_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=spec,
    )
    assert res.n_iter == 0


def test_label_spreading_numpy_max_iter():
    n = 3
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([0, 1, 0])
    labeled_mask = np.array([True, True, False])
    spec = LabelSpreadingSpec(max_iter=1, tol=0.0, alpha=0.5)

    res = label_spreading_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=spec,
    )

    assert res.n_iter == 1


def test_label_propagation_torch_no_labeled_nodes_small_graph():
    if torch is None:
        pytest.skip("torch not installed")

    n = 2
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    labeled_mask = torch.tensor([False, False], dtype=torch.bool)

    spec = LabelPropagationSpec(max_iter=10)

    _, n_iter, _ = label_propagation_torch(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=spec,
    )

    assert n_iter >= 1


def test_label_spreading_torch_invalid_edge_index():
    if torch is None:
        pytest.skip("torch not installed")

    n = 2
    edge_index = torch.zeros((3, 2), dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, True], dtype=torch.bool)

    spec = LabelSpreadingSpec()

    with pytest.raises(ValueError, match="edge_index must have shape"):
        label_spreading_torch(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
        )


def test_label_propagation_numpy_default_spec():
    n = 2
    edge_index = np.array([[0, 1], [1, 0]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([0, 1])
    labeled_mask = np.array([True, True])

    res = label_propagation_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=None,
    )
    assert res.n_iter > 0


def test_label_propagation_torch_default_spec():
    if torch is None:
        pytest.skip("torch not installed")
    n = 2
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, True], dtype=torch.bool)

    _, n_iter, _ = label_propagation_torch(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=None,
    )
    assert n_iter > 0


def test_label_propagation_torch_no_edge_weight():
    if torch is None:
        pytest.skip("torch not installed")
    n = 2
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    y = torch.tensor([0, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, True], dtype=torch.bool)

    _, n_iter, _ = label_propagation_torch(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=None,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelPropagationSpec(max_iter=1),
    )
    assert n_iter > 0


def test_label_spreading_numpy_default_spec():
    n = 2
    edge_index = np.array([[0, 1], [1, 0]])
    edge_weight = np.array([1.0, 1.0])
    y = np.array([0, 1])
    labeled_mask = np.array([True, True])

    res = label_spreading_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=None,
    )
    assert res.n_iter > 0


def test_label_spreading_torch_default_spec():
    if torch is None:
        pytest.skip("torch not installed")
    n = 2
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, True], dtype=torch.bool)

    _, n_iter, _ = label_spreading_torch(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=None,
    )
    assert n_iter > 0


def test_label_spreading_torch_no_edge_weight():
    if torch is None:
        pytest.skip("torch not installed")
    n = 2
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    y = torch.tensor([0, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, True], dtype=torch.bool)

    _, n_iter, _ = label_spreading_torch(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=None,
        y=y,
        labeled_mask=labeled_mask,
        spec=LabelSpreadingSpec(max_iter=1),
    )
    assert n_iter > 0


def test_dynamic_label_propagation_infer_num_classes_uses_full_labels():
    y = np.array([-1, 2, 2], dtype=np.int64)
    labeled_mask = np.array([False, False, False])
    assert _dlp_infer_num_classes(y, labeled_mask) == 3


def test_dynamic_label_propagation_knn_matrix_numpy_paths():
    P0 = np.eye(3, dtype=np.float32)
    assert np.allclose(_dlp_knn_numpy(P0, 0), P0)
    assert np.allclose(_dlp_knn_numpy(P0, 3), P0)
    with pytest.raises(ValueError, match="zero row sum"):
        _dlp_knn_numpy(np.zeros((3, 3), dtype=np.float32), 1)


@pytest.mark.parametrize(
    ("spec", "y", "labeled_mask", "edge_index", "edge_weight", "match"),
    [
        (
            DynamicLabelPropagationSpec(k_neighbors=0),
            np.array([0, 1, 0], dtype=np.int64),
            np.array([True, False, True]),
            *_simple_graph(),
            "k_neighbors must be positive",
        ),
        (
            DynamicLabelPropagationSpec(alpha=-0.1),
            np.array([0, 1, 0], dtype=np.int64),
            np.array([True, False, True]),
            *_simple_graph(),
            "alpha must be non-negative",
        ),
        (
            DynamicLabelPropagationSpec(),
            np.array([0, 1], dtype=np.int64),
            np.array([True, False, True]),
            *_simple_graph(),
            "y must have shape",
        ),
        (
            DynamicLabelPropagationSpec(),
            np.array([0, 1, 0], dtype=np.int64),
            np.array([True, False]),
            *_simple_graph(),
            "labeled_mask must have shape",
        ),
        (
            DynamicLabelPropagationSpec(),
            np.array([0, 1, 0], dtype=np.int64),
            np.array([False, False, False]),
            *_simple_graph(),
            "requires at least 1 labeled node",
        ),
        (
            DynamicLabelPropagationSpec(),
            np.array([0, 1, 0], dtype=np.int64),
            np.array([True, True, False]),
            *_isolated_graph(),
            "graph without isolated nodes",
        ),
    ],
)
def test_dynamic_label_propagation_numpy_validation(
    spec, y, labeled_mask, edge_index, edge_weight, match
):
    with pytest.raises(ValueError, match=match):
        dynamic_label_propagation_numpy(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
        )


def test_dynamic_label_propagation_numpy_lambda_zero():
    edge_index, edge_weight = _simple_graph()
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])
    res = dynamic_label_propagation_numpy(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=DynamicLabelPropagationSpec(k_neighbors=1, lambda_value=0.0, max_iter=1),
    )
    assert res.F.shape == (3, 2)


def test_dynamic_label_propagation_numpy_default_spec():
    edge_index, edge_weight = _simple_graph()
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])
    res = dynamic_label_propagation_numpy(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
    )
    assert res.F.shape == (3, 2)


def test_dynamic_label_propagation_dispatch_invalid_backend():
    edge_index, edge_weight = _simple_graph()
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, False, True])
    with pytest.raises(ValueError, match="backend must be one of"):
        dynamic_label_propagation(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            backend="bad",
        )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_dynamic_label_propagation_torch_paths():
    edge_index, edge_weight = _simple_graph()
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])
    res = dynamic_label_propagation(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=None,
        y=y,
        labeled_mask=labeled_mask,
        spec=DynamicLabelPropagationSpec(k_neighbors=1, lambda_value=0.0, max_iter=1),
        backend="torch",
        device="cpu",
    )
    assert res.F.shape == (3, 2)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_dynamic_label_propagation_torch_default_spec():
    edge_index, edge_weight = _simple_graph()
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    labeled_mask = torch.tensor([True, True, False], dtype=torch.bool)
    res = dynamic_label_propagation_torch(
        n_nodes=3,
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_weight=torch.as_tensor(edge_weight, dtype=torch.float32),
        y=y,
        labeled_mask=labeled_mask,
    )
    assert res.F.shape == (3, 2)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_dynamic_label_propagation_knn_matrix_torch_paths():
    P0 = torch.eye(3, dtype=torch.float32)
    assert torch.allclose(_dlp_knn_torch(P0, 0), P0)
    assert torch.allclose(_dlp_knn_torch(P0, 3), P0)
    with pytest.raises(ValueError, match="zero row sum"):
        _dlp_knn_torch(torch.zeros((3, 3), dtype=torch.float32), 1)


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("spec", "edge_index", "edge_weight", "y", "labeled_mask", "match"),
    [
        (
            DynamicLabelPropagationSpec(k_neighbors=0),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            torch.ones(2, dtype=torch.float32),
            torch.tensor([0, 1, 0], dtype=torch.long),
            torch.tensor([True, False, True], dtype=torch.bool),
            "k_neighbors must be positive",
        ),
        (
            DynamicLabelPropagationSpec(alpha=-0.1),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            torch.ones(2, dtype=torch.float32),
            torch.tensor([0, 1, 0], dtype=torch.long),
            torch.tensor([True, False, True], dtype=torch.bool),
            "alpha must be non-negative",
        ),
        (
            DynamicLabelPropagationSpec(),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            torch.ones(3, dtype=torch.float32),
            torch.tensor([0, 1, 0], dtype=torch.long),
            torch.tensor([True, False, True], dtype=torch.bool),
            "edge_weight must have shape",
        ),
        (
            DynamicLabelPropagationSpec(),
            torch.tensor([[0, 1, 2]], dtype=torch.long),
            None,
            torch.tensor([0, 1, 0], dtype=torch.long),
            torch.tensor([True, False, True], dtype=torch.bool),
            "edge_index must have shape",
        ),
        (
            DynamicLabelPropagationSpec(),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            None,
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([True, False, True], dtype=torch.bool),
            "y and labeled_mask must have shape",
        ),
        (
            DynamicLabelPropagationSpec(),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            None,
            torch.tensor([0, 1, 0], dtype=torch.long),
            torch.tensor([False, False, False], dtype=torch.bool),
            "requires at least 1 labeled node",
        ),
        (
            DynamicLabelPropagationSpec(),
            torch.empty((2, 0), dtype=torch.long),
            None,
            torch.tensor([0, 1, 0], dtype=torch.long),
            torch.tensor([True, True, False], dtype=torch.bool),
            "graph without isolated nodes",
        ),
    ],
)
def test_dynamic_label_propagation_torch_validation(
    spec, edge_index, edge_weight, y, labeled_mask, match
):
    with pytest.raises(ValueError, match=match):
        dynamic_label_propagation_torch(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
        )


def test_laplace_learning_infer_num_classes_uses_full_labels():
    y = np.array([-1, 3, 3], dtype=np.int64)
    labeled_mask = np.array([False, False, False])
    assert _laplace_infer_num_classes(y, labeled_mask) == 4


@pytest.mark.parametrize(
    ("y", "labeled_mask", "edge_index", "edge_weight", "match"),
    [
        (
            np.array([0, 1], dtype=np.int64),
            np.array([True, False, True]),
            *_simple_graph(),
            "y must have shape",
        ),
        (
            np.array([0, 1, 0], dtype=np.int64),
            np.array([True, False]),
            *_simple_graph(),
            "labeled_mask must have shape",
        ),
        (
            np.array([0, 1, 0], dtype=np.int64),
            np.array([False, False, False]),
            *_simple_graph(),
            "requires at least 1 labeled node",
        ),
    ],
)
def test_laplace_learning_numpy_validation(y, labeled_mask, edge_index, edge_weight, match):
    with pytest.raises(ValueError, match=match):
        laplace_learning_numpy(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


def test_laplace_learning_numpy_all_labeled_returns_onehot():
    edge_index, edge_weight = _simple_graph()
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, True])
    res = laplace_learning_numpy(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
    )
    assert res.n_iter == 1
    assert np.allclose(res.F[labeled_mask], np.eye(2, dtype=np.float32)[[0, 1, 0]])


def test_laplace_learning_numpy_singular_system():
    edge_index, edge_weight = _isolated_graph()
    y = np.array([0, 0], dtype=np.int64)
    labeled_mask = np.array([True, False])
    with pytest.raises(ValueError, match="nonsingular"):
        laplace_learning_numpy(
            n_nodes=2,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


def test_laplace_learning_dispatch_invalid_backend():
    edge_index, edge_weight = _simple_graph()
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, False, True])
    with pytest.raises(ValueError, match="backend must be one of"):
        laplace_learning(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            backend="bad",
        )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_laplace_learning_torch_paths():
    edge_index, edge_weight = _simple_graph()
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])
    res = laplace_learning(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        backend="torch",
        device="cpu",
    )
    assert res.F.shape == (3, 2)


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("edge_index", "edge_weight", "y", "labeled_mask", "match"),
    [
        (
            torch.tensor([[0, 1, 2]], dtype=torch.long),
            None,
            torch.tensor([0, 1, 0], dtype=torch.long),
            torch.tensor([True, False, True], dtype=torch.bool),
            "edge_index must have shape",
        ),
        (
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            torch.ones(3, dtype=torch.float32),
            torch.tensor([0, 1, 0], dtype=torch.long),
            torch.tensor([True, False, True], dtype=torch.bool),
            "edge_weight must have shape",
        ),
        (
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            None,
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([True, False, True], dtype=torch.bool),
            "y and labeled_mask must have shape",
        ),
        (
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            None,
            torch.tensor([0, 1, 0], dtype=torch.long),
            torch.tensor([False, False, False], dtype=torch.bool),
            "requires at least 1 labeled node",
        ),
    ],
)
def test_laplace_learning_torch_validation(edge_index, edge_weight, y, labeled_mask, match):
    with pytest.raises(ValueError, match=match):
        laplace_learning_torch(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


def test_lazy_random_walk_encode_binary_full_y():
    y = np.array([0, -1, -1], dtype=np.int64)
    labeled_mask = np.array([True, False, False])
    full_y = np.array([0, 1, 1], dtype=np.int64)
    y_enc, classes = _lrw_encode_binary(y, labeled_mask=labeled_mask, full_y=full_y)
    assert classes.tolist() == [0, 1]
    assert set(np.unique(y_enc)) <= {-1.0, 0.0, 1.0}


def test_lazy_random_walk_encode_binary_errors():
    y = np.array([0, 1, 2], dtype=np.int64)
    labeled_mask = np.array([True, True, True])
    with pytest.raises(ValueError, match="binary"):
        _lrw_encode_binary(y, labeled_mask=labeled_mask)


@pytest.mark.parametrize(
    ("spec", "y", "labeled_mask", "edge_index", "edge_weight", "match"),
    [
        (
            LazyRandomWalkSpec(alpha=1.1),
            np.array([0, 1, 0], dtype=np.int64),
            np.array([True, False, True]),
            *_simple_graph(),
            "alpha must be in",
        ),
        (
            LazyRandomWalkSpec(),
            np.array([0, 1], dtype=np.int64),
            np.array([True, False, True]),
            *_simple_graph(),
            "y must have shape",
        ),
        (
            LazyRandomWalkSpec(),
            np.array([0, 1, 0], dtype=np.int64),
            np.array([True, False]),
            *_simple_graph(),
            "labeled_mask must have shape",
        ),
        (
            LazyRandomWalkSpec(),
            np.array([0, 1, 0], dtype=np.int64),
            np.array([False, False, False]),
            *_simple_graph(),
            "requires at least 1 labeled node",
        ),
        (
            LazyRandomWalkSpec(),
            np.array([0, 1, 0], dtype=np.int64),
            np.array([True, False, True]),
            *_isolated_graph(),
            "graph without isolated nodes",
        ),
    ],
)
def test_lazy_random_walk_numpy_validation(spec, y, labeled_mask, edge_index, edge_weight, match):
    with pytest.raises(ValueError, match=match):
        lazy_random_walk_numpy(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
        )


def test_lazy_random_walk_numpy_linalg_error(monkeypatch):
    edge_index, edge_weight = _simple_graph()
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, False, True])

    def _boom(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "solve", _boom)
    with pytest.raises(ValueError, match="Failed to solve"):
        lazy_random_walk_numpy(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=LazyRandomWalkSpec(alpha=0.9),
        )


def test_laplace_learning_torch_all_labeled_returns_onehot():
    if torch is None:
        pytest.skip("torch not installed")
    edge_index, edge_weight = _simple_graph()
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    labeled_mask = torch.tensor([True, True, True], dtype=torch.bool)
    res = laplace_learning_torch(
        n_nodes=3,
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_weight=torch.as_tensor(edge_weight, dtype=torch.float32),
        y=y,
        labeled_mask=labeled_mask,
    )
    assert res.F.shape == (3, 2)


def test_lazy_random_walk_numpy_default_spec():
    edge_index, edge_weight = _simple_graph()
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, False, True])
    res = lazy_random_walk_numpy(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=None,
    )
    assert res.F.shape == (3, 2)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_lazy_random_walk_torch_paths():
    edge_index, edge_weight = _simple_graph()
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    labeled_mask = torch.tensor([True, False, True], dtype=torch.bool)
    res = lazy_random_walk_torch(
        n_nodes=3,
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_weight=None,
        y=y,
        labeled_mask=labeled_mask,
        spec=None,
    )
    assert res.F.shape == (3, 2)


def test_lazy_random_walk_dispatch_invalid_backend():
    edge_index, edge_weight = _simple_graph()
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, False, True])
    with pytest.raises(ValueError, match="backend must be one of"):
        lazy_random_walk(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            backend="bad",
        )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_lazy_random_walk_torch_backend_dispatch():
    edge_index, edge_weight = _simple_graph()
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, False, True])
    res = lazy_random_walk(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=None,
        backend="torch",
        device="cpu",
    )
    assert res.F.shape == (3, 2)


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize(
    ("spec", "edge_index", "edge_weight", "y", "labeled_mask", "match"),
    [
        (
            LazyRandomWalkSpec(alpha=1.1),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            torch.ones(2, dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([True, True], dtype=torch.bool),
            "alpha must be in",
        ),
        (
            LazyRandomWalkSpec(),
            torch.tensor([[0, 1, 2]], dtype=torch.long),
            None,
            torch.tensor([0, 1, 0], dtype=torch.long),
            torch.tensor([True, False, True], dtype=torch.bool),
            "edge_index must have shape",
        ),
        (
            LazyRandomWalkSpec(),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            torch.ones(3, dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([True, True], dtype=torch.bool),
            "edge_weight must have shape",
        ),
        (
            LazyRandomWalkSpec(),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            torch.ones(2, dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([True, False, True], dtype=torch.bool),
            "y and labeled_mask must have shape",
        ),
        (
            LazyRandomWalkSpec(),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            torch.ones(2, dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([False, False], dtype=torch.bool),
            "requires at least 1 labeled node",
        ),
        (
            LazyRandomWalkSpec(),
            torch.empty((2, 0), dtype=torch.long),
            None,
            torch.tensor([0, 1, 0], dtype=torch.long),
            torch.tensor([True, False, True], dtype=torch.bool),
            "graph without isolated nodes",
        ),
    ],
)
def test_lazy_random_walk_torch_validation(spec, edge_index, edge_weight, y, labeled_mask, match):
    with pytest.raises(ValueError, match=match):
        lazy_random_walk_torch(
            n_nodes=int(y.shape[0]),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
        )


def test_dynamic_label_propagation_torch_oom(monkeypatch):
    import importlib

    import torch

    dlp = importlib.import_module("modssc.transductive.methods.classic.dynamic_label_propagation")

    def _oom(**_kwargs):
        raise torch.cuda.OutOfMemoryError("oom")

    monkeypatch.setattr(dlp, "dynamic_label_propagation_torch", _oom)

    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    y = np.array([0, 1], dtype=np.int64)
    labeled = np.array([True, False])

    with pytest.raises(RuntimeError, match="out of memory"):
        dlp.dynamic_label_propagation(
            n_nodes=2,
            edge_index=edge_index,
            edge_weight=None,
            y=y,
            labeled_mask=labeled,
            backend="torch",
            device="cpu",
        )
