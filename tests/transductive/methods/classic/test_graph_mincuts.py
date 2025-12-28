from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from modssc.transductive.errors import OptionalDependencyError
from modssc.transductive.methods.classic.graph_mincuts import (
    GraphMincutsMethod,
    GraphMincutsSpec,
    _reachable_from_source_csr,
    graph_mincuts,
)

graph_mincuts_mod = importlib.import_module("modssc.transductive.methods.classic.graph_mincuts")


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
    weights = []

    def add_undirected(i: int, j: int, w: float) -> None:
        edges.append((i, j))
        edges.append((j, i))
        weights.append(w)
        weights.append(w)

    for i in range(3):
        for j in range(i + 1, 3):
            add_undirected(i, j, 10.0)
    for i in range(3, 6):
        for j in range(i + 1, 6):
            add_undirected(i, j, 10.0)

    add_undirected(2, 3, 0.1)

    edge_index = np.asarray(edges, dtype=np.int64).T
    edge_weight = np.asarray(weights, dtype=np.float32)
    return n, edge_index, edge_weight


def test_graph_mincuts_binary_two_clusters():
    pytest.importorskip("scipy")

    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[0] = True
    labeled_mask[3] = True

    res = graph_mincuts(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=GraphMincutsSpec(capacity_scale=1000.0),
    )

    pred = res.F.argmax(axis=1)
    assert pred.tolist() == y.tolist()


def test_graph_mincuts_method_fit_predict():
    pytest.importorskip("scipy")

    n, edge_index, edge_weight = _two_cluster_graph()
    X = np.zeros((n, 2), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[0] = True
    labeled_mask[3] = True

    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight),
        masks={"train_mask": labeled_mask},
    )

    method = GraphMincutsMethod(GraphMincutsSpec(capacity_scale=1000.0))
    method.fit(data)
    proba = method.predict_proba(data)

    assert proba.shape == (n, 2)
    assert proba.argmax(axis=1).tolist() == y.tolist()


def test_graph_mincuts_method_requires_train_mask():
    n, edge_index, edge_weight = _two_cluster_graph()
    X = np.zeros((n, 2), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    data = DummyNodeDataset(
        X=X,
        y=y,
        graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight),
        masks={},
    )

    with pytest.raises(ValueError, match="train_mask"):
        GraphMincutsMethod().fit(data)


def test_graph_mincuts_predict_proba_requires_fit():
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
        GraphMincutsMethod().predict_proba(data)


def test_graph_mincuts_requires_binary():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 2, 1, 1, 1], dtype=np.int64)

    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[0] = True
    labeled_mask[3] = True
    labeled_mask[2] = True

    with pytest.raises(ValueError):
        graph_mincuts(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


def test_graph_mincuts_invalid_shapes():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.array([True, True, False, False, False, False])

    with pytest.raises(ValueError, match="y must have shape"):
        graph_mincuts(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y[:-1],
            labeled_mask=labeled_mask,
        )

    with pytest.raises(ValueError, match="labeled_mask must have shape"):
        graph_mincuts(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask[:-1],
        )


def test_graph_mincuts_no_labeled_nodes():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.zeros(n, dtype=bool)

    with pytest.raises(ValueError, match="requires at least 1 labeled node"):
        graph_mincuts(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


def test_graph_mincuts_requires_each_class(monkeypatch):
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.zeros(n, dtype=np.int64)
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[:2] = True

    monkeypatch.setattr(
        graph_mincuts_mod.np,
        "unique",
        lambda arr: np.array([0, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="Need at least one labeled example per class"):
        graph_mincuts(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


def test_graph_mincuts_invalid_capacity_scale():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.array([True, True, False, False, False, False])

    with pytest.raises(ValueError, match="capacity_scale must be > 0"):
        graph_mincuts(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=GraphMincutsSpec(capacity_scale=0.0),
        )


def test_graph_mincuts_non_finite_edge_weight():
    n, edge_index, edge_weight = _two_cluster_graph()
    edge_weight = edge_weight.copy()
    edge_weight[0] = np.nan
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.array([True, True, False, False, False, False])

    with pytest.raises(ValueError, match="edge_weight contains non-finite"):
        graph_mincuts(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


def test_graph_mincuts_missing_maximum_flow(monkeypatch):
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.array([True, True, False, False, False, False])

    class FakeSparse:
        def coo_matrix(self, *args, **kwargs):
            class FakeCOO:
                def tocsr(self_inner):
                    return np.zeros((n + 2, n + 2), dtype=np.int64)

            return FakeCOO()

    def fake_optional_import(name, *args, **kwargs):
        if name == "scipy.sparse":
            return FakeSparse()
        if name == "scipy.sparse.csgraph":
            raise ImportError("no maximum_flow")
        raise ImportError(name)

    monkeypatch.setattr(graph_mincuts_mod, "optional_import", fake_optional_import)

    with pytest.raises(OptionalDependencyError, match="maximum_flow"):
        graph_mincuts(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


def test_graph_mincuts_maximum_flow_failure(monkeypatch):
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.array([True, True, False, False, False, False])

    class FakeSparse:
        def coo_matrix(self, *args, **kwargs):
            class FakeCOO:
                def tocsr(self_inner):
                    return np.zeros((n + 2, n + 2), dtype=np.int64)

            return FakeCOO()

    class FakeCsgraph:
        def maximum_flow(self, *args, **kwargs):
            raise RuntimeError("boom")

    def fake_optional_import(name, *args, **kwargs):
        if name == "scipy.sparse":
            return FakeSparse()
        if name == "scipy.sparse.csgraph":
            return FakeCsgraph()
        raise ImportError(name)

    monkeypatch.setattr(graph_mincuts_mod, "optional_import", fake_optional_import)

    with pytest.raises(RuntimeError, match="SciPy maximum_flow failed"):
        graph_mincuts(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


def test_graph_mincuts_constraint_violation(monkeypatch):
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.array([True, True, False, False, False, False])

    class FakeSparse:
        def coo_matrix(self, *args, **kwargs):
            class FakeCOO:
                def tocsr(self_inner):
                    return np.zeros((n + 2, n + 2), dtype=np.int64)

            return FakeCOO()

    class FakeFlow:
        flow = 0

    class FakeCsgraph:
        def maximum_flow(self, *args, **kwargs):
            return FakeFlow()

    def fake_optional_import(name, *args, **kwargs):
        if name == "scipy.sparse":
            return FakeSparse()
        if name == "scipy.sparse.csgraph":
            return FakeCsgraph()
        raise ImportError(name)

    monkeypatch.setattr(graph_mincuts_mod, "optional_import", fake_optional_import)
    monkeypatch.setattr(
        graph_mincuts_mod,
        "_reachable_from_source_csr",
        lambda residual, source: np.zeros((n + 2,), dtype=bool),
    )

    with pytest.raises(RuntimeError, match="Mincut constraints not satisfied"):
        graph_mincuts(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
        )


def test_reachable_from_source_csr_simple():
    class FakeCSR:
        def __init__(self):
            self.shape = (4, 4)
            self.indptr = np.array([0, 2, 3, 3, 3], dtype=np.int64)
            self.indices = np.array([1, 3, 2], dtype=np.int64)
            self.data = np.array([1, 0, 1], dtype=np.int64)

    reachable = _reachable_from_source_csr(FakeCSR(), source=0)
    assert reachable.tolist() == [True, True, True, False]


def test_graph_mincuts_success_without_scipy(monkeypatch):
    n_nodes = 2
    edge_index = np.array([[0], [1]], dtype=np.int64)
    edge_weight = np.array([1.0], dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    labeled_mask = np.array([True, True])

    class FakeCSR:
        def __init__(self, shape):
            self.shape = shape

        def __sub__(self, other):
            return self

    class FakeCOO:
        def __init__(self, shape):
            self._shape = shape

        def tocsr(self):
            return FakeCSR(self._shape)

    class FakeSparse:
        def coo_matrix(self, *args, **kwargs):
            return FakeCOO(kwargs.get("shape"))

    class FakeFlow:
        def __init__(self, shape):
            self.flow = FakeCSR(shape)

    class FakeCsgraph:
        def maximum_flow(self, capacity, source, sink):
            return FakeFlow(capacity.shape)

    def fake_optional_import(name, *args, **kwargs):
        if name == "scipy.sparse":
            return FakeSparse()
        if name == "scipy.sparse.csgraph":
            return FakeCsgraph()
        raise ImportError(name)

    monkeypatch.setattr(graph_mincuts_mod, "optional_import", fake_optional_import)
    monkeypatch.setattr(
        graph_mincuts_mod,
        "_reachable_from_source_csr",
        lambda residual, source: np.array([False, True, True, False]),
    )

    res = graph_mincuts(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=GraphMincutsSpec(capacity_scale=1.0),
    )
    assert res.F.shape == (2, 2)
    assert res.F.argmax(axis=1).tolist() == [0, 1]


def test_graph_mincuts_method_fit_predict_without_scipy(monkeypatch):
    def fake_graph_mincuts(*args, **kwargs):
        return graph_mincuts_mod.DiffusionResult(
            F=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            n_iter=1,
            residual=0.0,
        )

    monkeypatch.setattr(graph_mincuts_mod, "graph_mincuts", fake_graph_mincuts)

    data = DummyNodeDataset(
        X=np.zeros((2, 2), dtype=np.float32),
        y=np.array([0, 1], dtype=np.int64),
        graph=DummyGraph(edge_index=np.array([[0], [1]], dtype=np.int64), edge_weight=None),
        masks={"train_mask": np.array([True, True])},
    )

    method = GraphMincutsMethod()
    method.fit(data)
    proba = method.predict_proba(data)
    assert proba.shape == (2, 2)
