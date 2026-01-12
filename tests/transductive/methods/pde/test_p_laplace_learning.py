from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from modssc.transductive.methods.pde.p_laplace_learning import (
    PLaplaceLearningMethod,
    PLaplaceLearningSpec,
    p_laplace_learning,
    p_laplace_learning_numpy,
)


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


def _line_graph(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    edge_index = np.array(
        [[i for i in range(n_nodes - 1)], [i + 1 for i in range(n_nodes - 1)]],
        dtype=np.int64,
    )
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    return edge_index, edge_weight


def test_p_laplace_learning_numpy_two_clusters():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[0] = True
    labeled_mask[3] = True

    res = p_laplace_learning_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PLaplaceLearningSpec(p=2.0, max_iter=5, cg_max_iter=500),
    )

    proba = res.F
    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    pred = proba.argmax(axis=1)
    assert pred.tolist() == y.tolist()


def test_p_laplace_learning_method_fit_predict():
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

    method = PLaplaceLearningMethod(PLaplaceLearningSpec(p=2.0, max_iter=5, cg_max_iter=500))
    method.fit(data)
    proba = method.predict_proba(data)

    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_p_laplace_requires_labeled_per_class():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[0] = True

    with pytest.raises(ValueError, match="labeled node per class"):
        p_laplace_learning_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PLaplaceLearningSpec(p=2.0),
        )


def test_p_laplace_learning_default_spec_all_labeled():
    edge_index, edge_weight = _line_graph(3)
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, True])
    res = p_laplace_learning_numpy(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
    )
    assert res.n_iter == 0
    assert res.F.shape == (3, 2)


def test_p_laplace_learning_invalid_p():
    edge_index, edge_weight = _line_graph(3)
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, False, True])
    with pytest.raises(ValueError, match="p must be"):
        p_laplace_learning_numpy(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PLaplaceLearningSpec(p=1.5),
        )


@pytest.mark.parametrize(
    ("y", "labeled_mask", "edge_index", "edge_weight", "match"),
    [
        (
            np.array([0, 1], dtype=np.int64),
            np.array([True, False, True]),
            *_line_graph(3),
            "y must have shape",
        ),
        (
            np.array([0, 1, 0], dtype=np.int64),
            np.array([True, False]),
            *_line_graph(3),
            "labeled_mask must have shape",
        ),
        (
            np.array([0, 1, 0], dtype=np.int64),
            np.array([False, False, False]),
            *_line_graph(3),
            "requires at least 1 labeled node",
        ),
        (
            np.array([-1, -1, -1], dtype=np.int64),
            np.array([True, False, True]),
            *_line_graph(3),
            "at least one valid label",
        ),
    ],
)
def test_p_laplace_learning_validation(y, labeled_mask, edge_index, edge_weight, match):
    with pytest.raises(ValueError, match=match):
        p_laplace_learning_numpy(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PLaplaceLearningSpec(p=2.0, max_iter=1, cg_max_iter=10),
        )


def test_p_laplace_learning_no_label_connections():
    n_nodes = 3
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])
    with pytest.raises(ValueError, match="connected to labels"):
        p_laplace_learning_numpy(
            n_nodes=n_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PLaplaceLearningSpec(p=2.0, max_iter=1, cg_max_iter=10),
        )


def test_p_laplace_learning_no_unlabeled_edges():
    n_nodes = 3
    edge_index = np.array([[0, 1], [2, 2]], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])
    res = p_laplace_learning_numpy(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PLaplaceLearningSpec(p=2.0, max_iter=1, cg_max_iter=10),
    )
    assert res.F.shape == (n_nodes, 2)


def test_p_laplace_learning_p_greater_than_two():
    edge_index, edge_weight = _line_graph(4)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.array([True, True, False, False])
    res = p_laplace_learning_numpy(
        n_nodes=4,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PLaplaceLearningSpec(p=3.0, max_iter=1, cg_max_iter=10),
    )
    assert res.F.shape == (4, 2)


def test_p_laplace_learning_no_symmetrize_no_zero_diagonal():
    edge_index, edge_weight = _line_graph(4)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.array([True, True, False, False])
    res = p_laplace_learning_numpy(
        n_nodes=4,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PLaplaceLearningSpec(
            p=2.0, max_iter=1, cg_max_iter=10, symmetrize=False, zero_diagonal=False
        ),
    )
    assert res.F.shape == (4, 2)


def test_p_laplace_learning_wrapper():
    edge_index, edge_weight = _line_graph(4)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.array([True, True, False, False])
    res = p_laplace_learning(
        n_nodes=4,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PLaplaceLearningSpec(p=2.0, max_iter=1, cg_max_iter=10),
    )
    assert res.F.shape == (4, 2)


def test_p_laplace_learning_b_weight_empty_branch(monkeypatch):
    n_nodes = 4
    edge_index = np.array([[2, 3], [3, 2]], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.array([True, True, False, False])

    pll = importlib.import_module("modssc.transductive.methods.pde.p_laplace_learning")

    orig_vstack = np.vstack

    def _vstack(arrs):
        if all(a.size == 0 for a in arrs):
            return np.zeros((2, 1), dtype=np.int64)
        return orig_vstack(arrs)

    monkeypatch.setattr(pll.np, "vstack", _vstack)

    res = p_laplace_learning_numpy(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PLaplaceLearningSpec(p=2.0, max_iter=1, cg_max_iter=10),
    )
    assert res.F.shape == (n_nodes, 2)


def test_p_laplace_method_errors():
    edge_index, edge_weight = _line_graph(3)
    X = np.zeros((3, 2), dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)
    data = DummyNodeDataset(
        X=X, y=y, graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight)
    )
    with pytest.raises(ValueError, match="train_mask"):
        PLaplaceLearningMethod().fit(data)
    with pytest.raises(RuntimeError, match="not fitted"):
        PLaplaceLearningMethod().predict_proba(data)
