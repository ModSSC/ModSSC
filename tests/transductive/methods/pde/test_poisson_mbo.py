from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from modssc.transductive.methods.pde.poisson_mbo import (
    PoissonMBOMethod,
    PoissonMBOSpec,
    _build_b_matrix,
    _build_b_prior,
    _symmetrize_edges,
    poisson_mbo_numpy,
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


def test_poisson_mbo_numpy_shapes():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[0] = True
    labeled_mask[3] = True

    res = poisson_mbo_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonMBOSpec(T=30, Ninner=2, Nouter=2, n_volume_iters=5),
    )

    assert res.F.shape == (n, 2)
    row_sums = res.F.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert np.isin(res.F, [0.0, 1.0]).all()


def test_poisson_mbo_method_fit_predict():
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

    method = PoissonMBOMethod(PoissonMBOSpec(T=30, Ninner=2, Nouter=2, n_volume_iters=5))
    method.fit(data)
    proba = method.predict_proba(data)

    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_poisson_mbo_requires_labeled_per_class():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[0] = True

    with pytest.raises(ValueError, match="labeled node per class"):
        poisson_mbo_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PoissonMBOSpec(T=5, Ninner=1, Nouter=1, n_volume_iters=2),
        )


def test_build_b_matrix_requires_labels():
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([False, False, False])
    with pytest.raises(ValueError, match="at least 1 labeled node"):
        _build_b_matrix(y=y, labeled_mask=labeled_mask, n_classes=2)


def test_build_b_matrix_requires_labeled_per_class():
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, False, False])
    with pytest.raises(ValueError, match="per class"):
        _build_b_matrix(y=y, labeled_mask=labeled_mask, n_classes=2)


def test_build_b_prior_strategies():
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])
    y_bar = np.array([0.5, 0.5], dtype=np.float32)

    out = _build_b_prior(
        y=y, labeled_mask=labeled_mask, n_classes=2, strategy="uniform", y_bar=y_bar
    )
    assert np.allclose(out.sum(), 1.0)

    out = _build_b_prior(
        y=y, labeled_mask=labeled_mask, n_classes=2, strategy="labeled", y_bar=y_bar
    )
    assert np.allclose(out, y_bar)

    out = _build_b_prior(y=y, labeled_mask=labeled_mask, n_classes=2, strategy="true", y_bar=y_bar)
    assert np.allclose(out.sum(), 1.0)


def test_build_b_prior_errors():
    y = np.array([-1, -1], dtype=np.int64)
    labeled_mask = np.array([False, False])
    y_bar = np.array([0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="valid label"):
        _build_b_prior(y=y, labeled_mask=labeled_mask, n_classes=2, strategy="true", y_bar=y_bar)
    with pytest.raises(ValueError, match="Unknown b_strategy"):
        _build_b_prior(
            y=np.array([0, 1], dtype=np.int64),
            labeled_mask=np.array([True, False]),
            n_classes=2,
            strategy="bad",
            y_bar=y_bar,
        )


def test_build_b_prior_true_counts_zero(monkeypatch):
    pmbo = importlib.import_module("modssc.transductive.methods.pde.poisson_mbo")

    def _zero_bincount(*args, **kwargs):
        minlength = kwargs.get("minlength")
        size = int(minlength) if minlength is not None else 0
        return np.zeros((size,), dtype=np.int64)

    monkeypatch.setattr(pmbo.np, "bincount", _zero_bincount)
    with pytest.raises(ValueError, match="requires at least one valid label"):
        _build_b_prior(
            y=np.array([0, 1], dtype=np.int64),
            labeled_mask=np.array([True, False]),
            n_classes=2,
            strategy="true",
            y_bar=np.array([0.5, 0.5], dtype=np.float32),
        )


def test_symmetrize_edges_zero_diag_false():
    edge_index = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    out_index, out_weight = _symmetrize_edges(edge_index, edge_weight, zero_diagonal=False)
    assert out_index.shape == (2, out_weight.shape[0])
    assert np.any(out_index[0] == out_index[1])


def test_poisson_mbo_numpy_b_override_validation():
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])

    with pytest.raises(ValueError, match="b must have shape"):
        poisson_mbo_numpy(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PoissonMBOSpec(b=np.array([1.0, 0.0, 0.0])),
        )
    with pytest.raises(ValueError, match="non-negative"):
        poisson_mbo_numpy(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PoissonMBOSpec(b=np.array([1.0, -1.0])),
        )
    with pytest.raises(ValueError, match="sum to a positive"):
        poisson_mbo_numpy(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PoissonMBOSpec(b=np.array([0.0, 0.0])),
        )


def test_poisson_mbo_numpy_b_override_normalized():
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])
    res = poisson_mbo_numpy(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonMBOSpec(T=2, Ninner=1, Nouter=1, n_volume_iters=1, b=np.array([2.0, 1.0])),
    )
    assert res.F.shape == (3, 2)


def test_poisson_mbo_numpy_symmetrize_false_zero_diag():
    edge_index = np.array([[0, 1, 1], [0, 2, 1]], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])
    res = poisson_mbo_numpy(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonMBOSpec(T=2, Ninner=1, Nouter=1, n_volume_iters=1, symmetrize=False),
    )
    assert res.F.shape == (3, 2)


def test_poisson_mbo_numpy_symmetrize_false_keep_diagonal():
    edge_index = np.array([[0, 1, 1], [0, 2, 1]], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])
    res = poisson_mbo_numpy(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonMBOSpec(
            T=2,
            Ninner=1,
            Nouter=1,
            n_volume_iters=1,
            symmetrize=False,
            zero_diagonal=False,
        ),
    )
    assert res.F.shape == (3, 2)


def test_poisson_mbo_numpy_spec_none_uses_default(monkeypatch):
    pmbo = importlib.import_module("modssc.transductive.methods.pde.poisson_mbo")

    class _TinySpec:
        def __init__(self) -> None:
            self.T = 1
            self.Ninner = 1
            self.Nouter = 1
            self.mu = 1.0
            self.d_tau = 1.0
            self.smin = 0.5
            self.smax = 2.0
            self.n_volume_iters = 1
            self.b_strategy = "uniform"
            self.b = None
            self.symmetrize = True
            self.zero_diagonal = True

    monkeypatch.setattr(pmbo, "PoissonMBOSpec", _TinySpec)

    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])
    res = pmbo.poisson_mbo_numpy(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=None,
    )
    assert res.F.shape == (3, 2)


@pytest.mark.parametrize(
    ("y", "labeled_mask", "edge_index", "edge_weight", "match"),
    [
        (
            np.array([0, 1], dtype=np.int64),
            np.array([True, False, True]),
            np.array([[0, 1], [1, 2]], dtype=np.int64),
            np.ones(2, dtype=np.float32),
            "y must have shape",
        ),
        (
            np.array([0, 1, 0], dtype=np.int64),
            np.array([True, False]),
            np.array([[0, 1], [1, 2]], dtype=np.int64),
            np.ones(2, dtype=np.float32),
            "labeled_mask must have shape",
        ),
        (
            np.array([-1, -1, -1], dtype=np.int64),
            np.array([True, False, True]),
            np.array([[0, 1], [1, 2]], dtype=np.int64),
            np.ones(2, dtype=np.float32),
            "at least one valid label",
        ),
    ],
)
def test_poisson_mbo_numpy_validation(y, labeled_mask, edge_index, edge_weight, match):
    with pytest.raises(ValueError, match=match):
        poisson_mbo_numpy(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PoissonMBOSpec(T=2, Ninner=1, Nouter=1, n_volume_iters=1),
        )


def test_poisson_mbo_numpy_no_edges():
    edge_index = np.empty((2, 0), dtype=np.int64)
    edge_weight = np.empty((0,), dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)
    labeled_mask = np.array([True, True, False])
    with pytest.raises(ValueError, match="at least one edge"):
        poisson_mbo_numpy(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PoissonMBOSpec(T=2, Ninner=1, Nouter=1, n_volume_iters=1),
        )


def test_poisson_mbo_method_errors():
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    X = np.zeros((3, 2), dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)
    data = DummyNodeDataset(
        X=X, y=y, graph=DummyGraph(edge_index=edge_index, edge_weight=edge_weight)
    )
    with pytest.raises(ValueError, match="train_mask"):
        PoissonMBOMethod().fit(data)
    with pytest.raises(RuntimeError, match="not fitted"):
        PoissonMBOMethod().predict_proba(data)
