from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

try:
    import torch
except Exception:
    torch = None

from modssc.transductive.methods.pde.poisson_learning import (
    PoissonLearningMethod,
    PoissonLearningSpec,
    _build_sources,
)

pl = importlib.import_module("modssc.transductive.methods.pde.poisson_learning")


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


def test_poisson_learning_numpy_two_clusters():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[0] = True
    labeled_mask[3] = True

    res = pl.poisson_learning(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonLearningSpec(backend="numpy", laplacian_kind="sym", eps=1e-6, max_iter=500),
    )

    pred = res.F.argmax(axis=1)
    assert pred.tolist() == y.tolist()


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_poisson_learning_torch_two_clusters():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    labeled_mask = torch.zeros((n,), dtype=torch.bool)
    labeled_mask[0] = True
    labeled_mask[3] = True

    res = pl.poisson_learning(
        n_nodes=n,
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_weight=torch.as_tensor(edge_weight, dtype=torch.float32),
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonLearningSpec(backend="torch", laplacian_kind="sym", eps=1e-6, max_iter=500),
    )

    pred = np.asarray(res.F).argmax(axis=1)
    assert pred.tolist() == [0, 0, 0, 1, 1, 1]


def test_poisson_learning_method_fit_predict():
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

    method = PoissonLearningMethod(PoissonLearningSpec(backend="numpy", max_iter=500))
    method.fit(data)
    proba = method.predict_proba(data)

    assert proba.shape == (n, 2)
    assert proba.argmax(axis=1).tolist() == y.tolist()


def test_build_sources_requires_labeled_nodes():
    Y = np.zeros((3, 2), dtype=np.float32)
    labeled_mask = np.zeros(3, dtype=bool)
    with pytest.raises(ValueError, match="requires at least 1 labeled"):
        _build_sources(Y_labeled=Y, labeled_mask=labeled_mask, center_sources=True)


def test_build_sources_center_false_zero_sum():
    Y = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    labeled_mask = np.array([True, True, False])
    B = _build_sources(Y_labeled=Y, labeled_mask=labeled_mask, center_sources=False)
    assert B.shape == (3, 2)
    assert np.allclose(B.mean(axis=0), 0.0, atol=1e-6)


def test_poisson_learning_numpy_invalid_shapes():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.array([True, False, True, False, False, False])

    with pytest.raises(ValueError, match="y must have shape"):
        pl.poisson_learning_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y[:-1],
            labeled_mask=labeled_mask,
            spec=PoissonLearningSpec(),
        )

    with pytest.raises(ValueError, match="labeled_mask must have shape"):
        pl.poisson_learning_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask[:-1],
            spec=PoissonLearningSpec(),
        )


def test_poisson_learning_numpy_requires_labels():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    labeled_mask = np.zeros(n, dtype=bool)

    with pytest.raises(ValueError, match="requires at least 1 labeled"):
        pl.poisson_learning_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PoissonLearningSpec(),
        )


def test_poisson_learning_numpy_eps_zero_branch(monkeypatch):
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labeled_mask = np.array([True, False, False, True, False, False])

    monkeypatch.setattr(pl, "laplacian_matvec_numpy", lambda **_: (lambda x: np.zeros_like(x)))

    def fake_cg(matvec, b, tol, max_iter):
        matvec(np.zeros_like(b))
        return SimpleNamespace(x=np.zeros_like(b), n_iter=1, residual_norm=0.0)

    monkeypatch.setattr(pl, "cg_solve_numpy", fake_cg)

    res = pl.poisson_learning_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonLearningSpec(eps=0.0, max_iter=1),
    )
    assert res.F.shape == (n, 2)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_poisson_learning_torch_invalid_edge_index():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, False, True, False, False, False], dtype=torch.bool)

    with pytest.raises(ValueError, match="edge_index must have shape"):
        pl.poisson_learning_torch(
            n_nodes=n,
            edge_index=torch.zeros((3, 2), dtype=torch.long),
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PoissonLearningSpec(),
        )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_poisson_learning_torch_length_and_labels():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = torch.tensor([0, 1], dtype=torch.long)
    labeled_mask = torch.tensor([False, False], dtype=torch.bool)

    with pytest.raises(ValueError, match="y must have length"):
        pl.poisson_learning_torch(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PoissonLearningSpec(),
        )

    y_full = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    labeled_mask = torch.zeros(n, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires at least 1 labeled"):
        pl.poisson_learning_torch(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y_full,
            labeled_mask=labeled_mask,
            spec=PoissonLearningSpec(),
        )


def test_poisson_learning_backend_auto_fallback(monkeypatch):
    def fake_optional_import(*args, **kwargs):
        raise ImportError("no torch")

    monkeypatch.setattr(pl, "optional_import", fake_optional_import)
    monkeypatch.setattr(
        pl,
        "poisson_learning_numpy",
        lambda **kwargs: pl.DiffusionResult(F=np.zeros((1, 1)), n_iter=0, residual=0.0),
    )

    res = pl.poisson_learning(
        n_nodes=1,
        edge_index=np.array([[0], [0]]),
        edge_weight=np.array([1.0]),
        y=np.array([0]),
        labeled_mask=np.array([True]),
        spec=PoissonLearningSpec(backend="auto"),
    )
    assert res.F.shape == (1, 1)


def test_poisson_learning_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        pl.poisson_learning(
            n_nodes=1,
            edge_index=np.array([[0], [0]]),
            edge_weight=np.array([1.0]),
            y=np.array([0]),
            labeled_mask=np.array([True]),
            spec=PoissonLearningSpec(backend="weird"),
        )


def test_poisson_learning_spec_none_uses_default(monkeypatch):
    monkeypatch.setattr(
        pl,
        "poisson_learning_numpy",
        lambda **kwargs: pl.DiffusionResult(F=np.zeros((1, 1)), n_iter=0, residual=0.0),
    )

    res = pl.poisson_learning(
        n_nodes=1,
        edge_index=np.array([[0], [0]]),
        edge_weight=np.array([1.0]),
        y=np.array([0]),
        labeled_mask=np.array([True]),
        spec=None,
    )
    assert res.F.shape == (1, 1)


def test_poisson_learning_auto_prefers_torch(monkeypatch):
    called = {"torch": 0}

    monkeypatch.setattr(pl, "optional_import", lambda *args, **kwargs: object())

    def fake_torch(**kwargs):
        called["torch"] += 1
        return pl.DiffusionResult(F=np.zeros((1, 1)), n_iter=0, residual=0.0)

    monkeypatch.setattr(pl, "poisson_learning_torch", fake_torch)

    res = pl.poisson_learning(
        n_nodes=1,
        edge_index=np.array([[0], [0]]),
        edge_weight=np.array([1.0]),
        y=np.array([0]),
        labeled_mask=np.array([True]),
        spec=PoissonLearningSpec(backend="auto"),
    )
    assert res.F.shape == (1, 1)
    assert called["torch"] == 1


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_poisson_learning_torch_eps_zero_branch(monkeypatch):
    n = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float32)
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    labeled_mask = torch.tensor([True, True, False], dtype=torch.bool)

    monkeypatch.setattr(pl, "laplacian_matvec_torch", lambda **_: (lambda x: torch.zeros_like(x)))

    def fake_cg(matvec, b, device, tol, max_iter):
        matvec(torch.zeros_like(b))
        return torch.zeros_like(b), {"n_iter": 1, "residual_norm": 0.0}

    monkeypatch.setattr(pl, "cg_solve_torch", fake_cg)

    res = pl.poisson_learning_torch(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonLearningSpec(eps=0.0, max_iter=1),
    )
    assert res.F.shape == (n, 2)


def test_poisson_learning_method_requires_train_mask():
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
        PoissonLearningMethod().fit(data)


def test_poisson_learning_predict_proba_requires_fit():
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
        PoissonLearningMethod().predict_proba(data)
