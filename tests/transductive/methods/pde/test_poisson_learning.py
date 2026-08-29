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


def test_poisson_learning_spec_defaults():
    spec = PoissonLearningSpec()
    assert spec.backend == "numpy"
    assert spec.solver == "conjugate_gradient"
    assert spec.laplacian_kind == "paper_normalized"
    assert spec.eps == pytest.approx(0.0)
    assert spec.center_sources is True
    assert spec.tol == pytest.approx(1e-3)
    assert spec.min_iter == 50
    assert spec.max_iter == 1000


def test_poisson_solver_is_fail_closed() -> None:
    pl._validate_solver(PoissonLearningSpec())
    pl._validate_solver(PoissonLearningSpec(solver="paper_iteration"))
    with pytest.raises(ValueError, match="Unknown Poisson solver"):
        pl._validate_solver(PoissonLearningSpec(solver="unknown"))


def test_poisson_paper_iteration_matches_reference_recurrence():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labeled_mask = np.array([True, False, False, True, False, False])

    spec = PoissonLearningSpec(
        backend="numpy",
        solver="paper_iteration",
        laplacian_kind="unnormalized",
        min_iter=5,
        max_iter=5,
    )
    result = pl.poisson_learning_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=spec,
    )

    W = np.zeros((n, n), dtype=np.float64)
    for src, dst, weight in zip(edge_index[0], edge_index[1], edge_weight, strict=True):
        if src != dst:
            W[int(dst), int(src)] += float(weight)
    degree = W.sum(axis=1)
    onehot = np.eye(2, dtype=np.float64)[y]
    source = np.zeros_like(onehot)
    source[labeled_mask] = onehot[labeled_mask] - onehot[labeled_mask].mean(axis=0)
    expected = np.zeros_like(source)
    inverse_degree_denominator = degree + 1.0e-10
    for _ in range(5):
        expected = (
            source / inverse_degree_denominator[:, None]
            + (W @ expected) / inverse_degree_denominator[:, None]
        )

    np.testing.assert_allclose(result.F, expected, rtol=0.0, atol=1.0e-14)
    assert result.n_iter == 5
    assert result.F.argmax(axis=1).tolist() == y.tolist()


def test_poisson_paper_iteration_ignores_shared_graph_self_loops() -> None:
    n, edge_index, edge_weight = _two_cluster_graph()
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labeled_mask = np.array([True, False, False, True, False, False])
    spec = PoissonLearningSpec(
        solver="paper_iteration",
        min_iter=7,
        max_iter=7,
    )

    without_loops = pl.poisson_learning_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=labels,
        labeled_mask=labeled_mask,
        spec=spec,
    )
    nodes = np.arange(n, dtype=np.int64)
    with_loops = pl.poisson_learning_numpy(
        n_nodes=n,
        edge_index=np.concatenate([edge_index, np.vstack([nodes, nodes])], axis=1),
        edge_weight=np.concatenate([edge_weight, np.full(n, 23.0, dtype=np.float32)]),
        y=labels,
        labeled_mask=labeled_mask,
        spec=spec,
    )

    np.testing.assert_array_equal(with_loops.F, without_loops.F)
    assert with_loops.n_iter == without_loops.n_iter
    assert with_loops.residual == without_loops.residual


def test_poisson_paper_decision_rule_matches_graphlearning_training_balance() -> None:
    scores = np.array([[2.0, 4.0], [1.0, 3.0]], dtype=np.float64)
    onehot = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    labeled_mask = np.array([True, True, True, False])

    unchanged = pl._apply_paper_decision_rule(
        scores,
        Y_labeled=onehot,
        labeled_mask=labeled_mask,
        spec=PoissonLearningSpec(balance_scores=False),
    )
    assert unchanged is scores

    archived_default = pl._apply_paper_decision_rule(
        scores,
        Y_labeled=onehot,
        labeled_mask=labeled_mask,
        spec=PoissonLearningSpec(balance_scores=True),
    )
    np.testing.assert_array_equal(
        archived_default,
        scores * np.array([1.5, 3.0])[None, :],
    )

    explicit_priors = pl._apply_paper_decision_rule(
        scores,
        Y_labeled=onehot,
        labeled_mask=labeled_mask,
        spec=PoissonLearningSpec(
            balance_scores=True,
            class_priors=(3.0, 1.0),
        ),
    )
    np.testing.assert_array_equal(
        explicit_priors,
        scores * np.array([1.125, 0.75])[None, :],
    )


@pytest.mark.parametrize(
    ("priors", "message"),
    [
        ((1.0,), "one value per labeled class"),
        ((1.0, 0.0), "finite and strictly positive"),
        ((1.0, np.nan), "finite and strictly positive"),
    ],
)
def test_poisson_paper_decision_rule_rejects_invalid_priors(priors, message) -> None:
    scores = np.ones((2, 2), dtype=np.float64)
    onehot = np.eye(2, dtype=np.float64)
    labeled_mask = np.ones(2, dtype=bool)
    with pytest.raises(ValueError, match=message):
        pl._apply_paper_decision_rule(
            scores,
            Y_labeled=onehot,
            labeled_mask=labeled_mask,
            spec=PoissonLearningSpec(
                balance_scores=True,
                class_priors=priors,
            ),
        )


def test_poisson_paper_decision_rule_requires_every_class_in_labels() -> None:
    with pytest.raises(ValueError, match="at least one label from every class"):
        pl._apply_paper_decision_rule(
            np.ones((2, 2), dtype=np.float64),
            Y_labeled=np.array([[1.0, 0.0], [0.0, 0.0]]),
            labeled_mask=np.ones(2, dtype=bool),
            spec=PoissonLearningSpec(balance_scores=True),
        )


@pytest.mark.parametrize(
    ("min_iter", "max_iter", "message"),
    [(-1, 2, "non-negative"), (0, 0, "positive"), (3, 2, "less than")],
)
def test_poisson_paper_iteration_validates_iteration_bounds(min_iter, max_iter, message):
    n, edge_index, edge_weight = _two_cluster_graph()
    with pytest.raises(ValueError, match=message):
        pl.poisson_learning_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=np.array([0, 0, 0, 1, 1, 1]),
            labeled_mask=np.array([True, False, False, True, False, False]),
            spec=PoissonLearningSpec(
                solver="paper_iteration",
                min_iter=min_iter,
                max_iter=max_iter,
            ),
        )


def test_poisson_paper_iteration_rejects_isolated_vertices_and_torch():
    with pytest.raises(ValueError, match="strictly positive degrees"):
        pl.poisson_learning_numpy(
            n_nodes=2,
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_weight=np.zeros((0,), dtype=np.float32),
            y=np.array([0, 1]),
            labeled_mask=np.array([True, True]),
            spec=PoissonLearningSpec(
                solver="paper_iteration",
                min_iter=0,
            ),
        )

    if torch is not None:
        with pytest.raises(ValueError, match="NumPy/CPU-only"):
            pl.poisson_learning_torch(
                n_nodes=2,
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                edge_weight=torch.ones(2),
                y=torch.tensor([0, 1]),
                labeled_mask=torch.tensor([True, True]),
                spec=PoissonLearningSpec(
                    solver="paper_iteration",
                ),
            )


def test_poisson_numpy_rejects_unknown_solver() -> None:
    n, edge_index, edge_weight = _two_cluster_graph()
    spec = PoissonLearningSpec(solver="unknown")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown Poisson solver"):
        pl.poisson_learning_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=np.array([0, 0, 0, 1, 1, 1]),
            labeled_mask=np.array([True, False, False, True, False, False]),
            spec=spec,
        )


def test_poisson_numpy_solver_dispatch_is_fail_closed(monkeypatch) -> None:
    n, edge_index, edge_weight = _two_cluster_graph()
    spec = PoissonLearningSpec(solver="unknown")  # type: ignore[arg-type]
    # Exercise the solver dispatch's defensive guard independently from the
    # public solver validation performed at the entry point.
    monkeypatch.setattr(pl, "_validate_solver", lambda _spec: None)

    with pytest.raises(ValueError, match="Unknown Poisson solver"):
        pl.poisson_learning_numpy(
            n_nodes=n,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=np.array([0, 0, 0, 1, 1, 1]),
            labeled_mask=np.array([True, False, False, True, False, False]),
            spec=spec,
        )


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


def test_poisson_learning_numpy_unnormalized_positive_degree():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[0] = True
    labeled_mask[3] = True

    res = pl.poisson_learning_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonLearningSpec(laplacian_kind="unnormalized", eps=0.0, max_iter=500),
    )

    assert res.F.shape == (n, 2)


def test_poisson_learning_paper_normalized_matches_normalized_transform():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[0] = True
    labeled_mask[3] = True

    res = pl.poisson_learning_numpy(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonLearningSpec(
            backend="numpy",
            laplacian_kind="paper_normalized",
            eps=0.0,
            tol=1e-10,
            max_iter=2000,
        ),
    )

    W = np.zeros((n, n), dtype=np.float64)
    for src, dst, w in zip(edge_index[0], edge_index[1], edge_weight, strict=True):
        W[int(dst), int(src)] += float(w)
    deg = W.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    inv_sqrt[deg > 0.0] = 1.0 / np.sqrt(deg[deg > 0.0])
    S = inv_sqrt[:, None] * W * inv_sqrt[None, :]
    L = np.eye(n) - S

    Y = np.eye(2, dtype=np.float64)[y]
    Y[~labeled_mask] = 0.0
    source = np.zeros_like(Y)
    source[labeled_mask] = Y[labeled_mask] - Y[labeled_mask].mean(axis=0)
    rhs = inv_sqrt[:, None] * source

    null_vec = np.sqrt(deg)
    A = L + np.outer(null_vec, null_vec) / float(np.dot(null_vec, null_vec))
    expected = np.zeros_like(rhs)
    for c in range(rhs.shape[1]):
        expected[:, c] = inv_sqrt * np.linalg.solve(A, rhs[:, c])

    np.testing.assert_allclose(res.F, expected, atol=2e-4)


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


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_poisson_learning_torch_unnormalized_positive_degree():
    n, edge_index, edge_weight = _two_cluster_graph()
    y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    labeled_mask = torch.zeros((n,), dtype=torch.bool)
    labeled_mask[0] = True
    labeled_mask[3] = True

    res = pl.poisson_learning_torch(
        n_nodes=n,
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_weight=torch.as_tensor(edge_weight, dtype=torch.float32),
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonLearningSpec(laplacian_kind="unnormalized", eps=0.0, max_iter=500),
        device="cpu",
    )

    assert res.F.shape == (n, 2)


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
    assert method.diagnostics_["solver"] == "conjugate_gradient"
    assert method.diagnostics_["converged"] is True


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

    monkeypatch.setattr(pl, "laplacian_matvec_numpy", lambda **_: np.zeros_like)

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


def test_poisson_learning_numpy_zero_degree_nullspace_fallbacks(monkeypatch):
    y = np.array([0, 1], dtype=np.int64)
    labeled_mask = np.array([True, True])
    edge_index = np.zeros((2, 0), dtype=np.int64)
    edge_weight = np.zeros((0,), dtype=np.float32)

    monkeypatch.setattr(pl, "laplacian_matvec_numpy", lambda **_: np.zeros_like)

    def fake_cg(matvec, b, tol, max_iter):
        matvec(np.zeros_like(b))
        return SimpleNamespace(x=np.zeros_like(b), n_iter=1, residual_norm=0.0)

    monkeypatch.setattr(pl, "cg_solve_numpy", fake_cg)

    for kind in ("paper_normalized", "unnormalized"):
        res = pl.poisson_learning_numpy(
            n_nodes=2,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PoissonLearningSpec(laplacian_kind=kind, eps=0.0),
        )
        assert res.F.shape == (2, 2)


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

    monkeypatch.setattr(pl, "laplacian_matvec_torch", lambda **_: torch.zeros_like)

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


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_poisson_learning_torch_zero_degree_nullspace_fallbacks(monkeypatch):
    y = torch.tensor([0, 1], dtype=torch.long)
    labeled_mask = torch.tensor([True, True], dtype=torch.bool)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_weight = torch.zeros((0,), dtype=torch.float32)

    monkeypatch.setattr(pl, "laplacian_matvec_torch", lambda **_: torch.zeros_like)

    def fake_cg(matvec, b, device, tol, max_iter):
        matvec(torch.zeros_like(b))
        return torch.zeros_like(b), {"n_iter": 1, "residual_norm": 0.0}

    monkeypatch.setattr(pl, "cg_solve_torch", fake_cg)

    for kind in ("paper_normalized", "unnormalized"):
        res = pl.poisson_learning_torch(
            n_nodes=2,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=PoissonLearningSpec(laplacian_kind=kind, eps=0.0),
            device="cpu",
        )
        assert res.F.shape == (2, 2)


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
