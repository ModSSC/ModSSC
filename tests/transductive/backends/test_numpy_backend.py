import numpy as np
import pytest

from modssc.transductive.backends.numpy_backend import (
    _ensure_edge_weight,
    cg_solve,
    normalize_edges,
    spmm,
)
from modssc.transductive.operators.normalize import normalize_edges_numpy
from modssc.transductive.operators.spmm import spmm_numpy


def test_ensure_edge_weight_error():
    with pytest.raises(ValueError, match="edge_weight must have shape"):
        _ensure_edge_weight(np.array([[1, 2]]), E=2)

    with pytest.raises(ValueError, match="edge_weight must have shape"):
        _ensure_edge_weight(np.array([1, 2, 3]), E=2)


def test_spmm_empty():
    n_nodes = 5
    edge_index = np.array([[], []], dtype=np.int64)
    X = np.random.randn(n_nodes, 3).astype(np.float32)

    out = spmm(n_nodes=n_nodes, edge_index=edge_index, edge_weight=None, X=X)

    assert out.shape == X.shape
    assert np.all(out == 0)


def test_spmm_1d_features():
    n_nodes = 3

    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    X = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    expected = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    out = spmm(n_nodes=n_nodes, edge_index=edge_index, edge_weight=None, X=X)

    np.testing.assert_allclose(out, expected)


def test_spmm_2d_features():
    n_nodes = 3

    edge_index = np.array([[0], [1]], dtype=np.int64)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    expected = np.zeros_like(X)
    expected[1] = [1.0, 2.0]

    out = spmm(n_nodes=n_nodes, edge_index=edge_index, edge_weight=None, X=X)

    np.testing.assert_allclose(out, expected)


def test_spmm_numpy_simple():
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    X = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    out = spmm_numpy(n_nodes=3, edge_index=edge_index, edge_weight=None, X=X)
    np.testing.assert_allclose(out, np.array([[0.0], [1.0], [2.0]], dtype=np.float32))


def test_normalize_edges_empty():
    n_nodes = 5
    edge_index = np.array([[], []], dtype=np.int64)

    w = normalize_edges(n_nodes=n_nodes, edge_index=edge_index, edge_weight=None, mode="sym")

    assert w.size == 0
    assert w.dtype == np.float32


def test_normalize_edges_modes():
    n_nodes = 3

    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    edge_weight = np.array([1.0, 1.0], dtype=np.float32)

    w_none = normalize_edges(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="none"
    )
    np.testing.assert_allclose(w_none, edge_weight)

    w_rw = normalize_edges(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="rw"
    )
    np.testing.assert_allclose(w_rw, [1.0, 1.0])

    w_sym = normalize_edges(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="sym"
    )
    assert np.all(np.isfinite(w_sym))


def test_normalize_edges_invalid_mode():
    n_nodes = 3
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)

    with pytest.raises(ValueError, match="Unknown normalization mode"):
        normalize_edges(n_nodes=n_nodes, edge_index=edge_index, edge_weight=None, mode="invalid")


def test_normalize_edges_numpy_rw_row_sums():
    edge_index = np.array([[0, 2], [1, 1]], dtype=np.int64)
    w = np.array([1.0, 3.0], dtype=np.float32)
    wn = normalize_edges_numpy(n_nodes=3, edge_index=edge_index, edge_weight=w, mode="rw")
    np.testing.assert_allclose(wn, np.array([0.25, 0.75], dtype=np.float32))


def test_cg_solve_simple_system():
    def matvec(x):
        return x

    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    x, info = cg_solve(matvec=matvec, b=b, tol=1e-6)

    assert info["converged"]
    np.testing.assert_allclose(x, b, atol=1e-5)


def test_cg_solve_with_x0():
    def matvec(x):
        return x

    b = np.array([1.0, 1.0], dtype=np.float32)
    x0 = np.array([0.9, 0.9], dtype=np.float32)

    x, info = cg_solve(matvec=matvec, b=b, x0=x0, tol=1e-6)

    assert info["converged"]
    np.testing.assert_allclose(x, b, atol=1e-5)


def test_cg_solve_already_converged():
    def matvec(x):
        return x

    b = np.zeros(3, dtype=np.float32)

    x, info = cg_solve(matvec=matvec, b=b)

    assert info["converged"]
    assert info["n_iter"] == 0
    assert info["residual_norm"] == 0.0
    np.testing.assert_allclose(x, b)


def test_cg_solve_max_iter():
    def matvec(x):
        return 2.0 * x

    b = np.array([2.0, 2.0], dtype=np.float32)

    x, info = cg_solve(matvec=matvec, b=b, max_iter=0)

    assert info["n_iter"] == 0
    assert not info["converged"]


def test_cg_solve_converges_in_steps():
    def matvec(x):
        return x * np.array([1.0, 10.0])

    b = np.array([1.0, 10.0], dtype=np.float32)

    x, info = cg_solve(matvec=matvec, b=b, tol=1e-6)

    assert info["converged"]
    assert info["n_iter"] <= 2
    np.testing.assert_allclose(x, np.array([1.0, 1.0]), atol=1e-5)


def test_cg_solve_break_denom_zero():
    def matvec(x):
        return np.zeros_like(x)

    b = np.array([1.0], dtype=np.float32)

    x, info = cg_solve(matvec=matvec, b=b)

    assert not info["converged"]
