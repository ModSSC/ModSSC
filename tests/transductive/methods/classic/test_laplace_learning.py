from __future__ import annotations

import importlib

import numpy as np
import pytest

try:
    import torch
except Exception:
    torch = None


def _assert_module_importable(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing.startswith("modssc"):
            raise
        pytest.skip(f"Optional dependency missing while importing {module_name}: {missing}")
    except Exception as exc:
        if exc.__class__.__name__ == "OptionalDependencyError" or 'pip install "modssc[' in str(
            exc
        ):
            pytest.skip(f"Optional dependency missing while importing {module_name}: {exc}")
        raise


def test_module_importable() -> None:
    _assert_module_importable("modssc.transductive.methods.classic.laplace_learning")


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_laplace_learning_torch_sparse_matches_numpy() -> None:
    mod = _assert_module_importable("modssc.transductive.methods.classic.laplace_learning")
    edge_index = np.array(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=np.int64,
    )
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    y = np.array([0, -1, -1, 1], dtype=np.int64)
    labeled_mask = np.array([True, False, False, True])
    spec = mod.LaplaceLearningSpec(cg_tol=1e-8, cg_max_iter=200)

    expected = mod.laplace_learning(
        n_nodes=4,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=spec,
        backend="numpy",
    )
    actual = mod.laplace_learning(
        n_nodes=4,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=spec,
        backend="torch",
        device="cpu",
    )

    np.testing.assert_allclose(actual.F, expected.F, atol=1e-5)


def test_laplace_learning_numpy_edge_cases(monkeypatch) -> None:
    mod = _assert_module_importable("modssc.transductive.methods.classic.laplace_learning")

    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_weight = np.ones(2, dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    labeled = np.array([True, True])
    res = mod.laplace_learning_numpy(
        n_nodes=2,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled,
    )
    np.testing.assert_allclose(res.F, np.eye(2, dtype=np.float32))

    with pytest.raises(ValueError, match="nonsingular"):
        mod.laplace_learning_numpy(
            n_nodes=2,
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_weight=None,
            y=np.array([0, 0], dtype=np.int64),
            labeled_mask=np.array([True, False]),
        )

    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    y = np.array([0, 1, 1], dtype=np.int64)
    labeled = np.array([True, False, True])
    res = mod.laplace_learning_numpy(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=np.ones(2, dtype=np.float32),
        y=y,
        labeled_mask=labeled,
    )
    assert res.F.shape == (3, 2)
    assert np.allclose(res.F[1, 1], 0.0)

    def fake_cg_negative(*args, **kwargs):
        return np.zeros(1, dtype=np.float32), -1

    monkeypatch.setattr(mod.sparse_linalg, "cg", fake_cg_negative)
    with pytest.raises(ValueError, match="sparse CG failed"):
        mod.laplace_learning_numpy(
            n_nodes=2,
            edge_index=edge_index,
            edge_weight=np.ones(2, dtype=np.float32),
            y=np.array([0, 0], dtype=np.int64),
            labeled_mask=np.array([True, False]),
        )

    def fake_cg_max_iter(*args, **kwargs):
        return np.ones(1, dtype=np.float32), 7

    monkeypatch.setattr(mod.sparse_linalg, "cg", fake_cg_max_iter)
    res = mod.laplace_learning_numpy(
        n_nodes=2,
        edge_index=edge_index,
        edge_weight=np.ones(2, dtype=np.float32),
        y=np.array([0, 0], dtype=np.int64),
        labeled_mask=np.array([True, False]),
    )
    assert res.n_iter == 7

    def fake_cg_nan(*args, **kwargs):
        return np.array([np.nan], dtype=np.float32), 0

    monkeypatch.setattr(mod.sparse_linalg, "cg", fake_cg_nan)
    with pytest.raises(ValueError, match="nonsingular"):
        mod.laplace_learning_numpy(
            n_nodes=2,
            edge_index=edge_index,
            edge_weight=np.ones(2, dtype=np.float32),
            y=np.array([0, 0], dtype=np.int64),
            labeled_mask=np.array([True, False]),
        )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_laplace_learning_torch_edge_cases(monkeypatch) -> None:
    mod = _assert_module_importable("modssc.transductive.methods.classic.laplace_learning")

    with pytest.raises(ValueError, match="b must have shape"):
        mod._cg_solve_torch_multi_rhs(
            matvec=lambda x: x,
            b=torch.ones(2, dtype=torch.float32),
            tol=1e-6,
            max_iter=1,
        )

    x, info = mod._cg_solve_torch_multi_rhs(
        matvec=lambda x: x,
        b=torch.zeros((2, 1), dtype=torch.float32),
        tol=1e-6,
        max_iter=5,
    )
    assert info["converged"] is True
    assert info["n_iter"] == 0
    torch.testing.assert_close(x, torch.zeros((2, 1), dtype=torch.float32))

    _x, info = mod._cg_solve_torch_multi_rhs(
        matvec=lambda x: torch.zeros_like(x),
        b=torch.ones((2, 1), dtype=torch.float32),
        tol=1e-6,
        max_iter=5,
    )
    assert info["converged"] is False
    assert info["n_iter"] == 0
    _x, info = mod._cg_solve_torch_multi_rhs(
        matvec=lambda x: x,
        b=torch.ones((2, 1), dtype=torch.float32),
        tol=1e-6,
        max_iter=0,
    )
    assert info["converged"] is False
    assert info["n_iter"] == 0
    x, info = mod._cg_solve_torch_multi_rhs(
        matvec=lambda x: x,
        b=torch.ones((2, 1), dtype=torch.float32),
        tol=1e-6,
        max_iter=5,
    )
    assert info["converged"] is True
    assert info["n_iter"] == 1
    torch.testing.assert_close(x, torch.ones((2, 1), dtype=torch.float32))

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.ones(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="edge_index must have shape"):
        mod.laplace_learning_torch(
            n_nodes=2,
            edge_index=torch.zeros((3, 1), dtype=torch.long),
            edge_weight=edge_weight,
            y=torch.tensor([0, 0]),
            labeled_mask=torch.tensor([True, False]),
        )
    with pytest.raises(ValueError, match="edge_weight must have shape"):
        mod.laplace_learning_torch(
            n_nodes=2,
            edge_index=edge_index,
            edge_weight=torch.ones(1),
            y=torch.tensor([0, 0]),
            labeled_mask=torch.tensor([True, False]),
        )
    with pytest.raises(ValueError, match="shape"):
        mod.laplace_learning_torch(
            n_nodes=2,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([0]),
            labeled_mask=torch.tensor([True, False]),
        )
    with pytest.raises(ValueError, match="at least 1 labeled"):
        mod.laplace_learning_torch(
            n_nodes=2,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([0, 0]),
            labeled_mask=torch.tensor([False, False]),
        )
    res = mod.laplace_learning_torch(
        n_nodes=2,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor([0, 1]),
        labeled_mask=torch.tensor([True, True]),
    )
    np.testing.assert_allclose(res.F, np.eye(2, dtype=np.float32))

    with pytest.raises(ValueError, match="nonsingular"):
        mod.laplace_learning_torch(
            n_nodes=2,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_weight=None,
            y=torch.tensor([0, 0]),
            labeled_mask=torch.tensor([True, False]),
        )

    original_spmm = mod.spmm_torch

    def fake_cg_calls_vector(*, matvec, b, tol, max_iter):
        out = matvec(torch.zeros(int(b.shape[0]), dtype=torch.float32))
        assert int(out.dim()) == 1
        return torch.zeros_like(b), {"converged": True, "n_iter": 1, "residual_norm": 0.0}

    monkeypatch.setattr(mod, "_cg_solve_torch_multi_rhs", fake_cg_calls_vector)
    mod.laplace_learning_torch(
        n_nodes=2,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor([0, 0]),
        labeled_mask=torch.tensor([True, False]),
        spec=mod.LaplaceLearningSpec(cg_max_iter=3),
    )

    def fake_zero_spmm(*, n_nodes, edge_index, edge_weight, X):
        return torch.zeros_like(X)

    monkeypatch.setattr(mod, "spmm_torch", fake_zero_spmm)
    res = mod.laplace_learning_torch(
        n_nodes=2,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor([0, 0]),
        labeled_mask=torch.tensor([True, False]),
    )
    assert res.F.shape == (2, 1)
    monkeypatch.setattr(mod, "spmm_torch", original_spmm)

    def fake_cg_not_converged(*args, **kwargs):
        b = kwargs["b"]
        return torch.zeros_like(b), {"converged": False, "n_iter": 3, "residual_norm": 1.0}

    monkeypatch.setattr(mod, "_cg_solve_torch_multi_rhs", fake_cg_not_converged)
    mod.laplace_learning_torch(
        n_nodes=2,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor([0, 0]),
        labeled_mask=torch.tensor([True, False]),
        spec=mod.LaplaceLearningSpec(cg_max_iter=3),
    )

    def fake_cg_nan(*args, **kwargs):
        b = kwargs["b"]
        return torch.full_like(b, float("nan")), {
            "converged": True,
            "n_iter": 1,
            "residual_norm": 0.0,
        }

    monkeypatch.setattr(mod, "_cg_solve_torch_multi_rhs", fake_cg_nan)
    with pytest.raises(ValueError, match="nonsingular"):
        mod.laplace_learning_torch(
            n_nodes=2,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([0, 0]),
            labeled_mask=torch.tensor([True, False]),
        )
