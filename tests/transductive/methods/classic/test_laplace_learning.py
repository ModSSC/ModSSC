from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

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


def test_laplace_solver_is_fail_closed() -> None:
    mod = _assert_module_importable("modssc.transductive.methods.classic.laplace_learning")
    mod._validate_solver(mod.LaplaceLearningSpec())
    mod._validate_solver(mod.LaplaceLearningSpec(solver="calder2020_conjugate_gradient"))
    with pytest.raises(ValueError, match="Unknown Laplace solver"):
        mod._validate_solver(mod.LaplaceLearningSpec(solver="unknown"))


def _archived_calder_conjugate_gradient(
    matrix: sparse.spmatrix,
    rhs: np.ndarray,
    *,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float]:
    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    scale = 1.0 / np.sqrt(diagonal + 1.0e-10)
    preconditioned = sparse.diags(scale) @ matrix.astype(np.float64) @ sparse.diags(scale)
    transformed_rhs = scale[:, None] * np.asarray(rhs, dtype=np.float64)
    solution = np.zeros_like(transformed_rhs)
    residual = transformed_rhs - preconditioned @ solution
    direction = residual
    squared = np.sum(residual**2, axis=0)
    error = 1.0
    iteration = 0
    while error > tol and iteration < max_iter:
        iteration += 1
        product = preconditioned @ direction
        alpha = squared / np.sum(direction * product, axis=0)
        solution += alpha * direction
        residual -= alpha * product
        squared_new = np.sum(residual**2, axis=0)
        error = float(np.sqrt(np.sum(squared_new)))
        direction = residual + (squared_new / squared) * direction
        squared = squared_new
    return scale[:, None] * solution, iteration, error


def test_calder2020_solver_matches_archived_alias_and_stopping_rule() -> None:
    mod = _assert_module_importable("modssc.transductive.methods.classic.laplace_learning")
    matrix = sparse.csr_matrix(
        np.array(
            [
                [4.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 3.0],
            ],
            dtype=np.float64,
        )
    )
    rhs = np.array(
        [
            [1.0, 0.5],
            [0.25, 1.0],
            [0.5, 0.25],
        ],
        dtype=np.float64,
    )
    expected = _archived_calder_conjugate_gradient(
        matrix,
        rhs,
        tol=1.0e-5,
        max_iter=100,
    )
    actual = mod._calder2020_conjugate_gradient(
        matrix,
        rhs,
        tol=1.0e-5,
        max_iter=100,
    )

    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]

    stopped = mod._calder2020_conjugate_gradient(
        matrix,
        rhs,
        tol=1.0,
        max_iter=100,
    )
    np.testing.assert_array_equal(stopped[0], np.zeros_like(rhs))
    assert stopped[1:] == (0, 1.0)


def test_calder2020_solver_requires_matrix_rhs_for_every_class() -> None:
    mod = _assert_module_importable("modssc.transductive.methods.classic.laplace_learning")
    matrix = sparse.eye(2, format="csr")
    with pytest.raises(ValueError, match="shape"):
        mod._calder2020_conjugate_gradient(
            matrix,
            np.ones(2),
            tol=1.0e-5,
            max_iter=10,
        )
    with pytest.raises(ValueError, match="non-zero source for every class"):
        mod._calder2020_conjugate_gradient(
            matrix,
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            tol=1.0e-5,
            max_iter=10,
        )


def test_laplace_numpy_rejects_unknown_solver() -> None:
    mod = _assert_module_importable("modssc.transductive.methods.classic.laplace_learning")
    with pytest.raises(ValueError, match="Unknown Laplace solver"):
        mod.laplace_learning_numpy(
            n_nodes=2,
            edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
            edge_weight=np.ones(2, dtype=np.float32),
            y=np.array([0, 0], dtype=np.int64),
            labeled_mask=np.array([True, False]),
            spec=mod.LaplaceLearningSpec(solver="unknown"),  # type: ignore[arg-type]
        )


def test_laplace_numpy_solver_dispatch_is_fail_closed(monkeypatch) -> None:
    mod = _assert_module_importable("modssc.transductive.methods.classic.laplace_learning")
    # Exercise the solver dispatch's defensive guard independently from the
    # public solver validation performed at the entry point.
    monkeypatch.setattr(mod, "_validate_solver", lambda _spec: None)
    with pytest.raises(ValueError, match="Unknown Laplace solver"):
        mod.laplace_learning_numpy(
            n_nodes=2,
            edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
            edge_weight=np.ones(2, dtype=np.float32),
            y=np.array([0, 0], dtype=np.int64),
            labeled_mask=np.array([True, False]),
            spec=mod.LaplaceLearningSpec(solver="unknown"),  # type: ignore[arg-type]
        )


def test_laplace_numpy_dispatches_calder_solver() -> None:
    mod = _assert_module_importable("modssc.transductive.methods.classic.laplace_learning")
    result = mod.laplace_learning_numpy(
        n_nodes=2,
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_weight=np.ones(2, dtype=np.float64),
        y=np.array([0, 0], dtype=np.int64),
        labeled_mask=np.array([True, False]),
        spec=mod.LaplaceLearningSpec(
            solver="calder2020_conjugate_gradient",
        ),
    )

    np.testing.assert_array_equal(result.F, np.ones((2, 1), dtype=np.float64))
    assert result.n_iter == 1
    assert result.residual == pytest.approx(0.0)


def test_laplace_shared_graph_self_loops_cancel_from_harmonic_system() -> None:
    mod = _assert_module_importable("modssc.transductive.methods.classic.laplace_learning")
    edge_index = np.array(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=np.int64,
    )
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    labeled_mask = np.array([True, False, False, True])
    spec = mod.LaplaceLearningSpec(
        solver="calder2020_conjugate_gradient",
        cg_tol=1.0e-8,
        cg_max_iter=100,
    )

    without_loops = mod.laplace_learning_numpy(
        n_nodes=4,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=labels,
        labeled_mask=labeled_mask,
        spec=spec,
    )
    nodes = np.arange(4, dtype=np.int64)
    with_loops = mod.laplace_learning_numpy(
        n_nodes=4,
        edge_index=np.concatenate([edge_index, np.vstack([nodes, nodes])], axis=1),
        edge_weight=np.concatenate([edge_weight, np.full(4, 23.0, dtype=np.float64)]),
        y=labels,
        labeled_mask=labeled_mask,
        spec=spec,
    )

    np.testing.assert_array_equal(with_loops.F, without_loops.F)
    assert with_loops.n_iter == without_loops.n_iter
    assert with_loops.residual == without_loops.residual


def test_laplace_method_auto_backend_records_device_dispatch(monkeypatch) -> None:
    mod = _assert_module_importable("modssc.transductive.methods.classic.laplace_learning")
    observed = {}

    def fake_laplace_learning(**kwargs):
        observed.update(kwargs)
        return mod.DiffusionResult(
            F=np.ones((1, 1), dtype=np.float32),
            n_iter=0,
            residual=0.0,
        )

    monkeypatch.setattr(mod, "validate_node_dataset", lambda _data: None)
    monkeypatch.setattr(mod, "laplace_learning", fake_laplace_learning)
    data = SimpleNamespace(
        y=np.array([0], dtype=np.int64),
        graph=SimpleNamespace(
            edge_index=np.array([[0], [0]], dtype=np.int64),
            edge_weight=np.ones(1, dtype=np.float32),
        ),
        masks={"train_mask": np.array([True])},
    )

    method = mod.LaplaceLearningMethod(mod.LaplaceLearningSpec(backend="auto"))
    method.fit(data, device="cpu")

    assert method.diagnostics_["backend"] == "torch"
    assert observed["backend"] == "torch"
    assert observed["device"] == "cpu"

    explicit = mod.LaplaceLearningMethod(mod.LaplaceLearningSpec(backend="numpy"))
    explicit.fit(data)
    assert explicit.diagnostics_["backend"] == "numpy"
    assert observed["backend"] == "numpy"


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
    with pytest.raises(ValueError, match="NumPy/CPU-only"):
        mod.laplace_learning_torch(
            n_nodes=2,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([0, 1]),
            labeled_mask=torch.tensor([True, True]),
            spec=mod.LaplaceLearningSpec(
                solver="calder2020_conjugate_gradient",
            ),
        )
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
