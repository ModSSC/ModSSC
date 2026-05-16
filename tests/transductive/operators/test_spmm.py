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
    _assert_module_importable("modssc.transductive.operators.spmm")


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torch_spmm_matches_numpy_without_sparse_coo() -> None:
    from modssc.transductive.backends import torch_backend

    edge_index = np.array([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=np.int64)
    edge_weight = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    x_vec = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    out_vec = torch_backend.spmm(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        X=x_vec,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    np.testing.assert_allclose(out_vec.numpy(), np.array([12.0, 2.0, 11.0], dtype=np.float32))

    x_mat = torch.stack([x_vec, x_vec + 10.0], dim=1)
    out_mat = torch_backend.spmm(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        X=x_mat,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = np.array([[12.0, 52.0], [2.0, 22.0], [11.0, 91.0]], dtype=np.float32)
    np.testing.assert_allclose(out_mat.numpy(), expected)
