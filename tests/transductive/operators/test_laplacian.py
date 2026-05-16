from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.transductive.backends import torch_backend
from modssc.transductive.operators.laplacian import laplacian_matvec_numpy, laplacian_matvec_torch
from modssc.transductive.types import DeviceSpec


def test_laplacian_matvec_numpy_sym():
    n_nodes = 2
    edge_index = np.array([[0, 1], [1, 0]])
    edge_weight = np.array([1.0, 1.0])

    matvec = laplacian_matvec_numpy(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, kind="sym"
    )

    x = np.array([[1.0], [0.0]])
    res = matvec(x)
    np.testing.assert_allclose(res, np.array([[1.0], [-1.0]]), atol=1e-6)


def test_laplacian_matvec_numpy_rw():
    n_nodes = 2
    edge_index = np.array([[0, 1], [1, 0]])
    edge_weight = np.array([1.0, 1.0])

    matvec = laplacian_matvec_numpy(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, kind="rw"
    )

    x = np.array([[1.0], [0.0]])
    res = matvec(x)
    np.testing.assert_allclose(res, np.array([[1.0], [-1.0]]), atol=1e-6)


def test_laplacian_matvec_numpy_unnormalized():
    n_nodes = 3
    edge_index = np.array([[0, 2, 1], [1, 1, 2]])
    edge_weight = np.array([2.0, 3.0, 4.0])

    matvec = laplacian_matvec_numpy(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        kind="unnormalized",
    )

    x = np.array([[1.0], [2.0], [3.0]])
    res = matvec(x)
    np.testing.assert_allclose(res, np.array([[0.0], [-1.0], [4.0]]), atol=1e-6)

    res_1d = matvec(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(res_1d, np.array([0.0, -1.0, 4.0]), atol=1e-6)


def test_laplacian_numpy_edge_weight_validation_and_empty_degree():
    matvec = laplacian_matvec_numpy(
        n_nodes=2,
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_weight=None,
        kind="unnormalized",
    )
    np.testing.assert_allclose(
        matvec(np.ones((2,), dtype=np.float32)), np.zeros((2,), dtype=np.float32)
    )

    with np.testing.assert_raises_regex(ValueError, "edge_weight must have shape"):
        laplacian_matvec_numpy(
            n_nodes=2,
            edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
            edge_weight=np.array([1.0], dtype=np.float32),
            kind="unnormalized",
        )


def test_laplacian_matvec_torch():
    with (
        patch.object(torch_backend, "_torch") as mock_get_torch,
        patch.object(torch_backend, "resolve_device") as mock_resolve,
        patch.object(torch_backend, "dtype_from_spec") as mock_dtype,
        patch.object(torch_backend, "spmm") as mock_spmm,
        patch("modssc.transductive.operators.laplacian.normalize_edges_torch") as mock_norm,
    ):
        mock_torch = MagicMock()
        mock_get_torch.return_value = mock_torch
        mock_resolve.return_value = "cpu_device"
        mock_dtype.return_value = "float32"

        mock_norm.return_value = "normalized_weights"
        mock_spmm.return_value = "spmm_result"

        mock_x = MagicMock()
        mock_x.__sub__.return_value = "final_result"

        matvec = laplacian_matvec_torch(
            n_nodes=3, edge_index="ei", edge_weight="ew", device=DeviceSpec(device="cpu")
        )

        res = matvec(mock_x)

        assert res == "final_result"

        mock_resolve.assert_called()
        mock_dtype.assert_called()
        mock_norm.assert_called_with(
            n_nodes=3,
            edge_index="ei",
            edge_weight="ew",
            mode="sym",
            device=DeviceSpec(device="cpu"),
        )
        mock_spmm.assert_called_with(
            n_nodes=3,
            edge_index="ei",
            edge_weight="normalized_weights",
            X=mock_x,
            device="cpu_device",
            dtype="float32",
        )
        mock_x.__sub__.assert_called_with("spmm_result")


def test_laplacian_matvec_torch_unnormalized_real_tensors():
    torch = pytest.importorskip("torch")
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)

    matvec = laplacian_matvec_torch(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        kind="unnormalized",
        device=DeviceSpec(device="cpu"),
    )
    torch.testing.assert_close(
        matvec(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)),
        torch.tensor([0.0, -1.0, 4.0], dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="edge_weight must have shape"):
        laplacian_matvec_torch(
            n_nodes=3,
            edge_index=edge_index,
            edge_weight=torch.ones(2, dtype=torch.float32),
            kind="unnormalized",
            device=DeviceSpec(device="cpu"),
        )

    matvec_empty = laplacian_matvec_torch(
        n_nodes=2,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_weight=None,
        kind="unnormalized",
        device=DeviceSpec(device="cpu"),
    )
    torch.testing.assert_close(
        matvec_empty(torch.ones((2, 1), dtype=torch.float32)),
        torch.zeros((2, 1), dtype=torch.float32),
    )
