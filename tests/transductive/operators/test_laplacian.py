from unittest.mock import MagicMock, patch

import numpy as np

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
