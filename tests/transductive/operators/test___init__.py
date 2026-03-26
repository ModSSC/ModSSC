from unittest.mock import patch

from modssc.transductive.operators.normalize import normalize_edges_torch
from modssc.transductive.operators.spmm import spmm_torch
from modssc.transductive.types import DeviceSpec


def test_normalize_edges_torch():
    with (
        patch("modssc.transductive.backends.torch_backend.resolve_device") as mock_resolve,
        patch("modssc.transductive.backends.torch_backend.dtype_from_spec") as mock_dtype,
        patch("modssc.transductive.backends.torch_backend.normalize_edges") as mock_normalize,
    ):
        mock_resolve.return_value = "cuda:0"
        mock_dtype.return_value = "float32"
        mock_normalize.return_value = "normalized_weights"

        result = normalize_edges_torch(
            n_nodes=10,
            edge_index="edge_index",
            edge_weight="edge_weight",
            mode="sym",
            device=DeviceSpec(device="cuda"),
        )

        assert result == "normalized_weights"
        mock_resolve.assert_called_once()
        mock_dtype.assert_called_once()
        mock_normalize.assert_called_once_with(
            n_nodes=10,
            edge_index="edge_index",
            edge_weight="edge_weight",
            mode="sym",
            device="cuda:0",
            dtype="float32",
        )


def test_spmm_torch():
    with (
        patch("modssc.transductive.backends.torch_backend.resolve_device") as mock_resolve,
        patch("modssc.transductive.backends.torch_backend.dtype_from_spec") as mock_dtype,
        patch("modssc.transductive.backends.torch_backend.to_tensor") as mock_to_tensor,
        patch("modssc.transductive.backends.torch_backend.spmm") as mock_spmm,
    ):
        mock_resolve.return_value = "cpu"
        mock_dtype.return_value = "float32"
        mock_to_tensor.return_value = "tensor_X"
        mock_spmm.return_value = "spmm_result"

        result = spmm_torch(
            n_nodes=5,
            edge_index="edge_index",
            edge_weight="edge_weight",
            X="X",
            device=DeviceSpec(device="cpu"),
        )

        assert result == "spmm_result"
        mock_resolve.assert_called_once()
        mock_dtype.assert_called_once()
        mock_to_tensor.assert_called_once()
        mock_spmm.assert_called_once()
