from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.transductive.backends import torch_backend
from modssc.transductive.errors import OptionalDependencyError
from modssc.transductive.types import DeviceSpec


@pytest.fixture
def mock_torch():
    with patch("modssc.transductive.backends.torch_backend._torch") as mock_import:
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        yield mock_module


def test_torch_import():
    with patch("modssc.transductive.backends.torch_backend.optional_import") as mock_opt:
        torch_backend._torch()
        mock_opt.assert_called_with("torch", extra="transductive-torch")


def test_resolve_device_cpu(mock_torch):
    spec = DeviceSpec(device="cpu")
    mock_torch.device.return_value = "cpu_device"
    dev = torch_backend.resolve_device(spec)
    assert dev == "cpu_device"


def test_resolve_device_cuda(mock_torch):
    spec = DeviceSpec(device="cuda")
    mock_torch.cuda.is_available.return_value = True
    mock_torch.device.return_value = "cuda_device"
    dev = torch_backend.resolve_device(spec)
    assert dev == "cuda_device"


def test_resolve_device_cuda_unavailable(mock_torch):
    spec = DeviceSpec(device="cuda")
    mock_torch.cuda.is_available.return_value = False
    with pytest.raises(OptionalDependencyError):
        torch_backend.resolve_device(spec)


def test_resolve_device_mps(mock_torch):
    spec = DeviceSpec(device="mps")
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.device.return_value = "mps_device"
    dev = torch_backend.resolve_device(spec)
    assert dev == "mps_device"


def test_resolve_device_mps_unavailable(mock_torch):
    spec = DeviceSpec(device="mps")
    mock_torch.backends.mps.is_available.return_value = False
    with pytest.raises(OptionalDependencyError):
        torch_backend.resolve_device(spec)


def test_resolve_device_auto(mock_torch):
    spec = DeviceSpec(device="auto")
    mock_torch.cuda.is_available.return_value = True
    mock_torch.device.return_value = "cuda_device"
    assert torch_backend.resolve_device(spec) == "cuda_device"

    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.device.return_value = "mps_device"
    assert torch_backend.resolve_device(spec) == "mps_device"

    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.device.return_value = "cpu_device"
    assert torch_backend.resolve_device(spec) == "cpu_device"


def test_resolve_device_unknown(mock_torch):
    spec = DeviceSpec(device="tpu")
    with pytest.raises(ValueError):
        torch_backend.resolve_device(spec)


def test_dtype_from_spec(mock_torch):
    mock_torch.float32 = "float32"
    mock_torch.float64 = "float64"
    assert torch_backend.dtype_from_spec(DeviceSpec(dtype="float32")) == "float32"
    assert torch_backend.dtype_from_spec(DeviceSpec(dtype="float64")) == "float64"
    with pytest.raises(ValueError):
        torch_backend.dtype_from_spec(DeviceSpec(dtype="int8"))


def test_to_tensor(mock_torch):
    x_np = np.array([1, 2])
    mock_tensor = MagicMock()
    mock_torch.from_numpy.return_value = mock_tensor
    mock_tensor.to.return_value = mock_tensor
    torch_backend.to_tensor(x_np, device="cpu", dtype="float32")
    mock_torch.from_numpy.assert_called_with(x_np)

    x_list = [1, 2]
    mock_torch.as_tensor.return_value = mock_tensor
    torch_backend.to_tensor(x_list, device="cpu")
    mock_torch.as_tensor.assert_called_with(x_list)


def test_spmm(mock_torch):
    mock_edge_index = MagicMock()
    mock_edge_index.shape = [2, 3]
    mock_edge_index.__getitem__.side_effect = ["src", "dst"]

    mock_weights = MagicMock()
    mock_weights.reshape.return_value = mock_weights
    mock_weights.numel.return_value = 3

    mock_X = MagicMock()
    mock_X.ndim = 2
    mock_X.shape = [5, 10]

    with patch("modssc.transductive.backends.torch_backend.to_tensor") as mock_to_tensor:
        mock_to_tensor.side_effect = [mock_edge_index, mock_weights]
        torch_backend.spmm(
            n_nodes=5, edge_index="ei", edge_weight="ew", X=mock_X, device="cpu", dtype="float32"
        )
        mock_torch.sparse_coo_tensor.assert_called()

        mock_edge_index_empty = MagicMock()
        mock_edge_index_empty.shape = [2, 0]
        mock_to_tensor.side_effect = [mock_edge_index_empty]
        mock_torch.zeros.return_value = "zeros"
        res = torch_backend.spmm(
            n_nodes=5, edge_index="ei", edge_weight=None, X=mock_X, device="cpu", dtype="float32"
        )
        assert res == "zeros"

        mock_to_tensor.side_effect = [mock_edge_index]
        mock_edge_index.__getitem__.side_effect = ["src", "dst"]
        mock_torch.ones.return_value = "ones"
        torch_backend.spmm(
            n_nodes=5, edge_index="ei", edge_weight=None, X=mock_X, device="cpu", dtype="float32"
        )
        mock_torch.ones.assert_called()

        mock_X_1d = MagicMock()
        mock_X_1d.ndim = 1
        mock_to_tensor.side_effect = [mock_edge_index, mock_weights]
        mock_edge_index.__getitem__.side_effect = ["src", "dst"]
        torch_backend.spmm(
            n_nodes=5, edge_index="ei", edge_weight="ew", X=mock_X_1d, device="cpu", dtype="float32"
        )
        mock_torch.sparse.mm.assert_called()

        mock_to_tensor.side_effect = [mock_edge_index, mock_weights]
        mock_edge_index.__getitem__.side_effect = ["src", "dst"]
        mock_weights.numel.return_value = 999
        with pytest.raises(ValueError):
            torch_backend.spmm(
                n_nodes=5,
                edge_index="ei",
                edge_weight="ew",
                X=mock_X,
                device="cpu",
                dtype="float32",
            )
        mock_weights.numel.return_value = 3


def test_normalize_edges(mock_torch):
    mock_edge_index = MagicMock()
    mock_edge_index.shape = [2, 3]
    mock_edge_index.__getitem__.side_effect = ["src", "dst"]

    mock_weights = MagicMock()
    mock_weights.reshape.return_value = mock_weights
    mock_weights.numel.return_value = 3

    mock_deg = MagicMock()
    mock_torch.zeros.return_value = mock_deg
    mock_deg.__getitem__.return_value = "deg_vals"

    mock_torch.ones.return_value = "ones"

    with patch("modssc.transductive.backends.torch_backend.to_tensor") as mock_to_tensor:
        mock_to_tensor.side_effect = [mock_edge_index, mock_weights]
        res = torch_backend.normalize_edges(
            n_nodes=5, edge_index="ei", edge_weight="ew", mode="none", device="cpu", dtype="float32"
        )
        assert res is mock_weights

        mock_to_tensor.side_effect = [mock_edge_index]
        mock_edge_index.__getitem__.side_effect = ["src", "dst"]
        res = torch_backend.normalize_edges(
            n_nodes=5, edge_index="ei", edge_weight=None, mode="none", device="cpu", dtype="float32"
        )
        mock_torch.ones.assert_called()
        assert res == "ones"

        mock_edge_index_empty = MagicMock()
        mock_edge_index_empty.shape = [2, 0]
        mock_to_tensor.side_effect = [mock_edge_index_empty]
        res = torch_backend.normalize_edges(
            n_nodes=5, edge_index="ei", edge_weight=None, mode="none", device="cpu", dtype="float32"
        )
        assert res == "ones"

        mock_to_tensor.side_effect = [mock_edge_index, mock_weights]
        mock_edge_index.__getitem__.side_effect = ["src", "dst"]
        torch_backend.normalize_edges(
            n_nodes=5, edge_index="ei", edge_weight="ew", mode="rw", device="cpu", dtype="float32"
        )

        mock_to_tensor.side_effect = [mock_edge_index, mock_weights]
        mock_edge_index.__getitem__.side_effect = ["src", "dst"]
        torch_backend.normalize_edges(
            n_nodes=5, edge_index="ei", edge_weight="ew", mode="sym", device="cpu", dtype="float32"
        )

        mock_to_tensor.side_effect = [mock_edge_index, mock_weights]
        mock_edge_index.__getitem__.side_effect = ["src", "dst"]
        with pytest.raises(ValueError):
            torch_backend.normalize_edges(
                n_nodes=5,
                edge_index="ei",
                edge_weight="ew",
                mode="invalid",
                device="cpu",
                dtype="float32",
            )

        mock_to_tensor.side_effect = [mock_edge_index, mock_weights]
        mock_edge_index.__getitem__.side_effect = ["src", "dst"]
        mock_weights.numel.return_value = 999
        with pytest.raises(ValueError):
            torch_backend.normalize_edges(
                n_nodes=5,
                edge_index="ei",
                edge_weight="ew",
                mode="none",
                device="cpu",
                dtype="float32",
            )
        mock_weights.numel.return_value = 3


def test_cg_solve(mock_torch):
    b = MagicMock()
    mock_torch.zeros_like.return_value = MagicMock(name="x_init")
    matvec = MagicMock()
    matvec.return_value = MagicMock(name="Ap")

    mock_torch.dot.return_value.item.return_value = 0.0
    mock_torch.sqrt.return_value.item.return_value = 0.0
    x, info = torch_backend.cg_solve(matvec=matvec, b=b)
    assert info["converged"]
    assert info["residual_norm"] == 0.0

    mock_torch.dot.return_value.item.side_effect = [1.0, 0.5, 0.0]
    mock_torch.sqrt.return_value.item.side_effect = [1.0, 0.0]
    x, info = torch_backend.cg_solve(matvec=matvec, b=b, max_iter=2)
    assert info["converged"]

    mock_torch.dot.return_value.item.side_effect = [1.0, 0.0]
    mock_torch.sqrt.return_value.item.side_effect = [1.0]
    x, info = torch_backend.cg_solve(matvec=matvec, b=b, max_iter=2)
    assert not info["converged"]

    mock_torch.dot.return_value.item.side_effect = [1.0, 0.5, 0.5, 0.5, 0.5]
    mock_torch.sqrt.return_value.item.side_effect = None
    mock_torch.sqrt.return_value.item.return_value = 1.0
    x, info = torch_backend.cg_solve(matvec=matvec, b=b, max_iter=1)
    assert not info["converged"]
    assert info["n_iter"] == 1
