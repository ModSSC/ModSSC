import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.transductive.adapters.pyg import to_pyg_data
from modssc.transductive.errors import OptionalDependencyError
from modssc.transductive.types import DeviceSpec


@pytest.fixture
def mock_torch_modules():
    with patch.dict(sys.modules, {"torch": MagicMock(), "torch_geometric.data": MagicMock()}):
        mock_torch = sys.modules["torch"]
        mock_tg = sys.modules["torch_geometric.data"]

        mock_torch.device = MagicMock(side_effect=lambda x: x)
        mock_torch.float32 = "float32"
        mock_torch.float64 = "float64"
        mock_torch.long = "long"
        mock_torch.bool = "bool"
        mock_torch.as_tensor = MagicMock(side_effect=lambda x, dtype=None, device=None: x)
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        mock_tg.Data = MagicMock()

        yield mock_torch, mock_tg


def test_to_pyg_data_basic(mock_torch_modules):
    _, mock_tg = mock_torch_modules

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    edge_index = np.array([[0, 1], [1, 0]])

    to_pyg_data(X=X, y=y, edge_index=edge_index)

    mock_tg.Data.assert_called_once()
    call_kwargs = mock_tg.Data.call_args[1]

    assert np.array_equal(call_kwargs["x"], X.astype(np.float32))
    assert np.array_equal(call_kwargs["y"], y.astype(np.int64))
    assert np.array_equal(call_kwargs["edge_index"], edge_index.astype(np.int64))


def test_to_pyg_data_with_weights_and_masks(mock_torch_modules):
    _, mock_tg = mock_torch_modules
    mock_data_instance = mock_tg.Data.return_value

    X = np.zeros((2, 2))
    y = np.zeros(2)
    edge_index = np.zeros((2, 2))
    edge_weight = np.array([0.5, 0.5])
    masks = {"train_mask": np.array([True, False])}

    to_pyg_data(X=X, y=y, edge_index=edge_index, edge_weight=edge_weight, masks=masks)

    assert hasattr(mock_data_instance, "edge_weight")
    assert np.array_equal(mock_data_instance.edge_weight, edge_weight.astype(np.float32))

    assert hasattr(mock_data_instance, "train_mask")
    assert np.array_equal(mock_data_instance.train_mask, masks["train_mask"])


def test_to_pyg_data_cuda_available(mock_torch_modules):
    mock_torch, _ = mock_torch_modules
    mock_torch.cuda.is_available.return_value = True

    to_pyg_data(X=[], y=[], edge_index=[], device=DeviceSpec(device="cuda"))

    mock_torch.device.assert_called_with("cuda")


def test_to_pyg_data_cuda_not_available(mock_torch_modules):
    mock_torch, _ = mock_torch_modules
    mock_torch.cuda.is_available.return_value = False

    with pytest.raises(OptionalDependencyError, match="CUDA not available"):
        to_pyg_data(X=[], y=[], edge_index=[], device=DeviceSpec(device="cuda"))


def test_to_pyg_data_mps_available(mock_torch_modules):
    mock_torch, _ = mock_torch_modules
    mock_torch.backends.mps.is_available.return_value = True

    to_pyg_data(X=[], y=[], edge_index=[], device=DeviceSpec(device="mps"))

    mock_torch.device.assert_called_with("mps")


def test_to_pyg_data_mps_not_available(mock_torch_modules):
    mock_torch, _ = mock_torch_modules
    mock_torch.backends.mps.is_available.return_value = False

    with pytest.raises(OptionalDependencyError, match="MPS not available"):
        to_pyg_data(X=[], y=[], edge_index=[], device=DeviceSpec(device="mps"))


def test_to_pyg_data_auto_device(mock_torch_modules):
    mock_torch, _ = mock_torch_modules

    mock_torch.cuda.is_available.return_value = True
    to_pyg_data(X=[], y=[], edge_index=[], device=DeviceSpec(device="auto"))
    mock_torch.device.assert_called_with("cuda")

    mock_torch.cuda.is_available.return_value = False
    to_pyg_data(X=[], y=[], edge_index=[], device=DeviceSpec(device="auto"))
    mock_torch.device.assert_called_with("cpu")


def test_to_pyg_data_dtype(mock_torch_modules):
    mock_torch, _ = mock_torch_modules

    mock_torch.as_tensor.reset_mock()
    to_pyg_data(X=[], y=[], edge_index=[], device=DeviceSpec(device="cpu", dtype="float32"))
    x_call = mock_torch.as_tensor.call_args_list[0]
    assert x_call.kwargs["dtype"] == "float32"
    assert x_call.kwargs["device"] == "cpu"

    mock_torch.as_tensor.reset_mock()
    to_pyg_data(X=[], y=[], edge_index=[], device=DeviceSpec(device="cpu", dtype="float64"))
    x_call = mock_torch.as_tensor.call_args_list[0]
    assert x_call.kwargs["dtype"] == "float64"
    assert x_call.kwargs["device"] == "cpu"


def test_to_pyg_data_unknown_device(mock_torch_modules):
    with pytest.raises(ValueError, match="Unknown device"):
        to_pyg_data(X=[], y=[], edge_index=[], device=DeviceSpec(device="tpu"))
