from unittest.mock import MagicMock, patch

import numpy as np

from modssc.transductive.operators.clamp import labels_to_onehot
from modssc.transductive.solvers.fixed_point import (
    FixedPointResult,
    fixed_point_diffusion_numpy,
    fixed_point_diffusion_torch,
)


def test_fixed_point_diffusion_numpy_max_iter():
    n_nodes = 2
    edge_index = np.array([[0, 1], [1, 0]])
    edge_weight = np.array([1.0, 1.0])
    Y = np.array([[1.0, 0.0], [0.0, 1.0]])
    train_mask = np.array([True, False])

    result = fixed_point_diffusion_numpy(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        Y=Y,
        train_mask=train_mask,
        alpha=0.5,
        max_iter=1,
        tol=1e-10,
        normalize_rows=True,
    )

    assert isinstance(result, FixedPointResult)
    assert result.n_iter == 1
    assert result.residual > 1e-10


def test_fixed_point_diffusion_numpy_no_norm_rows():
    n_nodes = 2
    edge_index = np.array([[0, 1], [1, 0]])
    edge_weight = np.array([1.0, 1.0])
    Y = np.array([[1.0, 0.0], [0.0, 1.0]])
    train_mask = np.array([True, False])

    result = fixed_point_diffusion_numpy(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        Y=Y,
        train_mask=train_mask,
        alpha=0.5,
        max_iter=1,
        normalize_rows=False,
    )
    assert isinstance(result, FixedPointResult)


def test_fixed_point_diffusion_numpy_runs():
    n_nodes = 4
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    edge_index = np.hstack([edge_index, edge_index[[1, 0], :]])
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    train_mask = np.array([True, False, False, True])
    Y = labels_to_onehot(y, n_classes=2)
    Y[~train_mask] = 0.0

    result = fixed_point_diffusion_numpy(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=None,
        Y=Y,
        train_mask=train_mask,
        alpha=0.9,
        max_iter=50,
        tol=1e-6,
        normalize_rows=True,
    )

    assert result.F.shape == (n_nodes, 2)
    assert result.n_iter >= 1


def test_fixed_point_diffusion_torch():
    with (
        patch("modssc.transductive.backends.torch_backend") as mock_backend,
        patch("modssc.transductive.solvers.fixed_point.normalize_edges_torch") as mock_norm,
    ):
        mock_torch = MagicMock()
        mock_backend._torch.return_value = mock_torch
        mock_backend.resolve_device.return_value = "cpu"
        mock_backend.dtype_from_spec.return_value = "float32"

        mock_Yt = MagicMock()
        mock_m = MagicMock()
        mock_backend.to_tensor.side_effect = [mock_Yt, mock_m]

        mock_F = MagicMock()
        mock_Yt.clone.return_value = mock_F

        mock_PF = MagicMock()
        mock_backend.spmm.return_value = mock_PF

        mock_torch.norm.return_value.item.return_value = 0.0

        result = fixed_point_diffusion_torch(
            n_nodes=10,
            edge_index=MagicMock(),
            edge_weight=None,
            Y=MagicMock(),
            train_mask=MagicMock(),
            device="cpu",
            max_iter=10,
            normalize_rows=True,
        )

        assert result["n_iter"] == 1
        assert result["residual"] == 0.0

        mock_backend.to_tensor.assert_called()
        mock_norm.assert_called()
        mock_backend.spmm.assert_called()


def test_fixed_point_diffusion_torch_no_convergence():
    with (
        patch("modssc.transductive.backends.torch_backend") as mock_backend,
        patch("modssc.transductive.solvers.fixed_point.normalize_edges_torch"),
    ):
        mock_torch = MagicMock()
        mock_backend._torch.return_value = mock_torch

        mock_torch.norm.return_value.item.return_value = 1.0

        result = fixed_point_diffusion_torch(
            n_nodes=10,
            edge_index=MagicMock(),
            edge_weight=None,
            Y=MagicMock(),
            train_mask=MagicMock(),
            device="cpu",
            max_iter=5,
            tol=0.1,
        )

        assert result["n_iter"] == 5
        assert result["residual"] == 1.0
