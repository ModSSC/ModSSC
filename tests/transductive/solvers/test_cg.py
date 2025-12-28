from unittest.mock import MagicMock, patch

import numpy as np

from modssc.transductive.backends import torch_backend
from modssc.transductive.solvers.cg import CGResult, cg_solve_numpy, cg_solve_torch
from modssc.transductive.types import DeviceSpec


def test_cg_solve_numpy():
    with patch("modssc.transductive.solvers.cg.cg_numpy") as mock_cg:
        mock_x = np.array([1.0, 2.0])
        mock_info = {"n_iter": 10, "residual_norm": 1e-7, "converged": True}
        mock_cg.return_value = (mock_x, mock_info)

        matvec = MagicMock()
        b = np.array([1.0, 1.0])
        x0 = np.array([0.0, 0.0])

        result = cg_solve_numpy(matvec=matvec, b=b, x0=x0, tol=1e-5, max_iter=50)

        assert isinstance(result, CGResult)
        assert np.array_equal(result.x, mock_x)
        assert result.n_iter == 10
        assert result.residual_norm == 1e-7
        assert result.converged is True

        mock_cg.assert_called_once_with(matvec=matvec, b=b, x0=x0, tol=1e-5, max_iter=50)


def test_cg_solve_numpy_defaults():
    with patch("modssc.transductive.solvers.cg.cg_numpy") as mock_cg:
        mock_cg.return_value = (np.array([0]), {})

        matvec = MagicMock()
        b = np.array([1.0])

        result = cg_solve_numpy(matvec=matvec, b=b)

        mock_cg.assert_called_once()
        call_kwargs = mock_cg.call_args[1]
        assert call_kwargs["x0"] is None
        assert call_kwargs["tol"] == 1e-6
        assert call_kwargs["max_iter"] == 1000

        assert result.n_iter == 0
        assert np.isnan(result.residual_norm)
        assert result.converged is False


def test_cg_solve_torch():
    with patch.object(torch_backend, "cg_solve") as mock_cg_solve:
        mock_x = "torch_tensor_x"
        mock_info = "info_dict"
        mock_cg_solve.return_value = (mock_x, mock_info)

        matvec = MagicMock()
        b = "torch_tensor_b"
        device = DeviceSpec(device="cpu")

        x, info = cg_solve_torch(matvec=matvec, b=b, device=device, tol=1e-4, max_iter=200)

        assert x == mock_x
        assert info == mock_info

        mock_cg_solve.assert_called_once_with(matvec=matvec, b=b, x0=None, tol=1e-4, max_iter=200)
