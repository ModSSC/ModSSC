from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..backends.numpy_backend import cg_solve as cg_numpy
from ..types import DeviceSpec


@dataclass(frozen=True)
class CGResult:
    x: np.ndarray
    n_iter: int
    residual_norm: float
    converged: bool


def cg_solve_numpy(
    *,
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> CGResult:
    x, info = cg_numpy(matvec=matvec, b=b, x0=x0, tol=tol, max_iter=max_iter)
    return CGResult(
        x=np.asarray(x),
        n_iter=int(info.get("n_iter", 0)),
        residual_norm=float(info.get("residual_norm", float("nan"))),
        converged=bool(info.get("converged", False)),
    )


def cg_solve_torch(
    *,
    matvec: Callable,
    b,
    device: DeviceSpec,
    x0=None,
    tol: float = 1e-6,
    max_iter: int = 1000,
):
    from ..backends import torch_backend

    x, info = torch_backend.cg_solve(matvec=matvec, b=b, x0=x0, tol=tol, max_iter=max_iter)
    return x, info
