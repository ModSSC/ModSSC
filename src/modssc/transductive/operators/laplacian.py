from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from ..types import DeviceSpec
from .normalize import normalize_edges_numpy, normalize_edges_torch
from .spmm import spmm_numpy

LaplacianKind = Literal["rw", "sym"]


def laplacian_matvec_numpy(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any | None,
    kind: LaplacianKind = "sym",
) -> Callable[[np.ndarray], np.ndarray]:
    # We build Lx = x - Sx where S is normalised adjacency
    w = normalize_edges_numpy(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode=kind
    )

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x - spmm_numpy(n_nodes=n_nodes, edge_index=edge_index, edge_weight=w, X=x)

    return matvec


def laplacian_matvec_torch(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any | None,
    device: DeviceSpec,
    kind: LaplacianKind = "sym",
):
    from ..backends import torch_backend

    dev = torch_backend.resolve_device(device)
    dtype = torch_backend.dtype_from_spec(device)
    w = normalize_edges_torch(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode=kind, device=device
    )

    def matvec(x):
        return x - torch_backend.spmm(
            n_nodes=n_nodes,
            edge_index=edge_index,
            edge_weight=w,
            X=x,
            device=dev,
            dtype=dtype,
        )

    return matvec
