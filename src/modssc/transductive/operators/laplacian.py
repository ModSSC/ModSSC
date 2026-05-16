from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from ..types import DeviceSpec
from .normalize import normalize_edges_numpy, normalize_edges_torch
from .spmm import spmm_numpy

LaplacianKind = Literal["rw", "sym", "unnormalized"]


def _edge_weights_numpy(edge_index: Any, edge_weight: Any | None) -> np.ndarray:
    edge_index_arr = np.asarray(edge_index, dtype=np.int64)
    n_edges = int(edge_index_arr.shape[1])
    if edge_weight is None:
        return np.ones(n_edges, dtype=np.float32)
    w = np.asarray(edge_weight, dtype=np.float32).reshape(-1)
    if int(w.shape[0]) != n_edges:
        raise ValueError(f"edge_weight must have shape ({n_edges},), got {w.shape}")
    return w


def laplacian_matvec_numpy(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any | None,
    kind: LaplacianKind = "sym",
) -> Callable[[np.ndarray], np.ndarray]:
    if kind == "unnormalized":
        edge_index_arr = np.asarray(edge_index, dtype=np.int64)
        w = _edge_weights_numpy(edge_index_arr, edge_weight)
        deg = np.zeros(int(n_nodes), dtype=np.float32)
        if edge_index_arr.size:
            np.add.at(deg, edge_index_arr[1], w)

        def matvec_unnormalized(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            wx = spmm_numpy(n_nodes=n_nodes, edge_index=edge_index_arr, edge_weight=w, X=x)
            if x.ndim == 1:
                return deg * x - wx
            return deg[:, None] * x - wx

        return matvec_unnormalized

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
    if kind == "unnormalized":
        torch = torch_backend._torch()
        edge_index_t = torch_backend.to_tensor(edge_index, device=dev, dtype=torch.long)
        n_edges = int(edge_index_t.shape[1])
        if edge_weight is None:
            w = torch.ones((n_edges,), device=dev, dtype=dtype)
        else:
            w = torch_backend.to_tensor(edge_weight, device=dev, dtype=dtype).reshape(-1)
            if int(w.numel()) != n_edges:
                raise ValueError(f"edge_weight must have shape ({n_edges},)")
        deg = torch.zeros((int(n_nodes),), device=dev, dtype=dtype)
        if n_edges:
            deg.scatter_add_(0, edge_index_t[1], w)

        def matvec_unnormalized(x):
            x = torch.as_tensor(x, dtype=dtype, device=dev)
            wx = torch_backend.spmm(
                n_nodes=n_nodes,
                edge_index=edge_index_t,
                edge_weight=w,
                X=x,
                device=dev,
                dtype=dtype,
            )
            if int(x.dim()) == 1:
                return deg * x - wx
            return deg.view(-1, 1) * x - wx

        return matvec_unnormalized

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
