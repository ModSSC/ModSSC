from __future__ import annotations

from typing import Any

import numpy as np

from ..types import DeviceSpec
from ..validation import _as_numpy


def spmm_numpy(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any | None,
    X: Any,
) -> np.ndarray:
    from ..backends.numpy_backend import spmm  # local import

    ei = _as_numpy(edge_index).astype(np.int64, copy=False)
    ew = None if edge_weight is None else _as_numpy(edge_weight)
    Xn = _as_numpy(X)
    return spmm(n_nodes=n_nodes, edge_index=ei, edge_weight=ew, X=Xn)


def spmm_torch(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any | None,
    X: Any,
    device: DeviceSpec,
):
    from ..backends import torch_backend

    dev = torch_backend.resolve_device(device)
    dtype = torch_backend.dtype_from_spec(device)
    Xt = torch_backend.to_tensor(X, device=dev, dtype=dtype)
    out = torch_backend.spmm(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        X=Xt,
        device=dev,
        dtype=dtype,
    )
    return out
