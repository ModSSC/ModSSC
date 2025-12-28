from __future__ import annotations

from typing import Any

import numpy as np

from ..types import DeviceSpec
from ..validation import _as_numpy


def normalize_edges_numpy(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any | None,
    mode: str,
) -> np.ndarray:
    from ..backends.numpy_backend import normalize_edges  # local import

    ei = _as_numpy(edge_index).astype(np.int64, copy=False)
    ew = None if edge_weight is None else _as_numpy(edge_weight)
    return normalize_edges(n_nodes=n_nodes, edge_index=ei, edge_weight=ew, mode=mode)


def normalize_edges_torch(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any | None,
    mode: str,
    device: DeviceSpec,
):
    from ..backends import torch_backend

    dev = torch_backend.resolve_device(device)
    dtype = torch_backend.dtype_from_spec(device)
    return torch_backend.normalize_edges(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mode=mode,
        device=dev,
        dtype=dtype,
    )
