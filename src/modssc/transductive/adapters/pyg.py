from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from modssc.device import mps_is_available, resolve_device_name

from ..errors import OptionalDependencyError
from ..optional import optional_import
from ..types import DeviceSpec
from ..validation import _as_numpy


def to_pyg_data(
    *,
    X: Any,
    y: Any,
    edge_index: Any,
    edge_weight: Any | None = None,
    masks: Mapping[str, Any] | None = None,
    device: DeviceSpec | None = None,
):
    """Convert arrays to torch_geometric.data.Data.

    Imports are lazy. If torch or torch_geometric are missing, raise OptionalDependencyError.
    """
    torch = optional_import("torch", extra="transductive-torch")
    tg = optional_import(
        "torch_geometric.data", extra="transductive-pyg", package_hint="torch_geometric"
    )
    Data = tg.Data  # type: ignore[attr-defined]

    dev = torch.device("cpu")
    dtype = torch.float32
    if device is not None:
        requested = device.device
        resolved = resolve_device_name(requested, torch=torch)
        # resolve device string without importing transductive backends
        if requested == "cuda" and not torch.cuda.is_available():
            raise OptionalDependencyError(
                "torch", "transductive-torch", message="CUDA not available"
            )
        if requested == "mps" and not mps_is_available(torch):
            raise OptionalDependencyError(
                "torch", "transductive-torch", message="MPS not available"
            )
        if resolved not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unknown device: {requested!r}")
        dev = torch.device(resolved)

        dtype = torch.float32 if device.dtype == "float32" else torch.float64

    Xn = _as_numpy(X).astype(np.float32, copy=False)
    yn = _as_numpy(y).astype(np.int64, copy=False)
    ei = _as_numpy(edge_index).astype(np.int64, copy=False)

    data = Data(
        x=torch.as_tensor(Xn, dtype=dtype, device=dev),
        y=torch.as_tensor(yn, dtype=torch.long, device=dev),
        edge_index=torch.as_tensor(ei, dtype=torch.long, device=dev),
    )
    if edge_weight is not None:
        ew = _as_numpy(edge_weight).astype(np.float32, copy=False)
        data.edge_weight = torch.as_tensor(ew, dtype=dtype, device=dev)

    if masks:
        for k, v in masks.items():
            mv = _as_numpy(v).astype(bool, copy=False)
            setattr(data, k, torch.as_tensor(mv, dtype=torch.bool, device=dev))

    return data
