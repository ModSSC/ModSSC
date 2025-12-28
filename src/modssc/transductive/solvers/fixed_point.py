from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..operators.clamp import hard_clamp, row_normalize
from ..operators.normalize import normalize_edges_numpy, normalize_edges_torch
from ..operators.spmm import spmm_numpy
from ..types import DeviceSpec


@dataclass(frozen=True)
class FixedPointResult:
    F: np.ndarray
    n_iter: int
    residual: float


def fixed_point_diffusion_numpy(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any | None,
    Y: np.ndarray,
    train_mask: np.ndarray,
    alpha: float = 0.99,
    max_iter: int = 200,
    tol: float = 1e-6,
    normalize_rows: bool = False,
    norm_mode: Literal["rw", "sym", "none"] = "rw",
) -> FixedPointResult:
    w = normalize_edges_numpy(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode=norm_mode
    )
    F = Y.astype(np.float32, copy=True)
    last_res = float("inf")
    for k in range(int(max_iter)):
        PF = spmm_numpy(n_nodes=n_nodes, edge_index=edge_index, edge_weight=w, X=F)
        F_new = alpha * PF + (1.0 - alpha) * Y
        F_new = hard_clamp(F_new, Y, train_mask)
        if normalize_rows:
            F_new = row_normalize(F_new)
        res = float(np.linalg.norm(F_new - F))
        F = F_new
        last_res = res
        if res <= tol:
            return FixedPointResult(F=F, n_iter=k + 1, residual=res)
    return FixedPointResult(F=F, n_iter=int(max_iter), residual=last_res)


def fixed_point_diffusion_torch(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any | None,
    Y: Any,
    train_mask: Any,
    device: DeviceSpec,
    alpha: float = 0.99,
    max_iter: int = 200,
    tol: float = 1e-6,
    normalize_rows: bool = False,
    norm_mode: Literal["rw", "sym", "none"] = "rw",
):
    from ..backends import torch_backend

    torch = torch_backend._torch()
    dev = torch_backend.resolve_device(device)
    dtype = torch_backend.dtype_from_spec(device)

    Yt = torch_backend.to_tensor(Y, device=dev, dtype=dtype)
    m = torch_backend.to_tensor(train_mask, device=dev, dtype=None).bool()

    w = normalize_edges_torch(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mode=norm_mode,
        device=device,
    )
    F = Yt.clone()

    last_res = float("inf")
    for k in range(int(max_iter)):
        PF = torch_backend.spmm(
            n_nodes=n_nodes,
            edge_index=edge_index,
            edge_weight=w,
            X=F,
            device=dev,
            dtype=dtype,
        )
        F_new = alpha * PF + (1.0 - alpha) * Yt
        F_new[m] = Yt[m]
        if normalize_rows:
            sums = torch.clamp(F_new.sum(dim=1, keepdim=True), min=1e-12)
            F_new = F_new / sums
        res = float(torch.norm(F_new - F).item())
        F = F_new
        last_res = res
        if res <= tol:
            return {"F": F, "n_iter": k + 1, "residual": res}
    return {"F": F, "n_iter": int(max_iter), "residual": last_res}
