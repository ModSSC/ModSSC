from __future__ import annotations

from typing import Literal

import numpy as np

from modssc.runtime.device import resolve_device_name

from ...optional import optional_import

Metric = Literal["cosine", "euclidean"]


def _as_float32_contiguous(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)
    return X


def knn_edges_torch(
    X: np.ndarray,
    *,
    k: int,
    metric: Metric,
    include_self: bool = False,
    chunk_size: int = 512,
    device: str | None = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Exact chunked kNN using PyTorch.

    This backend is useful when dense feature matrices fit on CUDA/MPS memory.
    It intentionally mirrors the numpy backend: it is exact, chunked on the
    query axis, and returns directed edges plus distances.
    """
    torch = optional_import(
        "torch",
        extra="graph",
        purpose="torch graph construction backend",
    )

    X_np = _as_float32_contiguous(X)
    n = int(X_np.shape[0])
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    k_eff = min(int(k), n) if include_self else min(int(k), max(n - 1, 0))
    if k_eff <= 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    resolved_device = resolve_device_name(device, torch=torch) or "cpu"
    torch_device = torch.device(resolved_device)
    X_t = torch.as_tensor(X_np, dtype=torch.float32, device=torch_device)

    if metric == "cosine":
        X_work = torch.nn.functional.normalize(X_t, p=2, dim=1, eps=1e-12)
        norms = None
    elif metric == "euclidean":
        X_work = X_t
        norms = torch.sum(X_work * X_work, dim=1)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    dist_parts: list[np.ndarray] = []

    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        Xi = X_work[start:end]

        if metric == "cosine":
            scores = 1.0 - (Xi @ X_work.T)
        else:
            assert norms is not None
            dot = Xi @ X_work.T
            scores = norms[start:end, None] + norms[None, :] - 2.0 * dot
            scores = torch.clamp(scores, min=0.0)

        if not include_self:
            rows = torch.arange(end - start, device=torch_device)
            cols = torch.arange(start, end, device=torch_device)
            scores[rows, cols] = float("inf")

        top_values, idx = torch.topk(scores, k=k_eff, dim=1, largest=False, sorted=True)
        if metric == "euclidean":
            top_values = torch.sqrt(torch.clamp(top_values, min=0.0))

        idx_np = idx.detach().cpu().numpy().astype(np.int64, copy=False)
        dsel = top_values.detach().cpu().numpy().astype(np.float32, copy=False)

        src = np.repeat(np.arange(start, end, dtype=np.int64), idx_np.shape[1])
        dst = idx_np.reshape(-1).astype(np.int64, copy=False)
        dflat = dsel.reshape(-1).astype(np.float32, copy=False)

        finite = np.isfinite(dflat)
        src = src[finite]
        dst = dst[finite]
        dflat = dflat[finite]

        if not include_self:
            keep = src != dst
            src = src[keep]
            dst = dst[keep]
            dflat = dflat[keep]

        if src.size:
            src_parts.append(src)
            dst_parts.append(dst)
            dist_parts.append(dflat)

    src_all = np.concatenate(src_parts) if src_parts else np.asarray([], dtype=np.int64)
    dst_all = np.concatenate(dst_parts) if dst_parts else np.asarray([], dtype=np.int64)
    dist_all = np.concatenate(dist_parts) if dist_parts else np.asarray([], dtype=np.float32)

    edge_index = np.vstack([src_all, dst_all]).astype(np.int64)
    return edge_index, dist_all
