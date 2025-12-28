from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

Metric = Literal["cosine", "euclidean"]


def _safe_l2_normalize(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, float(eps))
    return X / norms


def _save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp.npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        # best effort cleanup
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {k: np.asarray(npz[k]) for k in npz.files}


def knn_edges_numpy(
    X: np.ndarray,
    *,
    k: int,
    metric: Metric,
    include_self: bool = False,
    chunk_size: int = 512,
    work_dir: str | Path | None = None,
    resume: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact kNN edges using numpy (brute-force, chunked).

    Parameters
    ----------
    X:
        Dense feature matrix (n, d).
    k:
        Number of neighbors per node (excluding self).
    metric:
        "cosine" or "euclidean".
    include_self:
        Whether to allow i -> i in the neighbor list.
    chunk_size:
        Query batch size (affects memory).
    work_dir:
        If provided, stores each chunk result on disk as `knn_<start>_<end>.npz`.
        If `resume=True`, existing chunk files are reused. This enables resumable runs.
    resume:
        Reuse on-disk chunk files when `work_dir` is provided.

    Returns
    -------
    edge_index:
        Array with shape (2, E) of int64.
    distances:
        Array with shape (E,) of float32 distances (cosine distance or euclidean distance).
    """
    X = np.asarray(X)
    n = int(X.shape[0])
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    if metric == "cosine":
        Xn = _safe_l2_normalize(X.astype(np.float32, copy=False))
        norms = None
    else:
        Xn = X.astype(np.float32, copy=False)
        norms = np.sum(Xn * Xn, axis=1).astype(np.float32)

    k_eff = min(int(k), n) if include_self else min(int(k), max(n - 1, 0))
    if k_eff <= 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    # optional resumable storage
    wd = Path(work_dir) if work_dir is not None else None
    if wd is not None:
        wd.mkdir(parents=True, exist_ok=True)

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    dist_parts: list[np.ndarray] = []

    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        part_path = wd / f"knn_{start}_{end}.npz" if wd is not None else None

        if part_path is not None and resume and part_path.exists():
            data = _load_npz(part_path)
            src = np.asarray(data["src"], dtype=np.int64)
            dst = np.asarray(data["dst"], dtype=np.int64)
            dflat = np.asarray(data["dist"], dtype=np.float32)
        else:
            Xi = Xn[start:end]
            if metric == "cosine":
                sim = Xi @ Xn.T
                dist = (1.0 - sim).astype(np.float32)
            else:
                assert norms is not None
                dot = Xi @ Xn.T
                dist2 = norms[start:end, None] + norms[None, :] - 2.0 * dot
                dist2 = np.maximum(dist2, 0.0).astype(np.float32)

            if not include_self:
                rows = np.arange(start, end, dtype=np.int64) - start
                cols = np.arange(start, end, dtype=np.int64)
                if metric == "cosine":
                    dist[rows, cols] = np.inf
                else:
                    dist2[rows, cols] = np.inf

            # pick k_eff smallest distances per row
            if metric == "cosine":
                idx = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
                dsel = np.take_along_axis(dist, idx, axis=1)
            else:
                idx = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
                dsel2 = np.take_along_axis(dist2, idx, axis=1)
                dsel = np.sqrt(dsel2, out=dsel2)

            # stable ordering by distance
            order = np.argsort(dsel, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
            dsel = np.take_along_axis(dsel, order, axis=1)

            src = np.repeat(np.arange(start, end, dtype=np.int64), idx.shape[1])
            dst = idx.reshape(-1).astype(np.int64)
            dflat = dsel.reshape(-1).astype(np.float32)

            finite = np.isfinite(dflat)
            src = src[finite]
            dst = dst[finite]
            dflat = dflat[finite]

            # drop self neighbors if present
            if not include_self:
                keep = src != dst
                src = src[keep]
                dst = dst[keep]
                dflat = dflat[keep]

            if part_path is not None:
                _save_npz_atomic(part_path, src=src, dst=dst, dist=dflat)

        if src.size:
            src_parts.append(src)
            dst_parts.append(dst)
            dist_parts.append(dflat)

    src_all = np.concatenate(src_parts) if src_parts else np.asarray([], dtype=np.int64)
    dst_all = np.concatenate(dst_parts) if dst_parts else np.asarray([], dtype=np.int64)
    dist_all = np.concatenate(dist_parts) if dist_parts else np.asarray([], dtype=np.float32)

    edge_index = np.vstack([src_all, dst_all]).astype(np.int64)
    return edge_index, dist_all


def epsilon_edges_numpy(
    X: np.ndarray,
    *,
    radius: float,
    metric: Metric,
    include_self: bool = False,
    chunk_size: int = 512,
    work_dir: str | Path | None = None,
    resume: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Epsilon graph edges using numpy (brute-force, chunked)."""
    X = np.asarray(X)
    n = int(X.shape[0])
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    r = float(radius)
    if r <= 0:
        raise ValueError("radius must be > 0")

    if metric == "cosine":
        Xn = _safe_l2_normalize(X.astype(np.float32, copy=False))
        norms = None
    else:
        Xn = X.astype(np.float32, copy=False)
        norms = np.sum(Xn * Xn, axis=1).astype(np.float32)

    wd = Path(work_dir) if work_dir is not None else None
    if wd is not None:
        wd.mkdir(parents=True, exist_ok=True)

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    dist_parts: list[np.ndarray] = []

    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        part_path = wd / f"eps_{start}_{end}.npz" if wd is not None else None

        if part_path is not None and resume and part_path.exists():
            data = _load_npz(part_path)
            src = np.asarray(data["src"], dtype=np.int64)
            dst = np.asarray(data["dst"], dtype=np.int64)
            dflat = np.asarray(data["dist"], dtype=np.float32)
        else:
            Xi = Xn[start:end]
            if metric == "cosine":
                sim = Xi @ Xn.T
                dist = (1.0 - sim).astype(np.float32)
            else:
                assert norms is not None
                dot = Xi @ Xn.T
                dist2 = norms[start:end, None] + norms[None, :] - 2.0 * dot
                dist2 = np.maximum(dist2, 0.0).astype(np.float32)

            if not include_self:
                rows = np.arange(start, end, dtype=np.int64) - start
                cols = np.arange(start, end, dtype=np.int64)
                if metric == "cosine":
                    dist[rows, cols] = np.inf
                else:
                    dist2[rows, cols] = np.inf

            if metric == "cosine":
                rr, cc = np.where(dist <= r)
                dist_sel = dist
            else:
                r2 = float(r) * float(r)
                rr, cc = np.where(dist2 <= r2)
                dist_sel = dist2
            if rr.size:
                src = (rr.astype(np.int64) + int(start)).astype(np.int64)
                dst = cc.astype(np.int64)
                dflat = dist_sel[rr, cc].astype(np.float32)
                if metric != "cosine":
                    dflat = np.sqrt(dflat, out=dflat)
            else:
                src = np.asarray([], dtype=np.int64)
                dst = np.asarray([], dtype=np.int64)
                dflat = np.asarray([], dtype=np.float32)

            if part_path is not None:
                _save_npz_atomic(part_path, src=src, dst=dst, dist=dflat)

        if src.size:
            src_parts.append(src)
            dst_parts.append(dst)
            dist_parts.append(dflat)

    src_all = np.concatenate(src_parts) if src_parts else np.asarray([], dtype=np.int64)
    dst_all = np.concatenate(dst_parts) if dst_parts else np.asarray([], dtype=np.int64)
    dist_all = np.concatenate(dist_parts) if dist_parts else np.asarray([], dtype=np.float32)

    edge_index = np.vstack([src_all, dst_all]).astype(np.int64)
    return edge_index, dist_all
