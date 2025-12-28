from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...optional import optional_import

Metric = Literal["cosine", "euclidean"]


@dataclass(frozen=True)
class FaissParams:
    exact: bool = False
    hnsw_m: int = 32
    ef_search: int = 64
    ef_construction: int = 200


def _as_float32_contiguous(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)
    return X


def _l2_normalize(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, float(eps)).astype(np.float32)
    return (X / norms).astype(np.float32, copy=False)


def knn_search_faiss(
    query: np.ndarray,
    ref: np.ndarray,
    *,
    k: int,
    metric: Metric,
    include_self: bool = False,
    params: FaissParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """kNN search using FAISS (optional).

    Returns
    -------
    indices:
        (n_query, k_eff) int64 neighbor indices in `ref`.
    distances:
        (n_query, k_eff) float32 distances (cosine distance or euclidean distance).
    """
    faiss = optional_import("faiss", extra="graph-faiss")  # raises OptionalDependencyError

    params = params or FaissParams()
    q = _as_float32_contiguous(query)
    r = _as_float32_contiguous(ref)

    if metric == "cosine":
        q = _l2_normalize(q)
        r = _l2_normalize(r)
        metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        metric_type = faiss.METRIC_L2

    d = int(r.shape[1])
    if params.exact:
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexFlatL2(d)
    else:
        index = faiss.IndexHNSWFlat(d, int(params.hnsw_m), metric_type)
        index.hnsw.efSearch = int(params.ef_search)
        index.hnsw.efConstruction = int(params.ef_construction)

    index.add(r)

    k_eff = int(k)
    if not include_self and query is ref:
        k_eff = min(int(k) + 1, int(r.shape[0]))

    D, indices = index.search(q, int(k_eff))
    indices = indices.astype(np.int64, copy=False)

    if metric == "cosine":
        # D is inner product similarity in [-1, 1] if normalized.
        dist = (1.0 - D).astype(np.float32, copy=False)
    else:
        # D is squared L2 distance
        dist = np.sqrt(np.maximum(D, 0.0)).astype(np.float32)

    if not include_self and query is ref:
        # remove diagonal neighbors
        rows = np.arange(indices.shape[0], dtype=np.int64)[:, None]
        keep = rows != indices
        # keep at most k per row
        out_indices = np.full((indices.shape[0], int(k)), -1, dtype=np.int64)
        out_D = np.full((indices.shape[0], int(k)), np.inf, dtype=np.float32)
        for i in range(indices.shape[0]):
            sel = np.where(keep[i])[0]
            sel = sel[: int(k)]
            if sel.size:
                out_indices[i, : sel.size] = indices[i, sel]
                out_D[i, : sel.size] = dist[i, sel]
        indices = out_indices
        dist = out_D

    return indices, dist


def knn_edges_faiss(
    X: np.ndarray,
    *,
    k: int,
    metric: Metric,
    include_self: bool = False,
    params: FaissParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build directed kNN edges for X using FAISS."""
    idx, dist = knn_search_faiss(
        X,
        X,
        k=int(k),
        metric=metric,
        include_self=include_self,
        params=params,
    )

    n = int(X.shape[0])
    src = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    dst = idx.reshape(-1).astype(np.int64, copy=False)
    dflat = dist.reshape(-1).astype(np.float32, copy=False)

    valid = dst >= 0
    src = src[valid]
    dst = dst[valid]
    dflat = dflat[valid]

    edge_index = np.vstack([src, dst]).astype(np.int64)
    return edge_index, dflat
