from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...errors import GraphValidationError
from ...optional import optional_import
from ..backends.faiss_backend import FaissParams, knn_search_faiss
from ..backends.numpy_backend import _safe_l2_normalize

Metric = Literal["cosine", "euclidean"]
Backend = Literal["numpy", "sklearn", "faiss"]
AnchorMethod = Literal["random", "kmeans"]


@dataclass(frozen=True)
class AnchorParams:
    n_anchors: int | None = None
    anchors_k: int = 5
    method: AnchorMethod = "random"
    candidate_limit: int = 1000
    chunk_size: int = 512


def _derive_seed(seed: int, salt: int) -> int:
    # Deterministic 32-bit mixing (avoid Python's randomized hash).
    # Implemented with Python ints to get wrap-around semantics without numpy overflow warnings.
    x = (int(seed) + 0x9E3779B97F4A7C15 + int(salt)) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 30) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 31) & 0xFFFFFFFFFFFFFFFF
    return int(x & 0xFFFFFFFF)


def _choose_anchors(
    X: np.ndarray,
    *,
    n_anchors: int,
    method: AnchorMethod,
    seed: int,
) -> np.ndarray:
    if method == "random":
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(int(X.shape[0]), size=int(n_anchors), replace=False)
        return np.asarray(idx, dtype=np.int64)

    if method == "kmeans":
        sklearn_cluster = optional_import("sklearn.cluster", extra="graph")
        MiniBatchKMeans = sklearn_cluster.MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=int(n_anchors), random_state=int(seed), n_init="auto")
        km.fit(np.asarray(X, dtype=np.float32))
        # return synthetic anchor vectors as float32
        return np.asarray(km.cluster_centers_, dtype=np.float32)

    raise GraphValidationError(f"Unknown anchor method: {method!r}")


def _knn_query_numpy(
    query: np.ndarray,
    ref: np.ndarray,
    *,
    k: int,
    metric: Metric,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)

    if metric == "cosine":
        qn = _safe_l2_normalize(q)
        rn = _safe_l2_normalize(r)
        norms_q = None
        norms_r = None
    else:
        qn = q
        rn = r
        norms_q = np.sum(qn * qn, axis=1).astype(np.float32)
        norms_r = np.sum(rn * rn, axis=1).astype(np.float32)

    n_q = int(qn.shape[0])
    k_eff = min(int(k), int(rn.shape[0]))
    idx_all = np.empty((n_q, k_eff), dtype=np.int64)
    dist_all = np.empty((n_q, k_eff), dtype=np.float32)

    for start in range(0, n_q, int(chunk_size)):
        end = min(n_q, start + int(chunk_size))
        Qi = qn[start:end]
        if metric == "cosine":
            sim = Qi @ rn.T
            dist = (1.0 - sim).astype(np.float32)
        else:
            assert norms_q is not None and norms_r is not None
            dot = Qi @ rn.T
            dist2 = norms_q[start:end, None] + norms_r[None, :] - 2.0 * dot
            dist2 = np.maximum(dist2, 0.0).astype(np.float32)
            dist = np.sqrt(dist2).astype(np.float32)

        part_idx = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
        part_d = np.take_along_axis(dist, part_idx, axis=1)
        order = np.argsort(part_d, axis=1)
        part_idx = np.take_along_axis(part_idx, order, axis=1)
        part_d = np.take_along_axis(part_d, order, axis=1)

        idx_all[start:end] = part_idx.astype(np.int64)
        dist_all[start:end] = part_d.astype(np.float32)

    return idx_all, dist_all


def _knn_query_sklearn(
    query: np.ndarray,
    ref: np.ndarray,
    *,
    k: int,
    metric: Metric,
) -> tuple[np.ndarray, np.ndarray]:
    sklearn_neighbors = optional_import("sklearn.neighbors", extra="graph")
    NearestNeighbors = sklearn_neighbors.NearestNeighbors
    nn = NearestNeighbors(n_neighbors=int(k), metric=metric)
    nn.fit(np.asarray(ref, dtype=np.float32))
    dist, idx = nn.kneighbors(np.asarray(query, dtype=np.float32), return_distance=True)
    return idx.astype(np.int64), dist.astype(np.float32)


def _knn_query_faiss(
    query: np.ndarray,
    ref: np.ndarray,
    *,
    k: int,
    metric: Metric,
    faiss_params: FaissParams,
) -> tuple[np.ndarray, np.ndarray]:
    idx, dist = knn_search_faiss(
        np.asarray(query, dtype=np.float32),
        np.asarray(ref, dtype=np.float32),
        k=int(k),
        metric=metric,
        include_self=True,  # query != ref here
        params=faiss_params,
    )
    return idx.astype(np.int64), dist.astype(np.float32)


def anchor_edges(
    X: np.ndarray,
    *,
    k: int,
    metric: Metric,
    backend: Backend,
    seed: int,
    params: AnchorParams,
    faiss_params: FaissParams | None = None,
    include_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate kNN using an anchor graph strategy.

    Algorithm
    ---------
    1) Choose `m` anchors (random subset of nodes, or kmeans centroids).
    2) For each node, find its `r` nearest anchors.
    3) Candidates for a node are the union of nodes that share at least one anchor.
    4) Compute exact distances to candidates and keep the best `k`.

    This keeps the interface compatible with the rest of the construction pipeline.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise GraphValidationError("X must be 2D")
    n = int(X.shape[0])
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    m = params.n_anchors
    if m is None:
        m = int(max(10, round(np.sqrt(n))))
    m = int(min(max(1, m), n))

    # choose anchors (indices or vectors)
    anchors = _choose_anchors(X, n_anchors=m, method=params.method, seed=seed)
    if anchors.ndim == 1:
        A = np.asarray(X[anchors], dtype=np.float32)
    else:
        A = np.asarray(anchors, dtype=np.float32)

    r = int(min(max(1, params.anchors_k), m))

    if backend == "numpy":
        anchor_idx, _ = _knn_query_numpy(X, A, k=r, metric=metric, chunk_size=params.chunk_size)
    elif backend == "sklearn":
        anchor_idx, _ = _knn_query_sklearn(X, A, k=r, metric=metric)
    elif backend == "faiss":
        fp = faiss_params or FaissParams()
        anchor_idx, _ = _knn_query_faiss(X, A, k=r, metric=metric, faiss_params=fp)
    else:
        raise GraphValidationError(f"Unknown backend: {backend!r}")

    # build anchor -> nodes mapping
    anchor_to_nodes: list[list[int]] = [[] for _ in range(m)]
    for i in range(n):
        for a in anchor_idx[i]:
            anchor_to_nodes[int(a)].append(i)

    # pre-normalize for cosine distance
    if metric == "cosine":
        Xn = _safe_l2_normalize(np.asarray(X, dtype=np.float32))
    else:
        Xn = np.asarray(X, dtype=np.float32)
        norms = np.sum(Xn * Xn, axis=1).astype(np.float32)

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    dist_parts: list[np.ndarray] = []

    for i in range(n):
        # union candidates from the r anchors of i
        cand_set: set[int] = set()
        for a in anchor_idx[i]:
            cand_set.update(anchor_to_nodes[int(a)])
        if not include_self:
            cand_set.discard(i)

        if not cand_set:
            continue

        cand = np.fromiter(sorted(cand_set), dtype=np.int64)
        if cand.size > int(params.candidate_limit):
            # deterministic down-sampling
            per_rng = np.random.default_rng(_derive_seed(seed, i))
            cand = per_rng.choice(cand, size=int(params.candidate_limit), replace=False)
            cand = np.sort(cand.astype(np.int64))

        xi = Xn[i]
        Xc = Xn[cand]
        if metric == "cosine":
            # cosine distance on normalized vectors
            sim = Xc @ xi
            d = (1.0 - sim).astype(np.float32, copy=False)
        else:
            # euclidean distance
            assert metric == "euclidean"
            xi2 = float(norms[i])
            c2 = norms[cand]
            dot = Xc @ xi
            dist2 = (xi2 + c2 - 2.0 * dot).astype(np.float32)
            dist2 = np.maximum(dist2, 0.0)
            d = np.sqrt(dist2).astype(np.float32)

        kk = int(min(int(k), int(cand.size)))
        if kk <= 0:
            continue

        sel = np.argpartition(d, kth=kk - 1)[:kk]
        sel = sel[np.argsort(d[sel])]
        neigh = cand[sel].astype(np.int64, copy=False)
        dist_sel = d[sel].astype(np.float32, copy=False)

        src = np.full(neigh.shape[0], i, dtype=np.int64)
        src_parts.append(src)
        dst_parts.append(neigh)
        dist_parts.append(dist_sel)

    src_all = np.concatenate(src_parts) if src_parts else np.asarray([], dtype=np.int64)
    dst_all = np.concatenate(dst_parts) if dst_parts else np.asarray([], dtype=np.int64)
    dist_all = np.concatenate(dist_parts) if dist_parts else np.asarray([], dtype=np.float32)

    edge_index = np.vstack([src_all, dst_all]).astype(np.int64)
    # distances only; meta is handled by higher-level manifest/meta
    return edge_index, dist_all
