from __future__ import annotations

from typing import Literal

import numpy as np

from ...optional import optional_import

Metric = Literal["cosine", "euclidean"]


def knn_edges_sklearn(
    X: np.ndarray,
    *,
    k: int,
    metric: Metric,
    include_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact kNN edges using scikit-learn NearestNeighbors."""
    sklearn_neighbors = optional_import("sklearn.neighbors", extra="sklearn")

    X = np.asarray(X)
    n = int(X.shape[0])
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    # Ask for one extra neighbor to drop self if needed.
    n_neighbors = min(n, int(k) + (0 if include_self else 1))
    nn = sklearn_neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(X)

    dist, idx = nn.kneighbors(X, return_distance=True)

    dist = np.asarray(dist, dtype=np.float32)
    idx = np.asarray(idx, dtype=np.int64)
    n_rows, n_cols = idx.shape
    src = np.repeat(np.arange(n_rows, dtype=np.int64), n_cols)
    dst = idx.reshape(-1)
    dist_flat = dist.reshape(-1)
    if not include_self:
        mask = dst != src
        src = src[mask]
        dst = dst[mask]
        dist_flat = dist_flat[mask]

    edge_index = np.vstack([src, dst])
    distances = dist_flat.astype(np.float32, copy=False)
    return edge_index, distances


def epsilon_edges_sklearn(
    X: np.ndarray,
    *,
    radius: float,
    metric: Metric,
    include_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Epsilon graph edges using scikit-learn radius_neighbors."""
    sklearn_neighbors = optional_import("sklearn.neighbors", extra="sklearn")

    X = np.asarray(X)
    n = int(X.shape[0])
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    nn = sklearn_neighbors.NearestNeighbors(radius=float(radius), metric=metric)
    nn.fit(X)

    dists, inds = nn.radius_neighbors(X, return_distance=True, sort_results=True)

    lengths = np.fromiter((len(x) for x in inds), dtype=np.int64, count=n)
    if lengths.sum() == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    src = np.repeat(np.arange(n, dtype=np.int64), lengths)
    dst = np.concatenate(inds).astype(np.int64, copy=False)
    dist_flat = np.concatenate(dists).astype(np.float32, copy=False)

    if not include_self:
        mask = dst != src
        src = src[mask]
        dst = dst[mask]
        dist_flat = dist_flat[mask]

    edge_index = np.vstack([src, dst])
    distances = dist_flat.astype(np.float32, copy=False)
    return edge_index, distances
