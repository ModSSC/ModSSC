from __future__ import annotations

import numpy as np

from ...errors import GraphValidationError
from ...specs import GraphWeightsSpec, Metric


def compute_edge_weights(
    *,
    distances: np.ndarray,
    metric: Metric,
    weights: GraphWeightsSpec,
    edge_index: np.ndarray | None = None,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Compute edge weights from a distance array.

    Parameters
    ----------
    distances:
        1D array of distances for each edge.
        For cosine metric, this must be cosine distance in [0, 2].
    metric:
        Distance metric used by the builder.
    weights:
        Weight specification.

    Returns
    -------
    np.ndarray
        float32 weights array.
    """
    d = np.asarray(distances, dtype=np.float32)
    if d.ndim != 1:
        raise GraphValidationError("distances must be 1D")

    if weights.kind == "binary":
        return np.ones_like(d, dtype=np.float32)

    if weights.kind == "heat":
        sigma = float(weights.sigma or 0.0)
        if sigma <= 0:
            raise GraphValidationError("sigma must be > 0 for heat weights")
        return np.exp(-(d * d) / (2.0 * sigma * sigma)).astype(np.float32)

    if weights.kind == "cosine":
        if metric != "cosine":
            raise GraphValidationError("cosine weights require metric='cosine'")
        # cosine distance -> similarity in [-1, 1] roughly, but for normalized vectors it is [0,2]
        return (1.0 - d).astype(np.float32)

    if weights.kind == "knn_gaussian":
        if metric != "euclidean":
            raise GraphValidationError("knn_gaussian weights require metric='euclidean'")
        if edge_index is None or n_nodes is None:
            raise GraphValidationError("knn_gaussian weights require edge_index and n_nodes")
        ei = np.asarray(edge_index)
        if ei.shape != (2, d.shape[0]):
            raise GraphValidationError("edge_index must have shape (2, E) for knn_gaussian weights")
        src = ei[0].astype(np.int64, copy=False)
        n = int(n_nodes)
        if n < 0:
            raise GraphValidationError("n_nodes must be non-negative")
        if src.size and (src.min() < 0 or src.max() >= n):
            raise GraphValidationError("edge_index source ids out of range")
        d2 = d * d
        eps = np.zeros(n, dtype=np.float32)
        np.maximum.at(eps, src, d2)
        eps = np.maximum(eps, np.finfo(np.float32).eps)
        return np.exp(-4.0 * d2 / eps[src]).astype(np.float32)

    raise GraphValidationError(f"Unknown weight kind: {weights.kind!r}")
