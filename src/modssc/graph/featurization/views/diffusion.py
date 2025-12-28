from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import GraphValidationError, OptionalDependencyError
from ..ops.adjacency import adjacency_from_edge_index


def _to_dense(X: Any, *, max_elements: int = 10_000_000) -> np.ndarray:
    if isinstance(X, np.ndarray):
        return X
    if hasattr(X, "toarray"):
        # likely scipy sparse
        n_el = int(X.shape[0]) * int(X.shape[1])
        if n_el > max_elements:
            raise GraphValidationError(
                "Sparse-to-dense conversion too large for diffusion view. Use an embedding step first."
            )
        return np.asarray(X.toarray())
    return np.asarray(X)


def diffusion_view(
    *,
    X: Any,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    steps: int,
    alpha: float,
) -> np.ndarray:
    """Diffusion view using a random-walk normalized adjacency.

    This is similar in spirit to APPNP feature propagation:
        H_{t+1} = (1 - alpha) * P @ H_t + alpha * X

    where P is a row-stochastic transition matrix.
    """
    X0 = _to_dense(X).astype(np.float32, copy=False)
    if X0.ndim != 2 or int(X0.shape[0]) != int(n_nodes):
        raise GraphValidationError("X must have shape (n_nodes, d)")

    src = np.asarray(edge_index[0], dtype=np.int64)
    if edge_weight is None:
        if src.size and int(src.min()) >= 0:
            deg = np.bincount(src, minlength=int(n_nodes)).astype(np.float32, copy=False)
        else:
            deg = np.zeros(n_nodes, dtype=np.float32)
            np.add.at(deg, src, 1.0)
        w_norm = 1.0 / np.maximum(deg[src], 1e-12)
    else:
        w = np.asarray(edge_weight, dtype=np.float32)
        if src.size and int(src.min()) >= 0:
            deg = np.bincount(src, weights=w, minlength=int(n_nodes)).astype(np.float32, copy=False)
        else:
            deg = np.zeros(n_nodes, dtype=np.float32)
            np.add.at(deg, src, w)
        w_norm = w / np.maximum(deg[src], 1e-12)

    # Prefer CSR if scipy is available. Otherwise, use dense for small graphs.
    try:
        A = adjacency_from_edge_index(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=w_norm, format="csr"
        )
    except OptionalDependencyError:
        if n_nodes > 4096:
            raise OptionalDependencyError(
                extra="graph",
                message="scipy is required for diffusion view on large graphs",
            ) from None
        A = adjacency_from_edge_index(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=w_norm, format="dense"
        )

    H = X0.copy()
    a = float(alpha)
    scale = 1.0 - a
    X0_scaled = X0 * a
    for _ in range(int(steps)):
        H = A @ H
        H *= scale
        H += X0_scaled

    return H.astype(np.float32, copy=False)
