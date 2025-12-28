from __future__ import annotations

import numpy as np


def add_self_loops(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Add missing self-loops."""
    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)

    has_loop = np.zeros(n_nodes, dtype=bool)
    mask = src == dst
    has_loop[src[mask]] = True

    missing = np.where(~has_loop)[0].astype(np.int64)
    if missing.size == 0:
        return edge_index, edge_weight

    loops = np.vstack([missing, missing])
    ei = np.concatenate([edge_index, loops], axis=1)

    if edge_weight is None:
        return ei, None
    ew = np.asarray(edge_weight, dtype=np.float32)
    ew_loops = np.full(missing.shape[0], float(weight), dtype=np.float32)
    return ei, np.concatenate([ew, ew_loops], axis=0)
