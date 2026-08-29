from __future__ import annotations

import numpy as np


def zero_diagonal_edges(
    *,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Set a sparse graph diagonal to zero by removing explicit self-edges.

    Removing diagonal entries is the canonical sparse representation of
    ``W[i, i] = 0``. Non-diagonal edges retain their order and dtype.
    """

    edge_index_arr = np.asarray(edge_index)
    if edge_index_arr.ndim != 2 or edge_index_arr.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")

    keep = edge_index_arr[0] != edge_index_arr[1]
    if bool(np.all(keep)):
        return edge_index, edge_weight

    filtered_index = edge_index_arr[:, keep]
    if edge_weight is None:
        return filtered_index, None

    edge_weight_arr = np.asarray(edge_weight)
    if edge_weight_arr.ndim != 1 or edge_weight_arr.shape[0] != edge_index_arr.shape[1]:
        raise ValueError("edge_weight must be one-dimensional with one value per edge")
    return filtered_index, edge_weight_arr[keep]
