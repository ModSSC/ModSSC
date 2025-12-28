from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ...optional import optional_import

AdjFormat = Literal["csr", "dense"]


def adjacency_from_edge_index(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    format: AdjFormat = "csr",
) -> Any:
    """Build an adjacency matrix from an edge list.

    Parameters
    ----------
    n_nodes:
        Number of nodes.
    edge_index:
        (2, E) int array.
    edge_weight:
        (E,) float array or None. If None, weights are assumed to be 1.
    format:
        "csr" requires scipy. "dense" uses numpy.

    Returns
    -------
    Any
        scipy.sparse.csr_matrix or numpy.ndarray.
    """
    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)

    if format == "dense":
        A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        if edge_weight is None:
            if src.size:
                A[src, dst] = 1.0
        else:
            data = np.asarray(edge_weight, dtype=np.float32)
            A[src, dst] = data
        return A

    if format == "csr":
        if edge_weight is None:
            data = np.ones(src.shape[0], dtype=np.float32)
        else:
            data = np.asarray(edge_weight, dtype=np.float32)
        scipy_sparse = optional_import("scipy.sparse", extra="graph")
        coo_matrix = scipy_sparse.coo_matrix
        return coo_matrix((data, (src, dst)), shape=(n_nodes, n_nodes)).tocsr()

    raise ValueError(f"Unknown format: {format!r}")
