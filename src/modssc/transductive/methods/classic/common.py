from __future__ import annotations

import numpy as np


def infer_num_classes(y: np.ndarray, labeled_mask: np.ndarray | None = None) -> int:
    y_valid = y[y >= 0]
    n_classes = int(y_valid.max()) + 1 if y_valid.size else 1
    return max(1, n_classes)


def build_affinity_matrix(
    *, n_nodes: int, edge_index: np.ndarray, edge_weight: np.ndarray
) -> np.ndarray:
    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    src = edge_index[0]
    dst = edge_index[1]
    np.add.at(W, (dst, src), edge_weight)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W
