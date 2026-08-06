from __future__ import annotations

from typing import Literal

import numpy as np

NormalizeMode = Literal["none", "rw", "sym"]


def normalize_edge_weights(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    mode: NormalizeMode,
    eps: float = 1e-12,
) -> np.ndarray | None:
    """Normalize edge weights.

    - rw: D^{-1} A (row-normalized)
    - sym: D^{-1/2} A D^{-1/2}

    If edge_weight is None, returns None (interpreted as implicit ones).
    """
    if mode == "none":
        return edge_weight
    if edge_weight is None:
        return None

    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)
    weight_dtype = np.float64 if np.asarray(edge_weight).dtype == np.float64 else np.float32
    w = np.asarray(edge_weight, dtype=weight_dtype)

    if src.size and int(src.min()) >= 0:
        deg = np.bincount(src, weights=w, minlength=int(n_nodes)).astype(
            weight_dtype,
            copy=False,
        )
    else:
        deg = np.zeros(n_nodes, dtype=weight_dtype)
        np.add.at(deg, src, w)

    if mode == "rw":
        scale = 1.0 / np.maximum(deg[src], float(eps))
        return (w * scale).astype(weight_dtype)

    if mode == "sym":
        inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, float(eps)))
        return (w * inv_sqrt[src] * inv_sqrt[dst]).astype(weight_dtype)

    raise ValueError(f"Unknown normalization mode: {mode!r}")
