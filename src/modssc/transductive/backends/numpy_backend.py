from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def _ensure_edge_weight(edge_weight: np.ndarray | None, E: int) -> np.ndarray:
    if edge_weight is None:
        return np.ones(E, dtype=np.float32)
    w = np.asarray(edge_weight)
    if w.ndim != 1 or w.shape[0] != E:
        raise ValueError(f"edge_weight must have shape (E,), got {w.shape}")
    return w.astype(np.float32, copy=False)


def spmm(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    X: np.ndarray,
) -> np.ndarray:
    """Sparse adjacency times dense features using edge list.

    Convention: messages flow src -> dst
    out[dst] += w * X[src]
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size == 0:
        return np.zeros((n_nodes,) + X.shape[1:], dtype=X.dtype)

    src = edge_index[0]
    dst = edge_index[1]
    w = _ensure_edge_weight(edge_weight, edge_index.shape[1]).astype(X.dtype, copy=False)

    out = np.zeros((n_nodes,) + X.shape[1:], dtype=X.dtype)
    if X.ndim == 1:
        np.add.at(out, dst, w * X[src])
    else:
        np.add.at(out, dst, (w[:, None] * X[src]))
    return out


def normalize_edges(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    mode: str,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return normalized edge weights.

    mode:
      - none: keep weights
      - rw: random walk (row normalised on dst)
      - sym: symmetric normalisation
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    E = int(edge_index.shape[1])
    w = _ensure_edge_weight(edge_weight, E).astype(np.float64, copy=False)
    if E == 0:
        return w.astype(np.float32)

    src = edge_index[0]
    dst = edge_index[1]

    deg = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(deg, dst, w)

    if mode == "none":
        out = w
    elif mode == "rw":
        out = w / np.maximum(deg[dst], eps)
    elif mode == "sym":
        out = w / np.sqrt(np.maximum(deg[dst], eps) * np.maximum(deg[src], eps))
    else:
        raise ValueError(f"Unknown normalization mode: {mode!r}")

    return out.astype(np.float32, copy=False)


def cg_solve(
    *,
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> tuple[np.ndarray, dict]:
    """Conjugate gradient for SPD operators, numpy implementation."""
    b = np.asarray(b, dtype=np.float64)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).copy()

    r = b - matvec(x)
    p = r.copy()
    rs_old = float(np.dot(r, r))

    info: dict = {
        "n_iter": 0,
        "converged": False,
        "residual_norm": math.sqrt(rs_old) if rs_old >= 0 else float("nan"),
    }
    if rs_old == 0.0:
        info["converged"] = True
        info["residual_norm"] = 0.0
        return x.astype(np.float32), info

    for k in range(int(max_iter)):
        Ap = matvec(p)
        denom = float(np.dot(p, Ap))
        if denom == 0.0:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(np.dot(r, r))
        info["n_iter"] = k + 1
        info["residual_norm"] = float(np.sqrt(rs_new))
        if info["residual_norm"] <= tol:
            info["converged"] = True
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x.astype(np.float32), info
