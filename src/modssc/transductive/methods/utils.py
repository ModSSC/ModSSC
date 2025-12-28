from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


NormMode = Literal["rw", "sym"]


@dataclass(frozen=True)
class DiffusionResult:
    """Result of an iterative diffusion process.

    Attributes
    ----------
    F:
        Array of shape (n_nodes, n_classes) with class scores or probabilities.
    n_iter:
        Number of iterations performed.
    residual:
        Infinity norm of the last update, used as a convergence indicator.
    """

    F: np.ndarray
    n_iter: int
    residual: float


def _validate_graph_inputs(
    *, n_nodes: int, edge_index: np.ndarray, edge_weight: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_index.dtype.kind not in ("i", "u"):
        edge_index = edge_index.astype(np.int64, copy=False)

    if edge_weight is None:
        w = np.ones(edge_index.shape[1], dtype=np.float32)
    else:
        w = np.asarray(edge_weight, dtype=np.float32)
        if w.shape != (edge_index.shape[1],):
            raise ValueError("edge_weight must have shape (E,)")

    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")

    src = edge_index[0]
    dst = edge_index[1]
    if src.min(initial=0) < 0 or dst.min(initial=0) < 0:
        raise ValueError("edge_index must be non-negative")
    if src.max(initial=0) >= n_nodes or dst.max(initial=0) >= n_nodes:
        raise ValueError("edge_index contains node id >= n_nodes")

    return edge_index.astype(np.int64, copy=False), w


def spmm_numpy(
    *, n_nodes: int, edge_index: np.ndarray, edge_weight: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Sparse matrix multiplication out = A @ X.

    The sparse matrix A is represented by edge_index (src, dst) and edge_weight,
    and is interpreted as A[dst, src] = edge_weight[e].

    Parameters
    ----------
    n_nodes:
        Number of nodes (rows of A, first dimension of output).
    edge_index:
        int array of shape (2, E): src and dst indices.
    edge_weight:
        float array of shape (E,).
    X:
        Dense array of shape (n_nodes, d).

    Returns
    -------
    out:
        Dense array of shape (n_nodes, d).
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] != n_nodes:
        raise ValueError("X must have shape (n_nodes, d)")
    src = edge_index[0]
    dst = edge_index[1]
    out = np.zeros((n_nodes, X.shape[1]), dtype=X.dtype)
    np.add.at(out, dst, edge_weight[:, None] * X[src])
    return out


def degrees_numpy(*, n_nodes: int, edge_index: np.ndarray, edge_weight: np.ndarray) -> np.ndarray:
    """Row degrees with the same A[dst, src] convention."""
    dst = edge_index[1]
    deg = np.zeros((n_nodes,), dtype=edge_weight.dtype)
    np.add.at(deg, dst, edge_weight)
    return deg


def normalize_edge_weight_numpy(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    mode: NormMode,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return normalized edge weights for diffusion."""
    deg = degrees_numpy(n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight)
    deg = np.maximum(deg, eps)

    src = edge_index[0]
    dst = edge_index[1]

    if mode == "rw":
        return edge_weight / deg[dst]
    if mode == "sym":
        return edge_weight / np.sqrt(deg[src] * deg[dst])
    raise ValueError(f"Unknown norm mode: {mode!r}")


def to_numpy(x: object) -> np.ndarray:
    """Convert torch-like objects to numpy without requiring torch at runtime."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def spmm_torch(  # pragma: no cover
    *, n_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor, X: torch.Tensor
) -> torch.Tensor:
    """Sparse matrix multiplication out = A @ X in torch.

    A is represented by edge_index (src, dst) and edge_weight, with A[dst, src] = w.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for spmm_torch")

    if X.ndim != 2 or int(X.shape[0]) != int(n_nodes):
        raise ValueError("X must have shape (n_nodes, d)")

    src = edge_index[0].long()
    dst = edge_index[1].long()
    out = torch.zeros((n_nodes, X.shape[1]), dtype=X.dtype, device=X.device)
    out.index_add_(0, dst, edge_weight.view(-1, 1) * X[src])
    return out


def degrees_torch(  # pragma: no cover
    *, n_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor
) -> torch.Tensor:
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for degrees_torch")
    dst = edge_index[1].long()
    deg = torch.zeros((n_nodes,), dtype=edge_weight.dtype, device=edge_weight.device)
    deg.index_add_(0, dst, edge_weight)
    return deg


def normalize_edge_weight_torch(  # pragma: no cover
    *,
    n_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    mode: NormMode,
    eps: float = 1e-12,
) -> torch.Tensor:
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for normalize_edge_weight_torch")

    deg = degrees_torch(n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight)
    deg = torch.clamp(deg, min=float(eps))

    src = edge_index[0].long()
    dst = edge_index[1].long()

    if mode == "rw":
        return edge_weight / deg[dst]
    if mode == "sym":
        return edge_weight / torch.sqrt(deg[src] * deg[dst])
    raise ValueError(f"Unknown norm mode: {mode!r}")


def labels_to_onehot(y, n_classes: int):
    """Convert integer class labels to a one-hot matrix.

    This utility is used by some PDE-based methods (e.g., Poisson learning).

    Parameters
    ----------
    y:
        1D array/tensor of integer labels.
    n_classes:
        Number of classes.

    Returns
    -------
    One-hot encoded labels as a numpy.ndarray or torch.Tensor depending on the
    input type.
    """
    if n_classes <= 0:
        raise ValueError(f"n_classes must be positive, got {n_classes}")

    if (
        "torch" in globals() and torch is not None and isinstance(y, torch.Tensor)
    ):  # pragma: no cover
        y = y.long().reshape(-1)
        out = torch.zeros((y.numel(), n_classes), device=y.device, dtype=torch.float32)
        out[torch.arange(y.numel(), device=y.device), y] = 1.0
        return out

    y_np = np.asarray(y, dtype=np.int64).reshape(-1)
    out_np = np.zeros((y_np.shape[0], n_classes), dtype=np.float32)
    out_np[np.arange(y_np.shape[0]), y_np] = 1.0
    return out_np
