from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import GraphValidationError


def _as_int64(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.int64)


def _as_float32(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


@dataclass(frozen=True)
class GraphArtifact:
    """Canonical graph representation.

    Notes
    -----
    For reproducible experiments, graph construction should be fingerprinted and cached
    (see :func:`modssc.graph.build_graph`).
    """

    n_nodes: int
    edge_index: np.ndarray
    edge_weight: np.ndarray | None = None

    directed: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ei = _as_int64(self.edge_index)
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise GraphValidationError("edge_index must have shape (2, E)")
        if ei.size and (ei.min() < 0 or ei.max() >= self.n_nodes):
            raise GraphValidationError("edge_index contains node ids outside [0, n_nodes)")
        object.__setattr__(self, "edge_index", ei)

        if self.edge_weight is not None:
            ew = _as_float32(self.edge_weight)
            if ew.ndim != 1 or ew.shape[0] != ei.shape[1]:
                raise GraphValidationError("edge_weight must have shape (E,)")
            object.__setattr__(self, "edge_weight", ew)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": int(self.n_nodes),
            "n_edges": int(self.n_edges),
            "directed": bool(self.directed),
            "has_edge_weight": self.edge_weight is not None,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class NodeDataset:
    """Node classification dataset for transductive methods."""

    X: Any
    y: np.ndarray
    graph: GraphArtifact
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate X first dimension (works for numpy and scipy sparse)
        if not hasattr(self.X, "shape"):
            raise GraphValidationError("X must expose a shape attribute")
        if int(self.X.shape[0]) != int(self.graph.n_nodes):
            raise GraphValidationError("X must have shape (n_nodes, d)")

        y = _as_int64(self.y)
        if y.ndim not in (1, 2):
            raise GraphValidationError("y must have shape (n,) or (n, C)")
        if y.shape[0] != self.graph.n_nodes:
            raise GraphValidationError("y must have the same first dimension as graph.n_nodes")
        object.__setattr__(self, "y", y)

        new_masks: dict[str, np.ndarray] = {}
        for k, v in self.masks.items():
            m = np.asarray(v, dtype=bool)
            if m.ndim != 1 or m.shape[0] != self.graph.n_nodes:
                raise GraphValidationError(f"Mask {k!r} must have shape (n_nodes,)")
            new_masks[str(k)] = m
        object.__setattr__(self, "masks", new_masks)


@dataclass(frozen=True)
class DatasetViews:
    """One or more tabular views derived from a dataset."""

    views: dict[str, Any]
    y: np.ndarray
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        y = _as_int64(self.y)
        object.__setattr__(self, "y", y)

        if not self.views:
            raise GraphValidationError("views cannot be empty")

        # validate consistent first dimension
        n: int | None = None
        for name, v in self.views.items():
            if not hasattr(v, "shape"):
                raise GraphValidationError(f"view {name!r} must expose a shape attribute")
            if int(v.ndim) != 2:
                raise GraphValidationError(f"view {name!r} must be 2D")
            if n is None:
                n = int(v.shape[0])
            elif int(v.shape[0]) != n:
                raise GraphValidationError("All views must have the same number of samples")

        assert n is not None
        if y.shape[0] != n:
            raise GraphValidationError("y must have the same first dimension as views")

        new_masks: dict[str, np.ndarray] = {}
        for k, v in self.masks.items():
            m = np.asarray(v, dtype=bool)
            if m.ndim != 1 or m.shape[0] != n:
                raise GraphValidationError(f"Mask {k!r} must have shape (n,)")
            new_masks[str(k)] = m
        object.__setattr__(self, "masks", new_masks)
