from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.errors import OptionalDependencyError
from modssc.transductive.methods.utils import DiffusionResult, _validate_graph_inputs
from modssc.transductive.optional import optional_import
from modssc.transductive.validation import validate_node_dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphMincutsSpec:
    """Graph Mincuts (Blum & Chawla, 2001) — transductive semi-supervised learning.

    This implementation focuses on the *binary* s-t min-cut formulation:
      - All labeled nodes of class A are forced to the SOURCE side.
      - All labeled nodes of class B are forced to the SINK side.
      - Unlabeled nodes are assigned by the minimum cut.

    Notes
    -----
    * Multi-class extensions exist (multiway cut / alpha-expansion), but are out of scope
      for this wave. We raise if more than 2 labeled classes are present.
    * We rely on SciPy's `scipy.sparse.csgraph.maximum_flow` for performance.
      Capacities must be integers, so edge weights are scaled and rounded.

    Parameters
    ----------
    backend:
        Currently only "scipy" is supported.
    capacity_scale:
        Positive scaling applied to edge weights before rounding to integer capacities.
        If your graph uses binary weights, you can keep this at 1.
    min_capacity:
        After rounding, capacities are clipped to at least this value (avoids 0-capacity edges).
    """

    backend: Literal["scipy"] = "scipy"
    capacity_scale: float = 1000.0
    min_capacity: int = 1


def _reachable_from_source_csr(residual_csr, source: int) -> np.ndarray:
    """Return boolean mask of nodes reachable from `source` in residual graph (CSR)."""
    n_total = int(residual_csr.shape[0])
    visited = np.zeros(n_total, dtype=bool)
    stack = [int(source)]
    visited[int(source)] = True

    indptr = residual_csr.indptr
    indices = residual_csr.indices
    data = residual_csr.data

    while stack:
        u = stack.pop()
        start = int(indptr[u])
        end = int(indptr[u + 1])
        for j in range(start, end):
            v = int(indices[j])
            cap = data[j]
            if cap > 0 and not visited[v]:
                visited[v] = True
                stack.append(v)

    return visited


def graph_mincuts(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: GraphMincutsSpec | None = None,
) -> DiffusionResult:
    """Run Graph Mincuts (binary) on a pre-built graph.

    Parameters
    ----------
    n_nodes:
        Number of nodes.
    edge_index:
        (2, E) integer array of directed edges.
    edge_weight:
        (E,) edge weights. If None, uses 1.0 for all edges.
    y:
        (n_nodes,) integer labels for all nodes (only `labeled_mask` is used as supervision).
    labeled_mask:
        (n_nodes,) boolean mask of labeled nodes.
    spec:
        GraphMincutsSpec.

    Returns
    -------
    DiffusionResult
        `F` is a (n_nodes, 2) score matrix with hard 0/1 assignments from the min-cut
        (column 1 = SOURCE side, column 0 = SINK side).
    """
    if spec is None:
        spec = GraphMincutsSpec()

    # Validate arrays + normalize shapes
    edge_index, edge_weight = _validate_graph_inputs(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

    y = np.asarray(y).reshape(-1).astype(np.int64, copy=False)
    if y.shape[0] != int(n_nodes):
        raise ValueError(f"y must have shape (n_nodes,), got {y.shape}")

    labeled_mask = np.asarray(labeled_mask).reshape(-1).astype(bool, copy=False)
    if labeled_mask.shape[0] != int(n_nodes):
        raise ValueError(f"labeled_mask must have shape (n_nodes,), got {labeled_mask.shape}")

    labeled_idx = np.flatnonzero(labeled_mask)
    if labeled_idx.size == 0:
        raise ValueError("GraphMincuts requires at least 1 labeled node.")

    classes = np.unique(y[labeled_idx])
    if classes.size != 2:
        raise ValueError(
            f"GraphMincuts (this wave) supports only binary problems. Got classes={classes.tolist()}."
        )

    # Map to {0,1} deterministically by sorted class id
    classes_sorted = np.sort(classes)
    neg_class = int(classes_sorted[0])
    pos_class = int(classes_sorted[1])

    pos_nodes = labeled_idx[y[labeled_idx] == pos_class]
    neg_nodes = labeled_idx[y[labeled_idx] == neg_class]

    if pos_nodes.size == 0 or neg_nodes.size == 0:
        raise ValueError(
            f"Need at least one labeled example per class. Got pos={pos_nodes.size} neg={neg_nodes.size}."
        )

    if not np.isfinite(edge_weight).all():
        raise ValueError("edge_weight contains non-finite values.")

    if spec.capacity_scale <= 0:
        raise ValueError("capacity_scale must be > 0.")

    # Optional dependency: SciPy
    sp = optional_import("scipy.sparse", extra="sklearn")
    try:
        maximum_flow = optional_import("scipy.sparse.csgraph", extra="sklearn").maximum_flow
    except Exception as e:
        raise OptionalDependencyError("scipy.sparse.csgraph.maximum_flow", extra="sklearn") from e

    # Convert capacities to integer (SciPy max-flow requires integers)
    cap = np.rint(np.asarray(edge_weight, dtype=np.float64) * float(spec.capacity_scale)).astype(
        np.int64
    )
    cap = np.maximum(cap, int(spec.min_capacity))

    # Terminals
    source = int(n_nodes)
    sink = int(n_nodes) + 1
    n_total = int(n_nodes) + 2

    # "Infinity" capacity: larger than any possible cut through graph edges
    total_cap = int(cap.sum())
    inf_cap = int(total_cap + 1)

    # Build sparse capacity matrix (directed)
    rows = []
    cols = []
    data = []

    # Graph edges
    rows.append(edge_index[0].astype(np.int64, copy=False))
    cols.append(edge_index[1].astype(np.int64, copy=False))
    data.append(cap.astype(np.int64, copy=False))

    # Source -> positive labeled (force to source side)
    rows.append(np.full(pos_nodes.shape[0], source, dtype=np.int64))
    cols.append(pos_nodes.astype(np.int64, copy=False))
    data.append(np.full(pos_nodes.shape[0], inf_cap, dtype=np.int64))

    # Negative labeled -> sink (force to sink side)
    rows.append(neg_nodes.astype(np.int64, copy=False))
    cols.append(np.full(neg_nodes.shape[0], sink, dtype=np.int64))
    data.append(np.full(neg_nodes.shape[0], inf_cap, dtype=np.int64))

    r = np.concatenate(rows)
    c = np.concatenate(cols)
    d = np.concatenate(data)

    capacity = sp.coo_matrix((d, (r, c)), shape=(n_total, n_total), dtype=np.int64).tocsr()

    # Max-flow / min-cut
    try:
        flow_res = maximum_flow(capacity, source, sink)
    except Exception as e:
        raise RuntimeError(f"SciPy maximum_flow failed: {type(e).__name__}: {e}") from e

    # Residual graph: SciPy returns a signed flow matrix; residual = capacity - flow
    residual = capacity - flow_res.flow

    reachable = _reachable_from_source_csr(residual, source)
    src_side = reachable[: int(n_nodes)]

    # Scores: hard assignment from the partition
    F = np.zeros((int(n_nodes), 2), dtype=np.float32)
    F[:, 1] = src_side.astype(np.float32)
    F[:, 0] = 1.0 - F[:, 1]

    # Sanity: labeled constraints must be satisfied (due to inf capacities)
    if not np.all(F[pos_nodes, 1] == 1.0) or not np.all(F[neg_nodes, 0] == 1.0):
        raise RuntimeError(
            "Mincut constraints not satisfied; check capacity scaling / graph validity."
        )

    return DiffusionResult(F=F, n_iter=0, residual=0.0)


class GraphMincutsMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="graph_mincuts",
        name="Graph Mincuts",
        year=2001,
        family="cut",
        supports_gpu=False,
        required_extra="sklearn",
        paper_title="Learning from Labeled and Unlabeled Data using Graph Mincuts",
    )

    def __init__(self, spec: GraphMincutsSpec | None = None) -> None:
        self.spec = spec or GraphMincutsSpec()
        self._result: DiffusionResult | None = None

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> GraphMincutsMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        validate_node_dataset(data)

        masks = getattr(data, "masks", None) or {}
        if "train_mask" not in masks:
            raise ValueError("data.masks must contain 'train_mask'")

        labeled_mask = np.asarray(masks["train_mask"], dtype=bool)
        g = data.graph
        logger.info(
            "Graph mincuts sizes: n_nodes=%s labeled=%s",
            int(np.asarray(data.y).shape[0]),
            int(labeled_mask.sum()),
        )

        self._result = graph_mincuts(
            n_nodes=int(np.asarray(data.y).shape[0]),
            edge_index=np.asarray(g.edge_index),
            edge_weight=(
                None if getattr(g, "edge_weight", None) is None else np.asarray(g.edge_weight)
            ),
            y=np.asarray(data.y),
            labeled_mask=labeled_mask,
            spec=self.spec,
        )
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("GraphMincutsMethod is not fitted yet. Call fit() first.")
        return np.asarray(self._result.F, dtype=np.float32)
