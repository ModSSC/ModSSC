from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.methods.utils import DiffusionResult, _validate_graph_inputs
from modssc.transductive.operators.clamp import labels_to_onehot
from modssc.transductive.solvers.cg import cg_solve_numpy
from modssc.transductive.validation import validate_node_dataset

logger = logging.getLogger(__name__)


def _coalesce_edges(
    edge_index: np.ndarray, edge_weight: np.ndarray, *, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    keys = src * int(n_nodes) + dst
    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    w = edge_weight[order].astype(np.float32, copy=False)

    uniq, idx = np.unique(keys, return_index=True)
    w_sum = np.add.reduceat(w, idx)
    src_u = (uniq // int(n_nodes)).astype(np.int64, copy=False)
    dst_u = (uniq % int(n_nodes)).astype(np.int64, copy=False)
    return np.vstack([src_u, dst_u]), w_sum


def _symmetrize_edges(
    edge_index: np.ndarray, edge_weight: np.ndarray, *, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    rev = edge_index[[1, 0], :]
    edge_index2 = np.concatenate([edge_index, rev], axis=1)
    edge_weight2 = np.concatenate([edge_weight, edge_weight], axis=0)
    return _coalesce_edges(edge_index2, edge_weight2, n_nodes=n_nodes)


def _spmm_vec(
    *, n_nodes: int, edge_index: np.ndarray, edge_weight: np.ndarray, x: np.ndarray
) -> np.ndarray:
    src = edge_index[0]
    dst = edge_index[1]
    out = np.zeros((n_nodes,), dtype=np.float32)
    np.add.at(out, dst, edge_weight * x[src])
    return out


@dataclass(frozen=True)
class PLaplaceLearningSpec:
    """Variational p-Laplace learning via Newton's method."""

    p: float = 3.0
    max_iter: int = 50
    tol: float = 1e-6
    cg_tol: float = 1e-6
    cg_max_iter: int = 2000
    eps: float = 1e-12
    symmetrize: bool = True
    zero_diagonal: bool = True


def p_laplace_learning_numpy(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: PLaplaceLearningSpec | None = None,
) -> DiffusionResult:
    if spec is None:
        spec = PLaplaceLearningSpec()

    p = float(spec.p)
    if p < 2.0:
        raise ValueError("p must be >= 2 for variational p-Laplace learning")

    edge_index, w = _validate_graph_inputs(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight
    )

    if spec.symmetrize:
        edge_index, w = _symmetrize_edges(edge_index, w, n_nodes=n_nodes)
    if spec.zero_diagonal:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        w = w[mask]

    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if y.shape != (n_nodes,):
        raise ValueError("y must have shape (n_nodes,)")
    labeled_mask = np.asarray(labeled_mask, dtype=bool).reshape(-1)
    if labeled_mask.shape != (n_nodes,):
        raise ValueError("labeled_mask must have shape (n_nodes,)")
    if not labeled_mask.any():
        raise ValueError("p-Laplace Learning requires at least 1 labeled node.")

    y_valid = y[y >= 0]
    if y_valid.size == 0:
        raise ValueError("y must contain at least one valid label.")
    n_classes = int(y_valid.max()) + 1
    label_counts = np.bincount(y[labeled_mask & (y >= 0)], minlength=n_classes)
    if np.any(label_counts == 0):
        raise ValueError("p-Laplace Learning requires at least one labeled node per class.")
    Y = labels_to_onehot(y, n_classes=n_classes).astype(np.float32, copy=False)
    Y[~labeled_mask] = 0.0

    unlabeled_idx = np.flatnonzero(~labeled_mask)
    labeled_idx = np.flatnonzero(labeled_mask)
    if unlabeled_idx.size == 0:
        return DiffusionResult(F=Y, n_iter=0, residual=0.0)

    map_u = -np.ones((n_nodes,), dtype=np.int64)
    map_l = -np.ones((n_nodes,), dtype=np.int64)
    map_u[unlabeled_idx] = np.arange(unlabeled_idx.size, dtype=np.int64)
    map_l[labeled_idx] = np.arange(labeled_idx.size, dtype=np.int64)

    src = edge_index[0]
    dst = edge_index[1]
    src_u = map_u[src]
    dst_u = map_u[dst]
    src_l = map_l[src]

    mask_uu = (src_u >= 0) & (dst_u >= 0)
    edge_index_uu = np.vstack([src_u[mask_uu], dst_u[mask_uu]])
    w_uu = w[mask_uu]

    mask_ul = (src_l >= 0) & (dst_u >= 0)
    edge_index_ul = np.vstack([src_l[mask_ul], dst_u[mask_ul]])
    w_ul = w[mask_ul]
    if edge_index_ul.shape[1] == 0:
        raise ValueError("p-Laplace Learning requires unlabeled nodes connected to labels.")

    g_all = Y[labeled_idx]
    f = np.zeros((unlabeled_idx.size,), dtype=np.float32)

    F_u = np.zeros((unlabeled_idx.size, n_classes), dtype=np.float32)
    n_iter_max = 0
    residual_max = 0.0

    # p=2 initialization (L(u) independent of u)
    def solve_linear(
        *, a_weight: np.ndarray, b_weight: np.ndarray, g: np.ndarray
    ) -> tuple[np.ndarray, int, float]:
        D = np.zeros((unlabeled_idx.size,), dtype=np.float32)
        if a_weight.size:
            np.add.at(D, edge_index_uu[1], a_weight)
        if b_weight.size:
            np.add.at(D, edge_index_ul[1], b_weight)
        D = D + float(spec.eps)

        def matvec(x: np.ndarray) -> np.ndarray:
            Ax = (
                _spmm_vec(
                    n_nodes=unlabeled_idx.size,
                    edge_index=edge_index_uu,
                    edge_weight=a_weight,
                    x=x,
                )
                if a_weight.size
                else np.zeros_like(x)
            )
            return D * x - Ax

        rhs = np.zeros((unlabeled_idx.size,), dtype=np.float32)
        if b_weight.size:
            np.add.at(rhs, edge_index_ul[1], b_weight * g[edge_index_ul[0]])
        rhs = rhs - f

        cg = cg_solve_numpy(
            matvec=matvec, b=rhs, tol=float(spec.cg_tol), max_iter=int(spec.cg_max_iter)
        )
        return cg.x.astype(np.float32, copy=False), int(cg.n_iter), float(cg.residual_norm)

    for c in range(n_classes):
        g = g_all[:, c].astype(np.float32, copy=False)

        a_weight = w_uu.astype(np.float32, copy=False)
        b_weight = w_ul.astype(np.float32, copy=False)
        u, n_iter, residual = solve_linear(a_weight=a_weight, b_weight=b_weight, g=g)

        if p > 2.0:
            for it in range(int(spec.max_iter)):
                diff = u[edge_index_uu[0]] - u[edge_index_uu[1]]
                a_weight = w_uu * np.abs(diff) ** (p - 2.0)

                diff_b = u[edge_index_ul[1]] - g[edge_index_ul[0]]
                b_weight = w_ul * np.abs(diff_b) ** (p - 2.0)

                v, n_iter_cg, residual_cg = solve_linear(
                    a_weight=a_weight.astype(np.float32, copy=False),
                    b_weight=b_weight.astype(np.float32, copy=False),
                    g=g,
                )
                u_new = ((p - 2.0) / (p - 1.0)) * u + (1.0 / (p - 1.0)) * v
                residual = float(np.max(np.abs(u_new - u)))
                u = u_new
                n_iter = it + 1
                if residual <= float(spec.tol):
                    break
            n_iter_max = max(n_iter_max, int(n_iter))
            residual_max = max(residual_max, float(residual))
        else:
            n_iter_max = max(n_iter_max, int(n_iter))
            residual_max = max(residual_max, float(residual))

        F_u[:, c] = u

    F = np.zeros((n_nodes, n_classes), dtype=np.float32)
    F[labeled_idx] = Y[labeled_idx]
    F[unlabeled_idx] = F_u

    # Convert unlabeled scores to probabilities; keep labeled nodes as one-hot.
    proba = F.copy()
    logits = F_u - np.max(F_u, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    proba_u = exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)
    proba[unlabeled_idx] = proba_u
    proba[labeled_idx] = Y[labeled_idx]

    return DiffusionResult(F=proba, n_iter=n_iter_max, residual=residual_max)


def p_laplace_learning(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: PLaplaceLearningSpec | None = None,
) -> DiffusionResult:
    return p_laplace_learning_numpy(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=spec,
    )


class PLaplaceLearningMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="p_laplace_learning",
        name="p-Laplace Learning",
        year=2022,
        family="pde",
        supports_gpu=False,
        paper_title="Analysis and Algorithms for lp-based Semi-Supervised Learning on Graphs",
    )

    def __init__(self, spec: PLaplaceLearningSpec | None = None) -> None:
        self.spec = spec or PLaplaceLearningSpec()
        self._result: DiffusionResult | None = None

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> PLaplaceLearningMethod:
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
            "p-Laplace sizes: n_nodes=%s labeled=%s",
            int(np.asarray(data.y).shape[0]),
            int(labeled_mask.sum()),
        )

        self._result = p_laplace_learning_numpy(
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
            raise RuntimeError("PLaplaceLearningMethod is not fitted yet. Call fit() first.")
        return np.asarray(self._result.F, dtype=np.float32)
