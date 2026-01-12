from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.methods.utils import (
    DiffusionResult,
    _validate_graph_inputs,
    degrees_numpy,
    spmm_numpy,
)
from modssc.transductive.operators.clamp import labels_to_onehot
from modssc.transductive.validation import validate_node_dataset

logger = logging.getLogger(__name__)


def _symmetrize_edges(
    edge_index: np.ndarray, edge_weight: np.ndarray, *, zero_diagonal: bool
) -> tuple[np.ndarray, np.ndarray]:
    rev = edge_index[[1, 0], :]
    edge_index = np.concatenate([edge_index, rev], axis=1)
    edge_weight = np.concatenate([edge_weight, edge_weight], axis=0)
    if zero_diagonal:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]
    return edge_index, edge_weight


def _proj_vertices(U: np.ndarray) -> np.ndarray:
    idx = np.argmax(U, axis=1)
    out = np.zeros_like(U, dtype=np.float32)
    out[np.arange(U.shape[0]), idx] = 1.0
    return out


def _build_b_matrix(
    *,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    Y = labels_to_onehot(y, n_classes=n_classes).astype(np.float32, copy=False)
    Y[~labeled_mask] = 0.0

    m = int(labeled_mask.sum())
    if m <= 0:
        raise ValueError("PoissonMBO requires at least 1 labeled node.")

    y_bar = Y[labeled_mask].mean(axis=0)
    if np.any(y_bar <= 0.0):
        raise ValueError("PoissonMBO requires at least one labeled node per class.")

    B = np.zeros_like(Y, dtype=np.float32)
    B[labeled_mask] = Y[labeled_mask] - y_bar
    return B, y_bar


def _build_b_prior(
    *,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    n_classes: int,
    strategy: Literal["uniform", "labeled", "true"],
    y_bar: np.ndarray,
) -> np.ndarray:
    if strategy == "uniform":
        b = np.full((n_classes,), 1.0 / float(n_classes), dtype=np.float32)
    elif strategy == "labeled":
        b = y_bar.astype(np.float32, copy=False)
    elif strategy == "true":
        y_valid = y[y >= 0]
        if y_valid.size == 0:
            raise ValueError("PoissonMBO requires at least one valid label for b=true.")
        counts = np.bincount(y_valid.astype(np.int64), minlength=n_classes).astype(np.float32)
        if float(counts.sum()) <= 0.0:
            raise ValueError("PoissonMBO b=true requires at least one valid label.")
        b = counts / float(counts.sum())
    else:
        raise ValueError(f"Unknown b_strategy: {strategy!r}")
    return b.astype(np.float32, copy=False)


def poisson_mbo_numpy(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: PoissonMBOSpec | None = None,
) -> DiffusionResult:
    if spec is None:
        spec = PoissonMBOSpec()

    edge_index, w = _validate_graph_inputs(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight
    )
    if spec.symmetrize:
        edge_index, w = _symmetrize_edges(edge_index, w, zero_diagonal=bool(spec.zero_diagonal))
    elif spec.zero_diagonal:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        w = w[mask]

    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if y.shape != (n_nodes,):
        raise ValueError("y must have shape (n_nodes,)")
    labeled_mask = np.asarray(labeled_mask, dtype=bool).reshape(-1)
    if labeled_mask.shape != (n_nodes,):
        raise ValueError("labeled_mask must have shape (n_nodes,)")

    y_valid = y[y >= 0]
    if y_valid.size == 0:
        raise ValueError("y must contain at least one valid label.")
    n_classes = int(y_valid.max()) + 1

    B, y_bar = _build_b_matrix(y=y, labeled_mask=labeled_mask, n_classes=n_classes)

    if spec.b is not None:
        b = np.asarray(spec.b, dtype=np.float32).reshape(-1)
        if b.shape != (n_classes,):
            raise ValueError(f"b must have shape ({n_classes},), got {b.shape}")
        if np.any(b < 0.0):
            raise ValueError("b must be non-negative.")
        if float(b.sum()) <= 0.0:
            raise ValueError("b must sum to a positive value.")
        b = b / float(b.sum())
    else:
        b = _build_b_prior(
            y=y,
            labeled_mask=labeled_mask,
            n_classes=n_classes,
            strategy=spec.b_strategy,
            y_bar=y_bar,
        )

    deg = degrees_numpy(n_nodes=n_nodes, edge_index=edge_index, edge_weight=w)
    max_deg = float(deg.max(initial=0.0))
    if max_deg <= 0.0:
        raise ValueError("PoissonMBO requires at least one edge with positive weight.")
    inv_deg = np.zeros_like(deg, dtype=np.float32)
    inv_mask = deg > 0
    inv_deg[inv_mask] = 1.0 / deg[inv_mask]

    U = np.zeros((n_nodes, n_classes), dtype=np.float32)
    residual = float("inf")

    # PoissonLearning iteration (Algorithm 1).
    for _ in range(int(spec.T)):
        WU = spmm_numpy(n_nodes=n_nodes, edge_index=edge_index, edge_weight=w, X=U)
        LU = deg[:, None] * U - WU
        update = inv_deg[:, None] * (B - LU)
        U = U + update
        residual = float(np.max(np.abs(update)))

    scale = b / y_bar
    U = U * scale[None, :]

    dt = 1.0 / max_deg

    for _ in range(int(spec.Nouter)):
        for _ in range(int(spec.Ninner)):
            WU = spmm_numpy(n_nodes=n_nodes, edge_index=edge_index, edge_weight=w, X=U)
            LU = deg[:, None] * U - WU
            update = -dt * (LU - float(spec.mu) * B)
            U = U + update
            residual = float(np.max(np.abs(update)))

        s = np.ones((n_classes,), dtype=np.float32)
        for _ in range(int(spec.n_volume_iters)):
            U_proj = _proj_vertices(U * s[None, :])
            b_hat = U_proj.mean(axis=0)
            s = s + float(spec.d_tau) * (b - b_hat)
            s = np.clip(s, float(spec.smin), float(spec.smax))

        U = _proj_vertices(U * s[None, :])

    total_iter = int(spec.T) + int(spec.Nouter) * int(spec.Ninner)
    return DiffusionResult(F=U, n_iter=total_iter, residual=residual)


@dataclass(frozen=True)
class PoissonMBOSpec:
    """Poisson MBO (Calder et al.) — volume constrained MBO with Poisson fidelity."""

    T: int = 200
    Ninner: int = 40
    Nouter: int = 20
    mu: float = 1.0
    d_tau: float = 10.0
    smin: float = 0.5
    smax: float = 2.0
    n_volume_iters: int = 100
    b_strategy: Literal["uniform", "labeled", "true"] = "uniform"
    b: np.ndarray | None = None
    symmetrize: bool = True
    zero_diagonal: bool = True


def poisson_mbo(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: PoissonMBOSpec | None = None,
) -> DiffusionResult:
    return poisson_mbo_numpy(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=spec,
    )


class PoissonMBOMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="poisson_mbo",
        name="Poisson MBO",
        year=2020,
        family="pde",
        supports_gpu=False,
        paper_title="Poisson Learning: Graph Based Semi-Supervised Learning at Very Low Label Rates",
        paper_pdf="https://arxiv.org/abs/2006.12037",
        official_code="https://github.com/jwcalder/GraphLearning",
    )

    def __init__(self, spec: PoissonMBOSpec | None = None) -> None:
        self.spec = spec or PoissonMBOSpec()
        self._result: DiffusionResult | None = None

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> PoissonMBOMethod:
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
            "Poisson MBO sizes: n_nodes=%s labeled=%s",
            int(np.asarray(data.y).shape[0]),
            int(labeled_mask.sum()),
        )

        self._result = poisson_mbo(
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
            raise RuntimeError("PoissonMBOMethod is not fitted yet. Call fit() first.")
        return np.asarray(self._result.F, dtype=np.float32)
