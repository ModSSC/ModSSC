from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.graph.construction.ops.symmetrize import symmetrize_edges as _symmetrize_mean_edges
from modssc.runtime.device import resolve_device_name
from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.methods.pde.poisson_learning import (
    PoissonLearningSpec,
    poisson_learning_numpy,
    poisson_learning_torch,
)
from modssc.transductive.methods.utils import (
    DiffusionResult,
    _validate_graph_inputs,
    degrees_numpy,
    degrees_torch,
    spmm_numpy,
    spmm_torch,
    to_numpy,
)
from modssc.transductive.operators.clamp import labels_to_onehot
from modssc.transductive.optional import optional_import
from modssc.transductive.validation import validate_node_dataset

logger = logging.getLogger(__name__)


def _coalesce_edges(
    edge_index: np.ndarray, edge_weight: np.ndarray, *, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    if edge_index.shape[1] == 0:
        return edge_index.astype(np.int64, copy=False), edge_weight.astype(np.float32, copy=False)
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    keys = src * int(n_nodes) + dst
    order = np.argsort(keys, kind="mergesort")
    keys_s = keys[order]
    w_s = edge_weight[order].astype(np.float32, copy=False)
    uniq, starts = np.unique(keys_s, return_index=True)
    w_sum = np.add.reduceat(w_s, starts).astype(np.float32, copy=False)
    src_u = (uniq // int(n_nodes)).astype(np.int64, copy=False)
    dst_u = (uniq % int(n_nodes)).astype(np.int64, copy=False)
    return np.vstack([src_u, dst_u]), w_sum


def _symmetrize_edges(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    *,
    zero_diagonal: bool,
    n_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper-style symmetrization without doubling an already symmetric graph."""

    n = int(n_nodes) if n_nodes is not None else int(edge_index.max(initial=-1)) + 1
    edge_index, edge_weight = _symmetrize_mean_edges(
        n_nodes=n,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mode="mean",
    )
    assert edge_weight is not None
    if zero_diagonal:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]
    return _coalesce_edges(edge_index, edge_weight, n_nodes=n)


def _proj_vertices(U: np.ndarray) -> np.ndarray:
    idx = np.argmax(U, axis=1)
    out = np.zeros_like(U, dtype=np.float32)
    out[np.arange(U.shape[0]), idx] = 1.0
    return out


def _predict_with_weights(U: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = np.asarray(U, dtype=np.float32)
    scores = scores - float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score > 0.0:
        scores = scores / max_score
    return np.argmax(scores * weights[None, :], axis=1).astype(np.int64, copy=False)


def _volume_label_projection(
    U: np.ndarray,
    *,
    class_priors: np.ndarray,
    weights: np.ndarray,
    tol: float,
    max_iter: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Apply the paper volume projection for similarity scores."""

    k = int(U.shape[1])
    err = float("inf")
    labels = _predict_with_weights(U, weights)
    for i in range(int(max_iter)):
        class_size = labels_to_onehot(labels, n_classes=k).mean(axis=0)
        grad = class_size - class_priors
        err = float(np.max(np.abs(grad)))
        if err <= float(tol):
            return labels, weights.astype(np.float32, copy=False), err, i
        weights = weights - float(dt) * grad.astype(np.float32, copy=False)
        if float(weights[0]) != 0.0:
            weights = weights / float(weights[0])
        labels = _predict_with_weights(U, weights)
    return labels, weights.astype(np.float32, copy=False), err, int(max_iter)


def _onehot_torch(labels: Any, *, n_classes: int) -> Any:
    torch = optional_import("torch", extra="transductive-torch")
    labels = labels.to(dtype=torch.long).view(-1)
    out = torch.zeros(
        (int(labels.numel()), int(n_classes)), dtype=torch.float32, device=labels.device
    )
    out[torch.arange(int(labels.numel()), device=labels.device), labels] = 1.0
    return out


def _predict_with_weights_torch(U: Any, weights: Any) -> Any:
    torch = optional_import("torch", extra="transductive-torch")
    scores = U.to(dtype=torch.float32)
    scores = scores - torch.min(scores)
    max_score = torch.max(scores)
    if float(max_score.detach().cpu()) > 0.0:
        scores = scores / max_score
    return torch.argmax(scores * weights.view(1, -1), dim=1).to(dtype=torch.long)


def _volume_label_projection_torch(
    U: Any,
    *,
    class_priors: Any,
    weights: Any,
    tol: float,
    max_iter: int,
    dt: float,
) -> tuple[Any, Any, float, int]:
    torch = optional_import("torch", extra="transductive-torch")
    k = int(U.shape[1])
    err = float("inf")
    labels = _predict_with_weights_torch(U, weights)
    for i in range(int(max_iter)):
        class_size = torch.bincount(labels, minlength=k).to(dtype=U.dtype) / float(labels.numel())
        grad = class_size - class_priors
        err = float(torch.max(torch.abs(grad)).detach().cpu())
        if err <= float(tol):
            return labels, weights.to(dtype=torch.float32), err, i
        weights = weights - float(dt) * grad.to(dtype=torch.float32)
        if float(weights[0].detach().cpu()) != 0.0:
            weights = weights / weights[0]
        labels = _predict_with_weights_torch(U, weights)
    return labels, weights.to(dtype=torch.float32), err, int(max_iter)


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
    strategy: Literal["uniform", "labeled", "true"] | bool,
    y_bar: np.ndarray,
) -> np.ndarray:
    if isinstance(strategy, bool):
        strategy = "true" if strategy else "uniform"
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
    if bool(getattr(spec, "symmetrize", True)):
        edge_index, w = _symmetrize_edges(
            edge_index, w, zero_diagonal=bool(getattr(spec, "zero_diagonal", True)), n_nodes=n_nodes
        )
    elif spec.zero_diagonal:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        w = w[mask]
        edge_index, w = _coalesce_edges(edge_index, w, n_nodes=n_nodes)
    else:
        edge_index, w = _coalesce_edges(edge_index, w, n_nodes=n_nodes)

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

    poisson_res = poisson_learning_numpy(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=w,
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonLearningSpec(
            backend="numpy",
            laplacian_kind="paper_normalized",
            eps=float(getattr(spec, "poisson_eps", 0.0)),
            center_sources=True,
            tol=float(getattr(spec, "tol", 1.0e-3)),
            max_iter=int(getattr(spec, "max_iter", 1000)),
        ),
    )
    labels = np.argmax(poisson_res.F, axis=1).astype(np.int64, copy=False)
    U = labels_to_onehot(labels, n_classes=n_classes).astype(np.float32, copy=False)

    dt = 1.0 / max_deg
    Db = float(spec.mu) * dt * B
    weights = np.ones((n_classes,), dtype=np.float32)
    residual = float(poisson_res.residual)
    volume_residual = float("inf")

    ns = int(getattr(spec, "Ninner", None) or getattr(spec, "Ns", 40))
    outer = int(getattr(spec, "Nouter", None) or getattr(spec, "T", 20))

    for _ in range(outer):
        for _ in range(ns):
            WU = spmm_numpy(n_nodes=n_nodes, edge_index=edge_index, edge_weight=w, X=U)
            LU = deg[:, None] * U - WU
            update = -dt * LU + Db
            U = U + update
            residual = float(np.max(np.abs(update)))

        labels, weights, volume_residual, _ = _volume_label_projection(
            U,
            class_priors=b,
            weights=weights,
            tol=float(getattr(spec, "volume_tol", 1.0e-3)),
            max_iter=int(getattr(spec, "volume_max_iter", 10_000)),
            dt=float(getattr(spec, "volume_dt", 0.1)),
        )
        U = labels_to_onehot(labels, n_classes=n_classes).astype(np.float32, copy=False)

    total_iter = int(poisson_res.n_iter) + outer * ns
    return DiffusionResult(F=U, n_iter=total_iter, residual=max(residual, volume_residual))


def poisson_mbo_torch(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any | None,
    y: Any,
    labeled_mask: Any,
    spec: PoissonMBOSpec | None = None,
    device: str | None = None,
) -> DiffusionResult:
    torch = optional_import("torch", extra="transductive-torch")
    if spec is None:
        spec = PoissonMBOSpec()

    edge_index, w = _validate_graph_inputs(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight
    )
    if bool(getattr(spec, "symmetrize", True)):
        edge_index, w = _symmetrize_edges(
            edge_index, w, zero_diagonal=bool(getattr(spec, "zero_diagonal", True)), n_nodes=n_nodes
        )
    elif spec.zero_diagonal:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        w = w[mask]
        edge_index, w = _coalesce_edges(edge_index, w, n_nodes=n_nodes)
    else:
        edge_index, w = _coalesce_edges(edge_index, w, n_nodes=n_nodes)

    y_np = np.asarray(y, dtype=np.int64).reshape(-1)
    if y_np.shape != (n_nodes,):
        raise ValueError("y must have shape (n_nodes,)")
    labeled_mask_np = np.asarray(labeled_mask, dtype=bool).reshape(-1)
    if labeled_mask_np.shape != (n_nodes,):
        raise ValueError("labeled_mask must have shape (n_nodes,)")

    y_valid = y_np[y_np >= 0]
    if y_valid.size == 0:
        raise ValueError("y must contain at least one valid label.")
    n_classes = int(y_valid.max()) + 1

    B_np, y_bar = _build_b_matrix(y=y_np, labeled_mask=labeled_mask_np, n_classes=n_classes)

    if spec.b is not None:
        b_np = np.asarray(spec.b, dtype=np.float32).reshape(-1)
        if b_np.shape != (n_classes,):
            raise ValueError(f"b must have shape ({n_classes},), got {b_np.shape}")
        if np.any(b_np < 0.0):
            raise ValueError("b must be non-negative.")
        if float(b_np.sum()) <= 0.0:
            raise ValueError("b must sum to a positive value.")
        b_np = b_np / float(b_np.sum())
    else:
        b_np = _build_b_prior(
            y=y_np,
            labeled_mask=labeled_mask_np,
            n_classes=n_classes,
            strategy=spec.b_strategy,
            y_bar=y_bar,
        )

    dev_name = resolve_device_name(device, torch=torch) or "cpu"
    dev = torch.device(dev_name)
    edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=dev)
    w_t = torch.as_tensor(w, dtype=torch.float32, device=dev)
    y_t = torch.as_tensor(y_np, dtype=torch.long, device=dev)
    labeled_mask_t = torch.as_tensor(labeled_mask_np, dtype=torch.bool, device=dev)
    B_t = torch.as_tensor(B_np, dtype=torch.float32, device=dev)
    b_t = torch.as_tensor(b_np, dtype=torch.float32, device=dev)

    deg = degrees_torch(n_nodes=n_nodes, edge_index=edge_index_t, edge_weight=w_t)
    max_deg = float(torch.max(deg).detach().cpu()) if int(deg.numel()) else 0.0
    if max_deg <= 0.0:
        raise ValueError("PoissonMBO requires at least one edge with positive weight.")

    poisson_res = poisson_learning_torch(
        n_nodes=n_nodes,
        edge_index=edge_index_t,
        edge_weight=w_t,
        y=y_t,
        labeled_mask=labeled_mask_t,
        spec=PoissonLearningSpec(
            backend="torch",
            laplacian_kind="paper_normalized",
            eps=float(getattr(spec, "poisson_eps", 0.0)),
            center_sources=True,
            tol=float(getattr(spec, "tol", 1.0e-3)),
            max_iter=int(getattr(spec, "max_iter", 1000)),
        ),
        device=str(dev),
    )
    labels = torch.as_tensor(np.argmax(poisson_res.F, axis=1), dtype=torch.long, device=dev)
    U = _onehot_torch(labels, n_classes=n_classes)

    dt = 1.0 / max_deg
    Db = float(spec.mu) * dt * B_t
    weights = torch.ones((n_classes,), dtype=torch.float32, device=dev)
    residual = float(poisson_res.residual)
    volume_residual = float("inf")

    ns = int(getattr(spec, "Ninner", None) or getattr(spec, "Ns", 40))
    outer = int(getattr(spec, "Nouter", None) or getattr(spec, "T", 20))

    for _ in range(outer):
        for _ in range(ns):
            WU = spmm_torch(n_nodes=n_nodes, edge_index=edge_index_t, edge_weight=w_t, X=U)
            LU = deg.view(-1, 1) * U - WU
            update = -dt * LU + Db
            U = U + update
            residual = float(torch.max(torch.abs(update)).detach().cpu())

        labels, weights, volume_residual, _ = _volume_label_projection_torch(
            U,
            class_priors=b_t,
            weights=weights,
            tol=float(getattr(spec, "volume_tol", 1.0e-3)),
            max_iter=int(getattr(spec, "volume_max_iter", 10_000)),
            dt=float(getattr(spec, "volume_dt", 0.1)),
        )
        U = _onehot_torch(labels, n_classes=n_classes)

    total_iter = int(poisson_res.n_iter) + outer * ns
    return DiffusionResult(
        F=to_numpy(U), n_iter=total_iter, residual=max(residual, volume_residual)
    )


@dataclass(frozen=True)
class PoissonMBOSpec:
    """Poisson MBO (Calder et al.) - volume constrained MBO with Poisson fidelity."""

    Ns: int = 40
    T: int = 20
    mu: float = 1.0
    tol: float = 1.0e-3
    max_iter: int = 1000
    poisson_eps: float = 0.0
    volume_tol: float = 1.0e-3
    volume_max_iter: int = 10_000
    volume_dt: float = 0.1
    b_strategy: Literal["uniform", "labeled", "true"] = "true"
    b: np.ndarray | None = None
    symmetrize: bool = True
    zero_diagonal: bool = True
    # Backward-compatible aliases accepted by older YAMLs/tests.
    Ninner: int | None = None
    Nouter: int | None = None
    d_tau: float = 10.0
    smin: float = 0.5
    smax: float = 2.0
    n_volume_iters: int = 100


def poisson_mbo(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: PoissonMBOSpec | None = None,
    backend: Literal["numpy", "torch", "auto"] = "numpy",
    device: str | None = None,
) -> DiffusionResult:
    if backend not in ("numpy", "torch", "auto"):
        raise ValueError("backend must be one of: numpy, torch, auto")
    if backend == "torch":
        return poisson_mbo_torch(
            n_nodes=n_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
            device=device,
        )
    if backend == "auto":
        try:
            optional_import("torch", extra="transductive-torch")
        except Exception:
            return poisson_mbo_numpy(
                n_nodes=n_nodes,
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=y,
                labeled_mask=labeled_mask,
                spec=spec,
            )
        return poisson_mbo_torch(
            n_nodes=n_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
            device=device,
        )
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
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Poisson Learning: Graph Based Semi-Supervised Learning at Very Low Label Rates",
        paper_pdf="https://proceedings.mlr.press/v119/calder20a/calder20a.pdf",
        official_code="https://github.com/jwcalder/GraphLearning",
    )

    def __init__(self, spec: PoissonMBOSpec | None = None) -> None:
        self.spec = spec or PoissonMBOSpec()
        self._result: DiffusionResult | None = None
        self._backend: str | None = None

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
        backend = "torch" if device is not None else "numpy"
        self._backend = backend
        logger.debug("backend=%s", backend)
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
            backend=backend,
            device=device,
        )
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("PoissonMBOMethod is not fitted yet. Call fit() first.")
        return np.asarray(self._result.F, dtype=np.float32)
