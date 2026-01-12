from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.device import resolve_device_name
from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.methods.utils import DiffusionResult, _validate_graph_inputs, to_numpy
from modssc.transductive.validation import validate_node_dataset

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LazyRandomWalkSpec:
    """Lazy Random Walks (Zhou & Schoelkopf, 2004)."""

    alpha: float = 0.99


def _encode_binary(
    y: np.ndarray,
    *,
    labeled_mask: np.ndarray,
    full_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1)
    labeled_y = y[np.asarray(labeled_mask, dtype=bool)]
    labeled_y = labeled_y[labeled_y >= 0]
    classes = np.unique(labeled_y)
    if classes.size != 2:
        pool = y if full_y is None else np.asarray(full_y).reshape(-1)
        pool = pool[pool >= 0]
        classes = np.unique(pool)
    if classes.size != 2:
        raise ValueError(
            f"LazyRandomWalk supports binary classification only (got {classes.size} classes)."
        )
    y_enc = np.zeros_like(y, dtype=np.float32)
    y_enc[y == classes[0]] = -1.0
    y_enc[y == classes[1]] = 1.0
    return y_enc, classes


def _build_affinity_matrix(
    *, n_nodes: int, edge_index: np.ndarray, edge_weight: np.ndarray
) -> np.ndarray:
    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    src = edge_index[0]
    dst = edge_index[1]
    np.add.at(W, (dst, src), edge_weight)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def lazy_random_walk_numpy(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: LazyRandomWalkSpec | None = None,
) -> DiffusionResult:
    if spec is None:
        spec = LazyRandomWalkSpec()
    if not (0.0 < float(spec.alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    edge_index, w = _validate_graph_inputs(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight
    )
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if y.shape != (n_nodes,):
        raise ValueError("y must have shape (n_nodes,)")
    labeled_mask = np.asarray(labeled_mask, dtype=bool).reshape(-1)
    if labeled_mask.shape != (n_nodes,):
        raise ValueError("labeled_mask must have shape (n_nodes,)")
    if not labeled_mask.any():
        raise ValueError("LazyRandomWalk requires at least 1 labeled node.")

    y_pm1, classes = _encode_binary(y, labeled_mask=labeled_mask)
    y_vec = np.zeros((n_nodes,), dtype=np.float32)
    y_vec[labeled_mask] = y_pm1[labeled_mask]

    W = _build_affinity_matrix(n_nodes=n_nodes, edge_index=edge_index, edge_weight=w)
    deg = W.sum(axis=1)
    if np.any(deg <= 0):
        raise ValueError("LazyRandomWalk requires a graph without isolated nodes.")

    d_inv_sqrt = 1.0 / np.sqrt(deg)
    S = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
    A = np.eye(n_nodes, dtype=np.float32) - float(spec.alpha) * S

    try:
        f = np.linalg.solve(A, y_vec)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Failed to solve (I - alpha S) f = y.") from exc

    pred_pos = f >= 0.0
    F = np.zeros((n_nodes, 2), dtype=np.float32)
    F[~pred_pos, 0] = 1.0
    F[pred_pos, 1] = 1.0

    return DiffusionResult(F=F, n_iter=1, residual=0.0)


def lazy_random_walk_torch(
    *,
    n_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    y: torch.Tensor,
    labeled_mask: torch.Tensor,
    spec: LazyRandomWalkSpec | None = None,
) -> DiffusionResult:
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for lazy_random_walk_torch")
    if spec is None:
        spec = LazyRandomWalkSpec()
    if not (0.0 < float(spec.alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_weight is None:
        w = torch.ones((int(edge_index.shape[1]),), dtype=torch.float32, device=edge_index.device)
    else:
        w = edge_weight.to(dtype=torch.float32)
        if w.ndim != 1 or int(w.shape[0]) != int(edge_index.shape[1]):
            raise ValueError("edge_weight must have shape (E,)")

    y = y.to(dtype=torch.long).view(-1)
    labeled_mask = labeled_mask.to(dtype=torch.bool).view(-1)
    if int(y.shape[0]) != int(n_nodes) or int(labeled_mask.shape[0]) != int(n_nodes):
        raise ValueError("y and labeled_mask must have shape (n_nodes,)")
    if not bool(labeled_mask.any().item()):
        raise ValueError("LazyRandomWalk requires at least 1 labeled node.")

    y_pm1, classes = _encode_binary(to_numpy(y), labeled_mask=to_numpy(labeled_mask).astype(bool))
    y_vec = torch.zeros((n_nodes,), dtype=torch.float32, device=y.device)
    y_vec[labeled_mask] = torch.as_tensor(y_pm1, dtype=torch.float32, device=y.device)[labeled_mask]

    W = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=edge_index.device)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    W.index_put_((dst, src), w, accumulate=True)
    W = 0.5 * (W + W.T)
    W.fill_diagonal_(0.0)

    deg = W.sum(dim=1)
    if bool((deg <= 0).any().item()):
        raise ValueError("LazyRandomWalk requires a graph without isolated nodes.")

    d_inv_sqrt = torch.rsqrt(deg)
    S = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
    A = torch.eye(n_nodes, dtype=torch.float32, device=W.device) - float(spec.alpha) * S

    try:
        f = torch.linalg.solve(A, y_vec.unsqueeze(1)).squeeze(1)
    except RuntimeError as exc:  # pragma: no cover
        raise ValueError("Failed to solve (I - alpha S) f = y.") from exc

    pred_pos = f >= 0.0
    F = torch.zeros((n_nodes, 2), dtype=torch.float32, device=f.device)
    F[~pred_pos, 0] = 1.0
    F[pred_pos, 1] = 1.0

    return DiffusionResult(F=to_numpy(F), n_iter=1, residual=0.0)


def lazy_random_walk(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: LazyRandomWalkSpec | None = None,
    backend: Literal["numpy", "torch", "auto"] = "auto",
    device: str | None = None,
) -> DiffusionResult:
    if spec is None:
        spec = LazyRandomWalkSpec()

    if backend not in ("numpy", "torch", "auto"):
        raise ValueError("backend must be one of: numpy, torch, auto")

    if backend == "numpy" or (backend == "auto" and (torch is None or device is None)):
        return lazy_random_walk_numpy(
            n_nodes=n_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
        )

    if torch is None:  # pragma: no cover
        raise ImportError("torch is not available")

    dev_name = resolve_device_name(device, torch=torch) or "cpu"
    dev = torch.device(dev_name)
    edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=dev)
    edge_weight_t = (
        None
        if edge_weight is None
        else torch.as_tensor(edge_weight, dtype=torch.float32, device=dev)
    )
    y_t = torch.as_tensor(y, dtype=torch.long, device=dev)
    labeled_t = torch.as_tensor(labeled_mask, dtype=torch.bool, device=dev)

    return lazy_random_walk_torch(
        n_nodes=n_nodes,
        edge_index=edge_index_t,
        edge_weight=edge_weight_t,
        y=y_t,
        labeled_mask=labeled_t,
        spec=spec,
    )


class LazyRandomWalkMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="lazy_random_walk",
        name="Lazy Random Walk",
        year=2004,
        family="propagation",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Learning from labeled and unlabeled data using random walks",
    )

    def __init__(self, spec: LazyRandomWalkSpec | None = None) -> None:
        self.spec = spec or LazyRandomWalkSpec()
        self._result: DiffusionResult | None = None

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> LazyRandomWalkMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        validate_node_dataset(data)

        masks = getattr(data, "masks", None) or {}
        if "train_mask" not in masks:
            raise ValueError("data.masks must contain 'train_mask'")

        labeled_mask = np.asarray(masks["train_mask"], dtype=bool)
        g = data.graph

        backend = "torch" if device is not None else "auto"
        logger.debug("backend=%s", backend)
        logger.info(
            "Lazy random walk sizes: n_nodes=%s labeled=%s",
            int(np.asarray(data.y).shape[0]),
            int(labeled_mask.sum()),
        )
        self._result = lazy_random_walk(
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
            raise RuntimeError("LazyRandomWalkMethod is not fitted yet. Call fit() first.")
        return np.asarray(self._result.F, dtype=np.float32)
