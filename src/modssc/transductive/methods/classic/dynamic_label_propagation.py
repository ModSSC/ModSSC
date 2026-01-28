from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.device import resolve_device_name
from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.methods.utils import DiffusionResult, _validate_graph_inputs, to_numpy
from modssc.transductive.operators.clamp import labels_to_onehot
from modssc.transductive.validation import validate_node_dataset

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DynamicLabelPropagationSpec:
    """Dynamic Label Propagation (Wang & Tsotsos, 2016)."""

    k_neighbors: int = 10
    alpha: float = 0.05
    lambda_value: float = 0.1
    max_iter: int = 30


def _infer_num_classes(y: np.ndarray, labeled_mask: np.ndarray | None = None) -> int:
    y_valid = y[y >= 0]
    n_classes = int(y_valid.max()) + 1 if y_valid.size else 1
    return max(1, n_classes)


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


def _knn_matrix_numpy(P0: np.ndarray, k: int) -> np.ndarray:
    n = int(P0.shape[0])
    if k <= 0 or k >= n:
        return P0.copy()

    P = np.zeros_like(P0)
    for i in range(n):
        row = P0[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(row, -k)[-k:]
        vals = P0[i, idx]
        s = float(vals.sum())
        if s <= 0.0:
            raise ValueError("KNN matrix construction failed due to zero row sum.")
        P[i, idx] = vals / s
    return P


def dynamic_label_propagation_numpy(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: DynamicLabelPropagationSpec | None = None,
) -> DiffusionResult:
    if spec is None:
        spec = DynamicLabelPropagationSpec()
    if spec.k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")
    if not (float(spec.alpha) >= 0.0):
        raise ValueError("alpha must be non-negative")

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
        raise ValueError("DynamicLabelPropagation requires at least 1 labeled node.")

    n_classes = _infer_num_classes(y)
    Y0 = labels_to_onehot(y, n_classes=n_classes).astype(np.float32, copy=False)
    Y0[~labeled_mask] = 0.0

    W = _build_affinity_matrix(n_nodes=n_nodes, edge_index=edge_index, edge_weight=w)
    deg = W.sum(axis=1)
    if np.any(deg <= 0):
        raise ValueError("DynamicLabelPropagation requires a graph without isolated nodes.")

    P0 = W / deg[:, None]
    P = _knn_matrix_numpy(P0, int(spec.k_neighbors))

    P_t = P0.copy()
    Y_t = Y0.copy()
    residual = float("inf")
    identity = np.eye(n_nodes, dtype=np.float32)

    for _ in range(int(spec.max_iter)):
        Y_next = P_t @ Y_t
        Y_next[labeled_mask] = Y0[labeled_mask]

        P_t = P @ (P_t + float(spec.alpha) * (Y_t @ Y_t.T)) @ P.T
        if float(spec.lambda_value) != 0.0:
            P_t = P_t + float(spec.lambda_value) * identity

        residual = float(np.max(np.abs(Y_next - Y_t)))
        Y_t = Y_next

    return DiffusionResult(F=Y_t, n_iter=int(spec.max_iter), residual=residual)


def _knn_matrix_torch(P0: torch.Tensor, k: int) -> torch.Tensor:
    n = int(P0.shape[0])
    if k <= 0 or k >= n:
        return P0.clone()

    P = torch.zeros_like(P0)
    row = P0.clone()
    row.fill_diagonal_(-float("inf"))
    vals, idx = torch.topk(row, k=k, dim=1, largest=True, sorted=False)
    s = vals.sum(dim=1, keepdim=True)
    if bool((s <= 0).any().item()):
        raise ValueError("KNN matrix construction failed due to zero row sum.")
    norm_vals = vals / s
    row_idx = torch.arange(n, device=P0.device).view(-1, 1).expand(-1, k)
    P[row_idx, idx] = norm_vals
    return P


def dynamic_label_propagation_torch(
    *,
    n_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    y: torch.Tensor,
    labeled_mask: torch.Tensor,
    spec: DynamicLabelPropagationSpec | None = None,
) -> DiffusionResult:
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for dynamic_label_propagation_torch")
    if spec is None:
        spec = DynamicLabelPropagationSpec()
    if spec.k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")
    if not (float(spec.alpha) >= 0.0):
        raise ValueError("alpha must be non-negative")

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
        raise ValueError("DynamicLabelPropagation requires at least 1 labeled node.")

    n_classes = _infer_num_classes(to_numpy(y))
    Y0_np = labels_to_onehot(to_numpy(y), n_classes=n_classes).astype(np.float32)
    Y0_np[~to_numpy(labeled_mask).astype(bool)] = 0.0
    Y0 = torch.from_numpy(Y0_np).to(device=y.device)

    W = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=edge_index.device)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    W.index_put_((dst, src), w, accumulate=True)
    W = 0.5 * (W + W.T)
    W.fill_diagonal_(0.0)

    deg = W.sum(dim=1)
    if bool((deg <= 0).any().item()):
        raise ValueError("DynamicLabelPropagation requires a graph without isolated nodes.")

    P0 = W / deg.view(-1, 1)
    P = _knn_matrix_torch(P0, int(spec.k_neighbors))

    P_t = P0.clone()
    Y_t = Y0.clone()
    residual = float("inf")
    identity = torch.eye(n_nodes, dtype=torch.float32, device=W.device)

    for _ in range(int(spec.max_iter)):
        Y_next = P_t @ Y_t
        Y_next[labeled_mask] = Y0[labeled_mask]

        P_t = P @ (P_t + float(spec.alpha) * (Y_t @ Y_t.T)) @ P.T
        if float(spec.lambda_value) != 0.0:
            P_t = P_t + float(spec.lambda_value) * identity

        residual = float(torch.max(torch.abs(Y_next - Y_t)).item())
        Y_t = Y_next

    return DiffusionResult(F=to_numpy(Y_t), n_iter=int(spec.max_iter), residual=residual)


def dynamic_label_propagation(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: DynamicLabelPropagationSpec | None = None,
    backend: Literal["numpy", "torch", "auto"] = "auto",
    device: str | None = None,
) -> DiffusionResult:
    if spec is None:
        spec = DynamicLabelPropagationSpec()

    if backend not in ("numpy", "torch", "auto"):
        raise ValueError("backend must be one of: numpy, torch, auto")

    if backend == "numpy" or (backend == "auto" and (torch is None or device is None)):
        return dynamic_label_propagation_numpy(
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

    try:
        return dynamic_label_propagation_torch(
            n_nodes=n_nodes,
            edge_index=edge_index_t,
            edge_weight=edge_weight_t,
            y=y_t,
            labeled_mask=labeled_t,
            spec=spec,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"DynamicLabelPropagation ran out of memory on device={dev}. "
            "Retry with device='cpu' or reduce graph size."
        ) from exc


class DynamicLabelPropagationMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="dynamic_label_propagation",
        name="Dynamic Label Propagation",
        year=2016,
        family="propagation",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Dynamic Label Propagation for Semi-Supervised Multi-Class Multi-Label Classification",
    )

    def __init__(self, spec: DynamicLabelPropagationSpec | None = None) -> None:
        self.spec = spec or DynamicLabelPropagationSpec()
        self._result: DiffusionResult | None = None

    def fit(
        self, data: Any, *, device: str | None = None, seed: int = 0
    ) -> DynamicLabelPropagationMethod:
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
            "Dynamic label propagation sizes: n_nodes=%s labeled=%s",
            int(np.asarray(data.y).shape[0]),
            int(labeled_mask.sum()),
        )
        self._result = dynamic_label_propagation(
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
            raise RuntimeError("DynamicLabelPropagationMethod is not fitted yet. Call fit() first.")
        return np.asarray(self._result.F, dtype=np.float32)
