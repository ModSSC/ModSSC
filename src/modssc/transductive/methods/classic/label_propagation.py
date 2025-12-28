from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.device import resolve_device_name
from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.methods.utils import (
    DiffusionResult,
    NormMode,
    _validate_graph_inputs,
    normalize_edge_weight_numpy,
    normalize_edge_weight_torch,
    spmm_numpy,
    spmm_torch,
    to_numpy,
)
from modssc.transductive.operators.clamp import labels_to_onehot
from modssc.transductive.validation import validate_node_dataset

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LabelPropagationSpec:
    """Configuration for Label Propagation (hard clamping).

    Notes
    -----
    This implements the common label propagation update:

        F_{t+1} = P F_t
        F_{t+1}[L] = Y[L]  (hard clamp)

    where P is a normalized affinity matrix (typically random-walk).
    """

    norm: NormMode = "rw"
    max_iter: int = 200
    tol: float = 1e-6
    normalize_rows: bool = True


def label_propagation_numpy(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: LabelPropagationSpec | None = None,
) -> DiffusionResult:
    """Run Label Propagation using NumPy.

    Parameters
    ----------
    n_nodes:
        Number of nodes.
    edge_index, edge_weight:
        Graph in COO form with A[dst, src] = w.
    y:
        Integer labels for all nodes (shape (n_nodes,)).
    labeled_mask:
        Boolean mask indicating which nodes are labeled.
    spec:
        Algorithm parameters.

    Returns
    -------
    DiffusionResult
    """
    if spec is None:
        spec = LabelPropagationSpec()
    edge_index, w = _validate_graph_inputs(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight
    )
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if y.shape != (n_nodes,):
        raise ValueError("y must have shape (n_nodes,)")
    labeled_mask = np.asarray(labeled_mask, dtype=bool).reshape(-1)
    if labeled_mask.shape != (n_nodes,):
        raise ValueError("labeled_mask must have shape (n_nodes,)")

    y_valid = y[y >= 0]
    if labeled_mask.any():
        y_labeled = y[labeled_mask]
        y_labeled = y_labeled[y_labeled >= 0]
        n_classes = (
            int(np.unique(y_labeled).size) if y_labeled.size else int(np.unique(y_valid).size)
        )
    else:
        n_classes = int(np.unique(y_valid).size)
    n_classes = max(1, n_classes)

    Y = labels_to_onehot(y, n_classes=n_classes).astype(np.float32, copy=False)
    Y[~labeled_mask] = 0.0

    w_norm = normalize_edge_weight_numpy(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=w, mode=spec.norm
    )

    F = Y.copy()
    residual = float("inf")
    for it in range(int(spec.max_iter)):
        F_new = spmm_numpy(n_nodes=n_nodes, edge_index=edge_index, edge_weight=w_norm, X=F)
        # hard clamp
        if labeled_mask.any():
            F_new[labeled_mask] = Y[labeled_mask]

        if spec.normalize_rows:
            row_sum = F_new.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum > 0, row_sum, 1.0)
            F_new = F_new / row_sum

        residual = float(np.max(np.abs(F_new - F)))
        F = F_new
        if residual <= float(spec.tol):
            return DiffusionResult(F=F, n_iter=it + 1, residual=residual)

    return DiffusionResult(F=F, n_iter=int(spec.max_iter), residual=residual)


def label_propagation_torch(
    *,
    n_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    y: torch.Tensor,
    labeled_mask: torch.Tensor,
    spec: LabelPropagationSpec | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Torch implementation (CPU/GPU depending on tensors device)."""
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for label_propagation_torch")

    if spec is None:
        spec = LabelPropagationSpec()

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

    # Determine n_classes from labeled nodes if possible
    if bool(labeled_mask.any().item()):
        y_labeled = y[labeled_mask]
        y_labeled = y_labeled[y_labeled >= 0]
        n_classes = int(torch.unique(y_labeled).numel()) if y_labeled.numel() else 0
    else:
        y_valid = y[y >= 0]
        n_classes = int(torch.unique(y_valid).numel()) if y_valid.numel() else 0
    n_classes = max(1, n_classes)

    # Build Y in torch then reuse the numpy helper to keep consistent behavior
    y_np = to_numpy(y)
    Y_np = labels_to_onehot(y_np, n_classes=n_classes).astype(np.float32)
    Y_np[~to_numpy(labeled_mask).astype(bool)] = 0.0
    Y = torch.from_numpy(Y_np).to(device=y.device)

    w_norm = normalize_edge_weight_torch(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=w, mode=spec.norm
    )

    F = Y.clone()
    residual = float("inf")
    for it in range(int(spec.max_iter)):
        F_new = spmm_torch(n_nodes=n_nodes, edge_index=edge_index, edge_weight=w_norm, X=F)
        if bool(labeled_mask.any().item()):
            F_new[labeled_mask] = Y[labeled_mask]

        if spec.normalize_rows:
            row_sum = F_new.sum(dim=1, keepdim=True)
            row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
            F_new = F_new / row_sum

        residual = float(torch.max(torch.abs(F_new - F)).item())
        F = F_new
        if residual <= float(spec.tol):
            return F, it + 1, residual

    return F, int(spec.max_iter), residual


def label_propagation(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: LabelPropagationSpec | None = None,
    backend: Literal["numpy", "torch", "auto"] = "auto",
    device: str | None = None,
) -> DiffusionResult:
    """Backend-dispatching label propagation.

    If backend is "torch" (or "auto" with torch available), computations can run on GPU
    by passing device="cuda" (or any torch device string).
    """
    if spec is None:
        spec = LabelPropagationSpec()

    if backend not in ("numpy", "torch", "auto"):
        raise ValueError("backend must be one of: numpy, torch, auto")

    if backend == "numpy":
        return label_propagation_numpy(
            n_nodes=n_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
        )

    use_torch = backend == "torch" or (backend == "auto" and torch is not None)
    if not use_torch:
        return label_propagation_numpy(
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

    F, n_iter, residual = label_propagation_torch(
        n_nodes=n_nodes,
        edge_index=edge_index_t,
        edge_weight=edge_weight_t,
        y=y_t,
        labeled_mask=labeled_t,
        spec=spec,
    )
    return DiffusionResult(F=to_numpy(F), n_iter=int(n_iter), residual=float(residual))


class LabelPropagationMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="label_propagation",
        name="Label Propagation",
        year=2002,
        family="propagation",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Learning from Labeled and Unlabeled Data with Label Propagation",
    )

    def __init__(self, spec: LabelPropagationSpec | None = None) -> None:
        self.spec = spec or LabelPropagationSpec()
        self._result: DiffusionResult | None = None

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> LabelPropagationMethod:
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
            "Label propagation sizes: n_nodes=%s labeled=%s",
            int(np.asarray(data.y).shape[0]),
            int(labeled_mask.sum()),
        )
        self._result = label_propagation(
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
            raise RuntimeError("LabelPropagationMethod is not fitted yet. Call fit() first.")
        return np.asarray(self._result.F, dtype=np.float32)
