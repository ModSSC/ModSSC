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
class LaplaceLearningSpec:
    """Laplace Learning (Gaussian fields and harmonic functions).

    This method has no hyperparameters in the original formulation; the graph
    structure fully determines the solution.
    """


def _infer_num_classes(y: np.ndarray, labeled_mask: np.ndarray | None = None) -> int:
    y_valid = y[y >= 0]
    n_classes = int(y_valid.max()) + 1 if y_valid.size else 1
    return max(1, n_classes)


def laplace_learning_numpy(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
) -> DiffusionResult:
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
        raise ValueError("LaplaceLearning requires at least 1 labeled node.")

    n_classes = _infer_num_classes(y)
    Y = labels_to_onehot(y, n_classes=n_classes).astype(np.float32, copy=False)
    Y[~labeled_mask] = 0.0

    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    src = edge_index[0]
    dst = edge_index[1]
    np.add.at(W, (dst, src), w)

    deg = W.sum(axis=1)
    L = -W
    np.fill_diagonal(L, deg)

    labeled_idx = np.flatnonzero(labeled_mask)
    unlabeled_idx = np.flatnonzero(~labeled_mask)
    if unlabeled_idx.size == 0:
        return DiffusionResult(F=Y, n_iter=1, residual=0.0)

    L_uu = L[np.ix_(unlabeled_idx, unlabeled_idx)]
    W_ul = W[np.ix_(unlabeled_idx, labeled_idx)]
    B = W_ul @ Y[labeled_idx]

    try:
        F_u = np.linalg.solve(L_uu, B)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "LaplaceLearning requires L_uu to be nonsingular; "
            "check graph connectivity and labeled coverage."
        ) from exc

    F = np.zeros((n_nodes, n_classes), dtype=np.float32)
    F[labeled_idx] = Y[labeled_idx]
    F[unlabeled_idx] = F_u
    return DiffusionResult(F=F, n_iter=1, residual=0.0)


def laplace_learning_torch(
    *,
    n_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    y: torch.Tensor,
    labeled_mask: torch.Tensor,
) -> DiffusionResult:
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for laplace_learning_torch")

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
        raise ValueError("LaplaceLearning requires at least 1 labeled node.")

    n_classes = _infer_num_classes(to_numpy(y))
    Y_np = labels_to_onehot(to_numpy(y), n_classes=n_classes).astype(np.float32)
    Y_np[~to_numpy(labeled_mask).astype(bool)] = 0.0
    Y = torch.from_numpy(Y_np).to(device=y.device)

    W = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=edge_index.device)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    W.index_put_((dst, src), w, accumulate=True)

    deg = W.sum(dim=1)
    L = torch.diag(deg) - W

    labeled_idx = torch.nonzero(labeled_mask, as_tuple=False).view(-1)
    unlabeled_idx = torch.nonzero(~labeled_mask, as_tuple=False).view(-1)
    if int(unlabeled_idx.numel()) == 0:
        return DiffusionResult(F=to_numpy(Y), n_iter=1, residual=0.0)

    L_uu = L.index_select(0, unlabeled_idx).index_select(1, unlabeled_idx)
    W_ul = W.index_select(0, unlabeled_idx).index_select(1, labeled_idx)
    B = W_ul @ Y.index_select(0, labeled_idx)

    try:
        F_u = torch.linalg.solve(L_uu, B)
    except RuntimeError as exc:  # pragma: no cover
        raise ValueError(
            "LaplaceLearning requires L_uu to be nonsingular; "
            "check graph connectivity and labeled coverage."
        ) from exc

    F = torch.zeros((n_nodes, n_classes), dtype=torch.float32, device=Y.device)
    F.index_copy_(0, labeled_idx, Y.index_select(0, labeled_idx))
    F.index_copy_(0, unlabeled_idx, F_u)
    return DiffusionResult(F=to_numpy(F), n_iter=1, residual=0.0)


def laplace_learning(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: LaplaceLearningSpec | None = None,
    backend: Literal["numpy", "torch", "auto"] = "auto",
    device: str | None = None,
) -> DiffusionResult:
    if spec is None:
        spec = LaplaceLearningSpec()

    if backend not in ("numpy", "torch", "auto"):
        raise ValueError("backend must be one of: numpy, torch, auto")

    if backend == "numpy" or (backend == "auto" and (torch is None or device is None)):
        return laplace_learning_numpy(
            n_nodes=n_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
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

    return laplace_learning_torch(
        n_nodes=n_nodes,
        edge_index=edge_index_t,
        edge_weight=edge_weight_t,
        y=y_t,
        labeled_mask=labeled_t,
    )


class LaplaceLearningMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="laplace_learning",
        name="Laplace Learning",
        year=2003,
        family="propagation",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Semi-supervised Learning Using Gaussian Fields and Harmonic Functions",
    )

    def __init__(self, spec: LaplaceLearningSpec | None = None) -> None:
        self.spec = spec or LaplaceLearningSpec()
        self._result: DiffusionResult | None = None

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> LaplaceLearningMethod:
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
            "Laplace learning sizes: n_nodes=%s labeled=%s",
            int(np.asarray(data.y).shape[0]),
            int(labeled_mask.sum()),
        )
        self._result = laplace_learning(
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
            raise RuntimeError("LaplaceLearningMethod is not fitted yet. Call fit() first.")
        return np.asarray(self._result.F, dtype=np.float32)
