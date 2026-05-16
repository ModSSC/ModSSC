from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as sparse_linalg

from modssc.runtime.device import resolve_device_name
from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.methods.classic.common import infer_num_classes as _infer_num_classes
from modssc.transductive.methods.utils import (
    DiffusionResult,
    _validate_graph_inputs,
    degrees_torch,
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


def _cg_solve_torch_multi_rhs(
    *,
    matvec: Any,
    b: Any,
    tol: float,
    max_iter: int,
) -> tuple[Any, dict[str, Any]]:
    """Conjugate gradient for multiple independent right-hand sides.

    The Laplace system matrix is shared across classes. Solving all class
    columns together avoids recomputing the sparse matvec once per class.
    """

    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for _cg_solve_torch_multi_rhs")
    if int(b.dim()) != 2:
        raise ValueError("b must have shape (n_unknowns, n_rhs)")

    x = torch.zeros_like(b)
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.sum(r * r, dim=0)
    residual = torch.sqrt(rs_old)
    active = residual > float(tol)
    n_iter = 0

    for k in range(int(max_iter)):
        if not bool(torch.any(active).detach().cpu()):
            break
        Ap = matvec(p)
        denom = torch.sum(p * Ap, dim=0)
        valid = active & (torch.abs(denom) > 1.0e-30)
        if not bool(torch.any(valid).detach().cpu()):
            break

        safe_denom = torch.where(valid, denom, torch.ones_like(denom))
        alpha = torch.where(valid, rs_old / safe_denom, torch.zeros_like(rs_old))
        x = x + p * alpha.view(1, -1)
        r = r - Ap * alpha.view(1, -1)

        rs_new = torch.sum(r * r, dim=0)
        residual = torch.sqrt(rs_new)
        active = residual > float(tol)
        beta = torch.where(
            valid, rs_new / torch.clamp(rs_old, min=1.0e-30), torch.zeros_like(rs_old)
        )
        p_next = r + p * beta.view(1, -1)
        p = torch.where(active.view(1, -1), p_next, torch.zeros_like(p_next))
        rs_old = rs_new
        n_iter = k + 1

    info = {
        "n_iter": n_iter,
        "converged": bool(torch.all(residual <= float(tol)).detach().cpu()),
        "residual_norm": float(torch.max(residual).detach().cpu())
        if int(residual.numel())
        else 0.0,
    }
    return x, info


@dataclass(frozen=True)
class LaplaceLearningSpec:
    """Laplace Learning (Gaussian fields and harmonic functions).

    This method has no hyperparameters in the original formulation; the graph
    structure fully determines the solution.
    """

    cg_tol: float = 1e-5
    cg_max_iter: int = 2000


def laplace_learning_numpy(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: LaplaceLearningSpec | None = None,
) -> DiffusionResult:
    if spec is None:
        spec = LaplaceLearningSpec()

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

    src = edge_index[0]
    dst = edge_index[1]
    W = sparse.coo_matrix(
        (w.astype(np.float32, copy=False), (dst, src)),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    ).tocsr()
    W.sum_duplicates()

    deg = np.asarray(W.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    L = sparse.diags(deg, offsets=0, format="csr", dtype=np.float32) - W

    labeled_idx = np.flatnonzero(labeled_mask)
    unlabeled_idx = np.flatnonzero(~labeled_mask)
    if unlabeled_idx.size == 0:
        return DiffusionResult(F=Y, n_iter=1, residual=0.0)
    _, components = csgraph.connected_components(W, directed=False, return_labels=True)
    labeled_components = set(int(c) for c in components[labeled_idx])
    if any(int(c) not in labeled_components for c in components[unlabeled_idx]):
        raise ValueError(
            "LaplaceLearning requires L_uu to be nonsingular; "
            "check graph connectivity and labeled coverage."
        )

    L_uu = L[unlabeled_idx][:, unlabeled_idx].tocsc()
    W_ul = W[unlabeled_idx][:, labeled_idx]
    B = W_ul @ Y[labeled_idx]

    F_u = np.zeros((unlabeled_idx.size, n_classes), dtype=np.float32)
    residual = 0.0
    n_iter = 0
    for c in range(n_classes):
        b = np.asarray(B[:, c]).reshape(-1).astype(np.float32, copy=False)
        if not np.any(b):
            continue
        x, info = sparse_linalg.cg(
            L_uu,
            b,
            rtol=float(spec.cg_tol),
            atol=0.0,
            maxiter=int(spec.cg_max_iter),
        )
        if info < 0:
            raise ValueError(
                "LaplaceLearning sparse CG failed; check graph connectivity and weights."
            )
        if info > 0:
            logger.warning(
                "LaplaceLearning CG reached max iterations: class=%s max_iter=%s",
                c,
                info,
            )
        x = np.asarray(x, dtype=np.float32)
        F_u[:, c] = x
        r = L_uu @ x - b
        denom = max(float(np.linalg.norm(b)), 1e-12)
        residual = max(residual, float(np.linalg.norm(r) / denom))
        n_iter = max(n_iter, int(info) if info > 0 else 0)
    if not np.all(np.isfinite(F_u)):
        raise ValueError(
            "LaplaceLearning requires L_uu to be nonsingular; "
            "check graph connectivity and labeled coverage."
        )

    F = np.zeros((n_nodes, n_classes), dtype=np.float32)
    F[labeled_idx] = Y[labeled_idx]
    F[unlabeled_idx] = F_u
    return DiffusionResult(F=F, n_iter=n_iter, residual=residual)


def laplace_learning_torch(
    *,
    n_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    y: torch.Tensor,
    labeled_mask: torch.Tensor,
    spec: LaplaceLearningSpec | None = None,
) -> DiffusionResult:
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for laplace_learning_torch")
    if spec is None:
        spec = LaplaceLearningSpec()

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

    labeled_idx = torch.nonzero(labeled_mask, as_tuple=False).view(-1)
    unlabeled_idx = torch.nonzero(~labeled_mask, as_tuple=False).view(-1)
    if int(unlabeled_idx.numel()) == 0:
        return DiffusionResult(F=to_numpy(Y), n_iter=1, residual=0.0)

    edge_index_np = to_numpy(edge_index).astype(np.int64, copy=False)
    w_np = to_numpy(w).astype(np.float32, copy=False)
    W_cpu = sparse.coo_matrix(
        (w_np, (edge_index_np[1], edge_index_np[0])),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    ).tocsr()
    W_cpu.sum_duplicates()
    _, components = csgraph.connected_components(W_cpu, directed=False, return_labels=True)
    labeled_idx_np = to_numpy(labeled_idx).astype(np.int64, copy=False)
    unlabeled_idx_np = to_numpy(unlabeled_idx).astype(np.int64, copy=False)
    labeled_components = set(int(c) for c in components[labeled_idx_np])
    if any(int(c) not in labeled_components for c in components[unlabeled_idx_np]):
        raise ValueError(
            "LaplaceLearning requires L_uu to be nonsingular; "
            "check graph connectivity and labeled coverage."
        )

    deg = degrees_torch(n_nodes=n_nodes, edge_index=edge_index, edge_weight=w)
    WY = spmm_torch(n_nodes=n_nodes, edge_index=edge_index, edge_weight=w, X=Y)
    B = WY.index_select(0, unlabeled_idx)

    def matvec_luu(x: Any) -> Any:
        x = torch.as_tensor(x, dtype=torch.float32, device=Y.device)
        squeeze = int(x.dim()) == 1
        if squeeze:
            x = x.view(-1, 1)
        x_all = torch.zeros((n_nodes, int(x.shape[1])), dtype=torch.float32, device=Y.device)
        x_all.index_copy_(0, unlabeled_idx, x)
        wx = spmm_torch(
            n_nodes=n_nodes,
            edge_index=edge_index,
            edge_weight=w,
            X=x_all,
        )
        lx = deg.view(-1, 1) * x_all - wx
        out = lx.index_select(0, unlabeled_idx)
        return out.view(-1) if squeeze else out

    F_u = torch.zeros((int(unlabeled_idx.numel()), n_classes), dtype=torch.float32, device=Y.device)
    residual = 0.0
    n_iter = 0
    active_idx = torch.nonzero(torch.any(B != 0.0, dim=0), as_tuple=False).view(-1)
    if int(active_idx.numel()):
        B_active = B.index_select(1, active_idx)
        X_active, info = _cg_solve_torch_multi_rhs(
            matvec=matvec_luu,
            b=B_active,
            tol=float(spec.cg_tol),
            max_iter=int(spec.cg_max_iter),
        )
        if not bool(info.get("converged", False)):
            logger.warning(
                "LaplaceLearning torch CG reached max iterations: max_iter=%s residual=%s",
                int(spec.cg_max_iter),
                float(info.get("residual_norm", float("nan"))),
            )
        F_u.index_copy_(1, active_idx, X_active)
        R = matvec_luu(X_active) - B_active
        denom = torch.clamp(torch.linalg.norm(B_active, dim=0), min=1.0e-12)
        residual_by_rhs = torch.linalg.norm(R, dim=0) / denom
        residual = float(torch.max(residual_by_rhs).detach().cpu())
        n_iter = int(info.get("n_iter", 0))
    if not bool(torch.all(torch.isfinite(F_u)).detach().cpu()):
        raise ValueError(
            "LaplaceLearning requires L_uu to be nonsingular; "
            "check graph connectivity and labeled coverage."
        )

    F = torch.zeros((n_nodes, n_classes), dtype=torch.float32, device=Y.device)
    F.index_copy_(0, labeled_idx, Y.index_select(0, labeled_idx))
    F.index_copy_(0, unlabeled_idx, F_u)
    return DiffusionResult(F=to_numpy(F), n_iter=n_iter, residual=residual)


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

    return laplace_learning_torch(
        n_nodes=n_nodes,
        edge_index=edge_index_t,
        edge_weight=edge_weight_t,
        y=y_t,
        labeled_mask=labeled_t,
        spec=spec,
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
        self._backend: str | None = None

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

        backend = "torch" if device is not None else "numpy"
        self._backend = backend
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
