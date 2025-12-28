from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.methods.utils import DiffusionResult, _validate_graph_inputs, to_numpy
from modssc.transductive.operators.clamp import labels_to_onehot
from modssc.transductive.operators.laplacian import laplacian_matvec_numpy, laplacian_matvec_torch
from modssc.transductive.optional import optional_import
from modssc.transductive.solvers.cg import cg_solve_numpy, cg_solve_torch
from modssc.transductive.types import DeviceSpec
from modssc.transductive.validation import validate_node_dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoissonLearningSpec:
    """Poisson Learning (Calder et al.) — graph-based SSL for very few labels.

    We solve, for each class k, a Poisson-like linear system on the graph:
        (L + eps I + (1/n) 11^T) f_k = b_k
    where:
      - L is a graph Laplacian ("sym" or "rw")
      - the (1/n)11^T term enforces solvability (removes constant nullspace component)
      - eps > 0 is a small ridge to handle disconnected graphs (multiple null eigenvectors)
      - b_k is a zero-sum source term defined on labeled nodes.

    This is implemented via Conjugate Gradient with a matrix-vector product.

    Parameters
    ----------
    backend:
        "numpy" or "torch". Torch backend supports GPU if tensors are placed on CUDA.
    laplacian_kind:
        "sym" (symmetric normalized Laplacian) or "rw" (random-walk Laplacian).
    eps:
        Small ridge added to the system to guarantee SPD (especially important if the graph is disconnected).
    center_sources:
        If True, each class source b_k is centered on labeled nodes so that sum(b_k)=0.
    tol, max_iter:
        CG stopping criteria.
    """

    backend: Literal["numpy", "torch", "auto"] = "numpy"
    laplacian_kind: Literal["sym", "rw"] = "sym"
    eps: float = 1e-6
    center_sources: bool = True
    tol: float = 1e-6
    max_iter: int = 2000


def _build_sources(
    *,
    Y_labeled: np.ndarray,
    labeled_mask: np.ndarray,
    center_sources: bool,
) -> np.ndarray:
    """Build per-class source terms b_k of shape (n, n_classes)."""
    n, c = Y_labeled.shape
    m = int(labeled_mask.sum())
    if m <= 0:
        raise ValueError("PoissonLearning requires at least 1 labeled node.")

    mask_f = labeled_mask.astype(np.float32)
    B = np.zeros((n, c), dtype=np.float32)

    for k in range(c):
        yk = Y_labeled[:, k].astype(np.float32, copy=False)
        if center_sources:
            pi = float(yk[labeled_mask].mean())
            bk = mask_f * (yk - pi)
        else:
            bk = mask_f * yk
            # Ensure zero-sum (required for Laplacian solvability)
            bk = bk - float(bk.mean())
        B[:, k] = bk

    return B


def poisson_learning_numpy(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: PoissonLearningSpec,
) -> DiffusionResult:
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
        raise ValueError("PoissonLearning requires at least 1 labeled node.")

    # Number of classes inferred from labeled nodes only
    classes = np.unique(y[labeled_idx])
    n_classes = int(classes.size)

    Y = labels_to_onehot(y, n_classes=n_classes).astype(np.float32, copy=False)
    Y[~labeled_mask] = 0.0

    B = _build_sources(
        Y_labeled=Y, labeled_mask=labeled_mask, center_sources=bool(spec.center_sources)
    )

    matvec_L = laplacian_matvec_numpy(
        n_nodes=int(n_nodes),
        edge_index=edge_index,
        edge_weight=edge_weight,
        kind=str(spec.laplacian_kind),
    )

    ones = np.ones(int(n_nodes), dtype=np.float32)
    eps = float(spec.eps)

    def matvec_A(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        out = matvec_L(x)
        if eps != 0.0:
            out = out + eps * x
        # (1/n) 11^T x == mean(x) * 1
        out = out + float(x.mean()) * ones
        return out

    F = np.zeros((int(n_nodes), n_classes), dtype=np.float32)
    n_iter_max = 0
    residual_max = 0.0

    for k in range(n_classes):
        b = B[:, k].astype(np.float32, copy=False)
        cg = cg_solve_numpy(matvec=matvec_A, b=b, tol=float(spec.tol), max_iter=int(spec.max_iter))
        F[:, k] = cg.x.astype(np.float32, copy=False)
        n_iter_max = max(n_iter_max, int(cg.n_iter))
        residual_max = max(residual_max, float(cg.residual_norm))

    return DiffusionResult(F=F, n_iter=n_iter_max, residual=residual_max)


def poisson_learning_torch(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any,
    y: Any,
    labeled_mask: Any,
    spec: PoissonLearningSpec,
) -> DiffusionResult:
    torch = optional_import("torch", extra="transductive-torch")

    # Convert to torch tensors (on the same device as y if possible)
    y_t = torch.as_tensor(y, dtype=torch.long)
    device = y_t.device

    edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    if edge_index_t.ndim != 2 or edge_index_t.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index_t.shape)}")

    edge_weight_t = torch.as_tensor(edge_weight, dtype=torch.float32, device=device)
    labeled_mask_t = torch.as_tensor(labeled_mask, dtype=torch.bool, device=device)

    n_nodes_i = int(n_nodes)
    if int(y_t.numel()) != n_nodes_i:
        raise ValueError(f"y must have length n_nodes={n_nodes_i}, got {int(y_t.numel())}")

    labeled_idx = torch.nonzero(labeled_mask_t, as_tuple=False).view(-1)
    if int(labeled_idx.numel()) == 0:
        raise ValueError("PoissonLearning requires at least 1 labeled node.")

    classes = torch.unique(y_t[labeled_idx]).detach().cpu().numpy()
    n_classes = int(classes.size)

    # Build Y one-hot on CPU via numpy helper then move to torch (keeps consistent encoding)
    Y_np = labels_to_onehot(to_numpy(y_t), n_classes=n_classes).astype(np.float32, copy=False)
    Y_np[~to_numpy(labeled_mask_t)] = 0.0
    B_np = _build_sources(
        Y_labeled=Y_np,
        labeled_mask=to_numpy(labeled_mask_t),
        center_sources=bool(spec.center_sources),
    )

    B_t = torch.as_tensor(B_np, dtype=torch.float32, device=device)

    matvec_L = laplacian_matvec_torch(
        n_nodes=n_nodes_i,
        edge_index=edge_index_t,
        edge_weight=edge_weight_t,
        device=DeviceSpec(device=str(device)),
        kind=str(spec.laplacian_kind),
    )

    ones = torch.ones((n_nodes_i,), dtype=torch.float32, device=device)
    eps = float(spec.eps)

    def matvec_A(x: Any) -> Any:
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        out = matvec_L(x)
        if eps != 0.0:
            out = out + eps * x
        out = out + x.mean() * ones
        return out

    F = torch.zeros((n_nodes_i, n_classes), dtype=torch.float32, device=device)
    n_iter_max = 0
    residual_max = 0.0

    for k in range(n_classes):
        b = B_t[:, k]
        x, info = cg_solve_torch(
            matvec=matvec_A,
            b=b,
            device=DeviceSpec(device=str(device)),
            tol=float(spec.tol),
            max_iter=int(spec.max_iter),
        )
        F[:, k] = x
        n_iter_max = max(n_iter_max, int(info.get("n_iter", 0)))
        residual_max = max(residual_max, float(info.get("residual_norm", 0.0)))

    return DiffusionResult(F=to_numpy(F), n_iter=n_iter_max, residual=residual_max)


def poisson_learning(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any,
    y: Any,
    labeled_mask: Any,
    spec: PoissonLearningSpec | None = None,
) -> DiffusionResult:
    """Backend-dispatching wrapper."""
    if spec is None:
        spec = PoissonLearningSpec()

    backend = str(spec.backend)
    if backend == "auto":
        try:
            optional_import("torch", extra="transductive-torch")
            backend = "torch"
        except Exception:
            backend = "numpy"

    if backend == "numpy":
        return poisson_learning_numpy(
            n_nodes=int(n_nodes),
            edge_index=np.asarray(edge_index),
            edge_weight=None if edge_weight is None else np.asarray(edge_weight),
            y=np.asarray(y),
            labeled_mask=np.asarray(labeled_mask),
            spec=spec,
        )

    if backend == "torch":
        return poisson_learning_torch(
            n_nodes=int(n_nodes),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            labeled_mask=labeled_mask,
            spec=spec,
        )

    raise ValueError(f"Unknown backend: {spec.backend!r}")


class PoissonLearningMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="poisson_learning",
        name="Poisson Learning",
        year=2017,
        family="pde",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Poisson Learning: Graph Based Semi-Supervised Learning at Very Low Label Rates",
    )

    def __init__(self, spec: PoissonLearningSpec | None = None) -> None:
        self.spec = spec or PoissonLearningSpec()
        self._result: DiffusionResult | None = None

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> PoissonLearningMethod:
        start = perf_counter()
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug(
            "spec=%s device=%s seed=%s backend=%s",
            self.spec,
            device,
            seed,
            self.spec.backend,
        )
        validate_node_dataset(data)

        masks = getattr(data, "masks", None) or {}
        if "train_mask" not in masks:
            raise ValueError("data.masks must contain 'train_mask'")

        labeled_mask = np.asarray(masks["train_mask"], dtype=bool)
        g = data.graph
        logger.info(
            "Poisson learning sizes: n_nodes=%s labeled=%s",
            int(np.asarray(data.y).shape[0]),
            int(labeled_mask.sum()),
        )

        self._result = poisson_learning(
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
            raise RuntimeError("PoissonLearningMethod is not fitted yet. Call fit() first.")
        return np.asarray(self._result.F, dtype=np.float32)
