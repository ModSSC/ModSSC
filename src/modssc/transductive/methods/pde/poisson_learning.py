from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import numpy as np

from modssc.runtime.device import resolve_device_name
from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.methods.utils import (
    DiffusionResult,
    _validate_graph_inputs,
    degrees_numpy,
    to_numpy,
)
from modssc.transductive.operators.clamp import labels_to_onehot
from modssc.transductive.operators.laplacian import laplacian_matvec_numpy, laplacian_matvec_torch
from modssc.transductive.optional import optional_import
from modssc.transductive.solvers.cg import cg_solve_numpy, cg_solve_torch
from modssc.transductive.types import DeviceSpec
from modssc.transductive.validation import validate_node_dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoissonLearningSpec:
    """Poisson Learning (Calder et al.) - graph-based SSL for very few labels.

    We solve, for each class k, a Poisson-like linear system on the graph:
        (L + eps I + (1/n) 11^T) f_k = b_k
    where:
      - L is a graph Laplacian ("unnormalized", "sym" or "rw")
      - the (1/n)11^T term enforces solvability (removes constant nullspace component)
      - eps is an optional ridge for disconnected graphs (multiple null eigenvectors)
      - b_k is a zero-sum source term defined on labeled nodes.

    This is implemented via Conjugate Gradient with a matrix-vector product.

    Parameters
    ----------
    backend:
        "numpy", "torch", or "auto". Torch backend supports CUDA and Apple MPS
        when a device is passed by the benchmark runner.
    laplacian_kind:
        "paper_normalized" uses the paper normalization pipeline:
        solve with the symmetric normalized Laplacian and transform the source/solution
        by D^-1/2. "unnormalized", "sym", and "rw" keep the direct ModSSC variants.
    eps:
        Optional ridge added to the system, useful when the graph is disconnected.
    center_sources:
        If True, each class source b_k is centered on labeled nodes so that sum(b_k)=0.
    tol, max_iter:
        CG stopping criteria.
    """

    backend: Literal["numpy", "torch", "auto"] = "numpy"
    laplacian_kind: Literal["paper_normalized", "unnormalized", "sym", "rw"] = "paper_normalized"
    eps: float = 0.0
    center_sources: bool = True
    tol: float = 1e-3
    max_iter: int = 1000
    solver: Literal["conjugate_gradient", "paper_iteration"] = field(
        default="conjugate_gradient", kw_only=True
    )
    balance_scores: bool = field(default=False, kw_only=True)
    class_priors: tuple[float, ...] | None = field(default=None, kw_only=True)
    min_iter: int = field(default=50, kw_only=True)
    require_convergence: bool = field(default=False, kw_only=True)


def _validate_solver(spec: PoissonLearningSpec) -> None:
    if spec.solver not in {"conjugate_gradient", "paper_iteration"}:
        raise ValueError(f"Unknown Poisson solver: {spec.solver!r}")


def _paper_iteration_numpy(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    source: np.ndarray,
    labeled_mask: np.ndarray,
    spec: PoissonLearningSpec,
) -> DiffusionResult:
    """Run Algorithm 1's degree-scaled Poisson iteration.

    This is the ``gradient_descent`` solver in the authors' GraphLearning
    implementation.  Self loops are removed before computing the degree, the
    Poisson source is scaled by ``D^-1``, and convergence is checked with the
    mixing chain used in the paper.  The implementation deliberately remains
    NumPy-only: the paper profile is a CPU profile and this keeps its arithmetic
    and stopping rule identical on every cluster.
    """

    min_iter = int(spec.min_iter)
    max_iter = int(spec.max_iter)
    if min_iter < 0:
        raise ValueError("min_iter must be non-negative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if min_iter > max_iter:
        raise ValueError("min_iter must be less than or equal to max_iter")

    src_all, dst_all = edge_index
    keep = src_all != dst_all
    src = src_all[keep]
    dst = dst_all[keep]
    weights = edge_weight[keep].astype(np.float64, copy=False)

    degree = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(degree, dst, weights)
    if np.any(degree <= 0.0):
        raise ValueError("paper_iteration requires a graph with strictly positive degrees")
    # Archived GraphLearningOld builds D from W + 1e-10 I after removing the
    # actual graph diagonal. The 1e-10 is therefore present in every inverse
    # degree used by Algorithm 1, but not in the stationary distribution.
    inverse_degree_denominator = degree + 1.0e-10

    rhs = np.asarray(source, dtype=np.float64) / inverse_degree_denominator[:, None]
    scores = np.zeros_like(rhs)

    # The auxiliary distribution detects when the random walk has mixed.  It
    # starts uniformly on the labeled vertices exactly as in Algorithm 1.
    mixing = labeled_mask.astype(np.float64)
    mixing /= float(mixing.sum())
    stationary = degree / float(degree.sum())
    stopping_tolerance = 1.0 / float(n_nodes)
    residual = float(np.max(np.abs(mixing - stationary)))

    iteration = 0
    while (iteration < min_iter or residual > stopping_tolerance) and iteration < max_iter:
        propagated = np.zeros_like(scores)
        np.add.at(propagated, dst, weights[:, None] * scores[src])
        scores = rhs + propagated / inverse_degree_denominator[:, None]

        # W^T D^-1 p, expressed with the repository's A[dst, src] edge
        # convention.  On the symmetric paper graph this is the invariant
        # distribution iteration stated by Calder et al.
        next_mixing = np.zeros_like(mixing)
        np.add.at(
            next_mixing,
            src,
            weights * (mixing[dst] / inverse_degree_denominator[dst]),
        )
        mixing = next_mixing
        residual = float(np.max(np.abs(mixing - stationary)))
        iteration += 1

    return DiffusionResult(
        F=scores,
        n_iter=iteration,
        residual=residual,
    )


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

    source_dtype = np.float64 if np.asarray(Y_labeled).dtype == np.float64 else np.float32
    mask_f = labeled_mask.astype(source_dtype)
    B = np.zeros((n, c), dtype=source_dtype)

    for k in range(c):
        yk = Y_labeled[:, k].astype(source_dtype, copy=False)
        if center_sources:
            pi = float(yk[labeled_mask].mean())
            bk = mask_f * (yk - pi)
        else:
            bk = mask_f * yk
            # Ensure zero-sum (required for Laplacian solvability)
            bk = bk - float(bk.mean())
        B[:, k] = bk

    return B


def _apply_paper_decision_rule(
    scores: np.ndarray,
    *,
    Y_labeled: np.ndarray,
    labeled_mask: np.ndarray,
    spec: PoissonLearningSpec,
) -> np.ndarray:
    """Apply Equation (2.4), ``diag(b / y_bar)``, to class scores."""

    if not spec.balance_scores:
        return scores
    observed = np.asarray(
        Y_labeled[labeled_mask].mean(axis=0),
        dtype=np.float64,
    )
    if np.any(observed <= 0.0):
        raise ValueError("paper decision rule requires at least one label from every class")
    n_classes = int(observed.size)
    if spec.class_priors is None:
        # GraphLearningOld's ``training_balance=True`` branch uses
        # ``diag(1 / c)`` when beta is absent. This is a global factor away
        # from a uniform probability prior, but retaining the archived scale
        # is required for exact score parity.
        target = np.ones(n_classes, dtype=np.float64)
    else:
        target = np.asarray(spec.class_priors, dtype=np.float64).reshape(-1)
        if target.shape != (n_classes,):
            raise ValueError("class_priors must contain one value per labeled class")
        if not np.all(np.isfinite(target)) or np.any(target <= 0.0):
            raise ValueError("class_priors must be finite and strictly positive")
        target = target / float(target.sum())
    return np.asarray(scores) * (target / observed)[None, :]


def poisson_learning_numpy(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    spec: PoissonLearningSpec,
) -> DiffusionResult:
    _validate_solver(spec)
    edge_index, edge_weight = _validate_graph_inputs(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        preserve_float64=spec.solver == "paper_iteration",
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

    score_dtype = np.float64 if spec.solver == "paper_iteration" else np.float32
    Y = labels_to_onehot(y, n_classes=n_classes).astype(score_dtype, copy=False)
    Y[~labeled_mask] = 0.0

    B = _build_sources(
        Y_labeled=Y, labeled_mask=labeled_mask, center_sources=bool(spec.center_sources)
    )

    if spec.solver == "paper_iteration":
        result = _paper_iteration_numpy(
            n_nodes=int(n_nodes),
            edge_index=edge_index,
            edge_weight=edge_weight,
            source=B,
            labeled_mask=labeled_mask,
            spec=spec,
        )
        return DiffusionResult(
            F=_apply_paper_decision_rule(
                result.F,
                Y_labeled=Y,
                labeled_mask=labeled_mask,
                spec=spec,
            ),
            n_iter=result.n_iter,
            residual=result.residual,
        )
    if spec.solver != "conjugate_gradient":
        raise ValueError(f"Unknown Poisson solver: {spec.solver!r}")

    laplacian_kind = str(spec.laplacian_kind)
    matvec_kind = "sym" if laplacian_kind == "paper_normalized" else laplacian_kind

    matvec_L = laplacian_matvec_numpy(
        n_nodes=int(n_nodes),
        edge_index=edge_index,
        edge_weight=edge_weight,
        kind=matvec_kind,
    )

    deg = degrees_numpy(n_nodes=int(n_nodes), edge_index=edge_index, edge_weight=edge_weight)
    inv_sqrt_deg = np.zeros(int(n_nodes), dtype=np.float32)
    deg_mask = deg > 0.0
    inv_sqrt_deg[deg_mask] = 1.0 / np.sqrt(deg[deg_mask])

    if laplacian_kind == "paper_normalized":
        B_solve = inv_sqrt_deg[:, None] * B
        null_vec = np.sqrt(np.maximum(deg, 0.0)).astype(np.float32, copy=False)
        denom = float(np.dot(null_vec, null_vec))
        if denom <= 0.0:
            null_vec = np.ones(int(n_nodes), dtype=np.float32)
            denom = float(n_nodes)
    else:
        B_solve = B
    if laplacian_kind == "unnormalized":
        null_vec = deg
        denom = float(null_vec.sum())
        if denom <= 0.0:
            null_vec = np.ones(int(n_nodes), dtype=np.float32)
            denom = float(n_nodes)
    elif laplacian_kind != "paper_normalized":
        null_vec = np.ones(int(n_nodes), dtype=np.float32)
        denom = float(n_nodes)
    eps = float(spec.eps)

    def matvec_A(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        out = matvec_L(x)
        if eps != 0.0:
            out = out + eps * x
        # Lift the constant nullspace while preserving the paper's weighted mean
        # constraint for the unnormalized graph Laplacian.
        out = out + float(np.dot(null_vec, x) / denom) * null_vec
        return out

    F = np.zeros((int(n_nodes), n_classes), dtype=np.float32)
    n_iter_max = 0
    residual_max = 0.0

    for k in range(n_classes):
        b = B_solve[:, k].astype(np.float32, copy=False)
        cg = cg_solve_numpy(matvec=matvec_A, b=b, tol=float(spec.tol), max_iter=int(spec.max_iter))
        x = cg.x.astype(np.float32, copy=False)
        if laplacian_kind == "paper_normalized":
            x = inv_sqrt_deg * x
        F[:, k] = x
        n_iter_max = max(n_iter_max, int(cg.n_iter))
        residual_max = max(residual_max, float(cg.residual_norm))

    F = _apply_paper_decision_rule(
        F,
        Y_labeled=Y,
        labeled_mask=labeled_mask,
        spec=spec,
    ).astype(np.float32, copy=False)
    return DiffusionResult(F=F, n_iter=n_iter_max, residual=residual_max)


def poisson_learning_torch(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any,
    y: Any,
    labeled_mask: Any,
    spec: PoissonLearningSpec,
    device: str | None = None,
) -> DiffusionResult:
    _validate_solver(spec)
    if spec.solver != "conjugate_gradient":
        raise ValueError("paper_iteration is a NumPy/CPU-only solver")
    torch = optional_import("torch", extra="transductive-torch")

    if device is None and hasattr(y, "device"):
        device = str(y.device)
    dev_name = resolve_device_name(device, torch=torch) or "cpu"
    device_t = torch.device(dev_name)

    y_t = torch.as_tensor(y, dtype=torch.long, device=device_t)

    edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=device_t)
    if edge_index_t.ndim != 2 or edge_index_t.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index_t.shape)}")

    edge_weight_t = (
        None
        if edge_weight is None
        else torch.as_tensor(edge_weight, dtype=torch.float32, device=device_t)
    )
    labeled_mask_t = torch.as_tensor(labeled_mask, dtype=torch.bool, device=device_t)

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

    B_t = torch.as_tensor(B_np, dtype=torch.float32, device=device_t)

    laplacian_kind = str(spec.laplacian_kind)
    matvec_kind = "sym" if laplacian_kind == "paper_normalized" else laplacian_kind

    matvec_L = laplacian_matvec_torch(
        n_nodes=n_nodes_i,
        edge_index=edge_index_t,
        edge_weight=edge_weight_t,
        device=DeviceSpec(device=str(device_t)),
        kind=matvec_kind,
    )

    edge_weight_for_deg = (
        torch.ones((edge_index_t.shape[1],), dtype=torch.float32, device=device_t)
        if edge_weight is None
        else torch.as_tensor(edge_weight, dtype=torch.float32, device=device_t).reshape(-1)
    )
    deg_t = torch.zeros((n_nodes_i,), dtype=torch.float32, device=device_t)
    if int(edge_index_t.shape[1]):
        deg_t.scatter_add_(0, edge_index_t[1], edge_weight_for_deg)
    inv_sqrt_deg = torch.zeros_like(deg_t)
    deg_mask = deg_t > 0.0
    inv_sqrt_deg[deg_mask] = torch.rsqrt(deg_t[deg_mask])

    if laplacian_kind == "paper_normalized":
        B_t = inv_sqrt_deg.view(-1, 1) * B_t
        null_vec = torch.sqrt(torch.clamp(deg_t, min=0.0))
        denom = (null_vec * null_vec).sum()
        if float(denom.detach().cpu()) <= 0.0:
            null_vec = torch.ones((n_nodes_i,), dtype=torch.float32, device=device_t)
            denom = torch.as_tensor(float(n_nodes_i), dtype=torch.float32, device=device_t)
    elif laplacian_kind == "unnormalized":
        null_vec = deg_t
        denom = null_vec.sum()
        if float(denom.detach().cpu()) <= 0.0:
            null_vec = torch.ones((n_nodes_i,), dtype=torch.float32, device=device_t)
            denom = torch.as_tensor(float(n_nodes_i), dtype=torch.float32, device=device_t)
    else:
        null_vec = torch.ones((n_nodes_i,), dtype=torch.float32, device=device_t)
        denom = torch.as_tensor(float(n_nodes_i), dtype=torch.float32, device=device_t)
    eps = float(spec.eps)

    def matvec_A(x: Any) -> Any:
        x = torch.as_tensor(x, dtype=torch.float32, device=device_t)
        out = matvec_L(x)
        if eps != 0.0:
            out = out + eps * x
        out = out + ((null_vec * x).sum() / denom) * null_vec
        return out

    F = torch.zeros((n_nodes_i, n_classes), dtype=torch.float32, device=device_t)
    n_iter_max = 0
    residual_max = 0.0

    for k in range(n_classes):
        b = B_t[:, k]
        x, info = cg_solve_torch(
            matvec=matvec_A,
            b=b,
            device=DeviceSpec(device=str(device_t)),
            tol=float(spec.tol),
            max_iter=int(spec.max_iter),
        )
        if laplacian_kind == "paper_normalized":
            x = inv_sqrt_deg * x
        F[:, k] = x
        n_iter_max = max(n_iter_max, int(info.get("n_iter", 0)))
        residual_max = max(residual_max, float(info.get("residual_norm", 0.0)))

    F_numpy = _apply_paper_decision_rule(
        to_numpy(F),
        Y_labeled=Y_np,
        labeled_mask=to_numpy(labeled_mask_t),
        spec=spec,
    ).astype(np.float32, copy=False)
    return DiffusionResult(F=F_numpy, n_iter=n_iter_max, residual=residual_max)


def poisson_learning(
    *,
    n_nodes: int,
    edge_index: Any,
    edge_weight: Any,
    y: Any,
    labeled_mask: Any,
    spec: PoissonLearningSpec | None = None,
    device: str | None = None,
) -> DiffusionResult:
    """Backend-dispatching wrapper."""
    if spec is None:
        spec = PoissonLearningSpec()
    _validate_solver(spec)

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
            device=device,
        )

    raise ValueError(f"Unknown backend: {spec.backend!r}")


class PoissonLearningMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="poisson_learning",
        name="Poisson Learning",
        year=2020,
        family="pde",
        supports_gpu=True,
        required_extra="transductive-torch",
        paper_title="Poisson Learning: Graph Based Semi-Supervised Learning at Very Low Label Rates",
        paper_pdf="https://proceedings.mlr.press/v119/calder20a/calder20a.pdf",
        official_code="https://github.com/jwcalder/GraphLearning",
    )

    def __init__(self, spec: PoissonLearningSpec | None = None) -> None:
        self.spec = spec or PoissonLearningSpec()
        self._result: DiffusionResult | None = None
        self.diagnostics_: dict[str, Any] = {}

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
        _validate_solver(self.spec)
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
            device=device,
        )
        self.diagnostics_ = {
            "solver": self.spec.solver,
            "decision_rule": (
                "paper_class_prior_correction" if self.spec.balance_scores else "raw_argmax"
            ),
            "iterations": int(self._result.n_iter),
            "mixing_residual": float(self._result.residual),
            "mixing_tolerance": 1.0 / float(np.asarray(data.y).shape[0]),
            "converged": bool(
                self.spec.solver != "paper_iteration"
                or self._result.residual <= 1.0 / float(np.asarray(data.y).shape[0])
            ),
        }
        logger.info("Finished %s.fit in %.3fs", self.info.method_id, perf_counter() - start)
        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("PoissonLearningMethod is not fitted yet. Call fit() first.")
        return np.asarray(self._result.F)
