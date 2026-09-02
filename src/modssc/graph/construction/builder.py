from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..errors import GraphValidationError
from ..specs import GraphBuilderSpec
from .backends.annoy_backend import AnnoyParams, knn_edges_annoy
from .backends.faiss_backend import FaissParams, knn_edges_faiss
from .backends.numpy_backend import epsilon_edges_numpy, knn_edges_numpy
from .backends.precomputed_backend import knn_edges_precomputed
from .backends.sklearn_backend import epsilon_edges_sklearn, knn_edges_sklearn
from .backends.torch_backend import knn_edges_torch
from .schemes.anchor import AnchorParams, anchor_edges

logger = logging.getLogger(__name__)


def resolve_graph_backend(spec: GraphBuilderSpec) -> str:
    """Resolve backend selection for this environment."""
    if spec.backend != "auto":
        return str(spec.backend)

    # Prefer sklearn when available (fast exact neighbors).
    try:
        import sklearn  # noqa: F401
    except Exception:
        logger.debug("Graph backend auto -> numpy (sklearn not available)")
        return "numpy"
    return "sklearn"


# Compatibility for callers that imported the historical private helper.
_pick_backend = resolve_graph_backend


def build_raw_edges(
    X: np.ndarray,
    *,
    spec: GraphBuilderSpec,
    seed: int,
    work_dir: str | Path | None = None,
    resume: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return directed edges and distances before symmetrization / self-loops."""
    spec.validate()

    X = np.asarray(X)
    backend = resolve_graph_backend(spec)
    logger.debug("Graph backend selected: scheme=%s backend=%s", spec.scheme, backend)

    wd = Path(work_dir) if work_dir is not None else None
    if wd is not None:
        wd.mkdir(parents=True, exist_ok=True)

    if spec.scheme == "knn":
        if spec.k is None:
            raise GraphValidationError("k must be set for knn scheme")
        include_self = bool(spec.include_self_in_knn)
        if backend == "precomputed":
            if spec.precomputed_path is None or spec.precomputed_sha256 is None:
                raise GraphValidationError(
                    "precomputed backend requires precomputed_path and precomputed_sha256"
                )
            edge_index, dist = knn_edges_precomputed(
                X,
                k=int(spec.k),
                metric=spec.metric,
                include_self=include_self,
                path=spec.precomputed_path,
                expected_sha256=spec.precomputed_sha256,
            )
        elif backend == "faiss":
            fp = FaissParams(
                exact=bool(spec.faiss_exact),
                hnsw_m=int(spec.faiss_hnsw_m),
                ef_search=int(spec.faiss_ef_search),
                ef_construction=int(spec.faiss_ef_construction),
            )
            edge_index, dist = knn_edges_faiss(
                X,
                k=int(spec.k),
                metric=spec.metric,
                include_self=include_self,
                params=fp,
            )
        elif backend == "annoy":
            annoy_params = AnnoyParams(
                n_trees=int(spec.annoy_n_trees),
                search_k=int(spec.annoy_search_k),
                query_k=(None if spec.annoy_query_k is None else int(spec.annoy_query_k)),
                seed=int(seed),
                rerank=bool(spec.annoy_rerank),
            )
            edge_index, dist = knn_edges_annoy(
                X,
                k=int(spec.k),
                metric=spec.metric,  # type: ignore[arg-type]
                include_self=include_self,
                params=annoy_params,
            )
        elif backend == "sklearn":
            edge_index, dist = knn_edges_sklearn(
                X, k=int(spec.k), metric=spec.metric, include_self=include_self
            )
        elif backend == "numpy":
            edge_index, dist = knn_edges_numpy(
                X,
                k=int(spec.k),
                metric=spec.metric,
                include_self=include_self,
                chunk_size=int(spec.chunk_size),
                work_dir=(wd / "knn" if wd is not None else None),
                resume=bool(resume),
            )
        elif backend == "torch":
            edge_index, dist = knn_edges_torch(
                X,
                k=int(spec.k),
                metric=spec.metric,
                include_self=include_self,
                chunk_size=int(spec.chunk_size),
                device="auto",
            )
        else:
            raise GraphValidationError(f"Unknown backend: {backend!r}")

    elif spec.scheme == "epsilon":
        if spec.radius is None:
            raise GraphValidationError("radius must be set for epsilon scheme")
        if backend == "faiss":
            raise GraphValidationError("faiss backend does not support epsilon scheme")
        if backend == "sklearn":
            edge_index, dist = epsilon_edges_sklearn(
                X, radius=float(spec.radius), metric=spec.metric, include_self=False
            )
        elif backend == "numpy":
            edge_index, dist = epsilon_edges_numpy(
                X,
                radius=float(spec.radius),
                metric=spec.metric,
                include_self=False,
                chunk_size=int(spec.chunk_size),
                work_dir=(wd / "epsilon" if wd is not None else None),
                resume=bool(resume),
            )
        else:
            raise GraphValidationError(f"Unknown backend: {backend!r}")

    elif spec.scheme == "anchor":
        if spec.k is None:
            raise GraphValidationError("k must be set for anchor scheme")
        if backend not in ("numpy", "sklearn", "faiss"):
            raise GraphValidationError(f"Unknown backend: {backend!r}")

        ap = AnchorParams(
            n_anchors=spec.n_anchors,
            anchors_k=int(spec.anchors_k),
            method=spec.anchors_method,
            candidate_limit=int(spec.candidate_limit),
            chunk_size=int(spec.chunk_size),
        )
        fp = FaissParams(
            exact=bool(spec.faiss_exact),
            hnsw_m=int(spec.faiss_hnsw_m),
            ef_search=int(spec.faiss_ef_search),
            ef_construction=int(spec.faiss_ef_construction),
        )
        edge_index, dist = anchor_edges(
            X,
            k=int(spec.k),
            metric=spec.metric,
            backend=backend,  # type: ignore[arg-type]
            seed=int(seed),
            params=ap,
            faiss_params=fp if backend == "faiss" else None,
            include_self=False,
        )
    else:
        raise GraphValidationError(f"Unknown scheme: {spec.scheme!r}")

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise GraphValidationError("edge_index must have shape (2, E)")
    if dist.ndim != 1 or dist.shape[0] != edge_index.shape[1]:
        raise GraphValidationError("distances must have shape (E,)")

    distance_dtype = np.float64 if spec.edge_weight_dtype == "float64" else np.float32
    return edge_index.astype(np.int64), dist.astype(distance_dtype)
