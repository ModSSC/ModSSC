from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..artifacts import GraphArtifact
from ..cache import GraphCache
from ..errors import GraphValidationError
from ..fingerprint import fingerprint_array, fingerprint_dict
from ..specs import GraphBuilderSpec
from ..validation import validate_builder_spec, validate_features
from .builder import build_raw_edges
from .ops.normalize import normalize_edge_weights
from .ops.self_loops import add_self_loops
from .ops.symmetrize import symmetrize_edges
from .ops.weights import compute_edge_weights

logger = logging.getLogger(__name__)


def _degree_summary(edge_index: np.ndarray, n_nodes: int) -> tuple[int, float, int, int]:
    src = edge_index[0]
    deg = np.bincount(src, minlength=int(n_nodes))
    return int(deg.min()), float(deg.mean()), int(deg.max()), int((deg == 0).sum())


def _graph_fingerprint(
    *,
    dataset_fingerprint: str,
    preprocess_fingerprint: str | None,
    spec: GraphBuilderSpec,
    seed: int,
) -> str:
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "preprocess_fingerprint": preprocess_fingerprint,
        "spec": spec.to_dict(),
        "seed": int(seed),
    }
    return fingerprint_dict(payload)


def build_graph(
    X: Any,
    *,
    spec: GraphBuilderSpec,
    seed: int = 0,
    dataset_fingerprint: str | None = None,
    preprocess_fingerprint: str | None = None,
    cache: bool = True,
    cache_dir: str | Path | None = None,
    edge_shard_size: int | None = None,
    resume: bool = True,
) -> GraphArtifact:
    """Build a graph from a dense feature matrix.

    Parameters
    ----------
    X:
        A 2D dense array-like of shape (n_nodes, n_features).
    spec:
        GraphBuilderSpec controlling scheme/backend/weights/normalization.
    seed:
        Seed used for deterministic components (notably the anchor scheme).
    dataset_fingerprint:
        Optional precomputed fingerprint for X (useful when X is already cached upstream).
    preprocess_fingerprint:
        Optional fingerprint of the preprocessing pipeline.
    cache:
        Whether to cache the built graph on disk.
    cache_dir:
        Override the default cache directory.
    edge_shard_size:
        If provided, store the edge arrays in sharded `.npz` files with at most this many
        edges per shard.
    resume:
        If True and `cache=True`, partial numpy chunk computations are resumed from the
        cache entry work directory when available.

    Returns
    -------
    GraphArtifact
    """
    start = perf_counter()
    validate_features(X)
    validate_builder_spec(spec)

    X_arr = np.asarray(X)
    n_nodes = int(X_arr.shape[0])

    ds_fp = dataset_fingerprint or fingerprint_array(X_arr)
    spec_fp = fingerprint_dict(spec.to_dict())
    g_fp = _graph_fingerprint(
        dataset_fingerprint=ds_fp,
        preprocess_fingerprint=preprocess_fingerprint,
        spec=spec,
        seed=int(seed),
    )

    cache_store = GraphCache(
        root=Path(cache_dir) if cache_dir is not None else GraphCache.default().root,
        edge_shard_size=edge_shard_size,
    )

    if cache and cache_store.exists(g_fp):
        graph, _ = cache_store.load(g_fp)
        logger.info(
            "Graph cached: fingerprint=%s n_nodes=%s n_edges=%s duration_s=%.3f",
            g_fp,
            graph.n_nodes,
            int(graph.edge_index.shape[1]),
            perf_counter() - start,
        )
        return graph

    # Optional resumable work directory inside the cache entry (only used by numpy backend).
    work_dir: Path | None = None
    if cache and resume:
        work_dir = cache_store.entry_dir(g_fp) / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)

    # Build raw edges + distances
    logger.info(
        "Graph build start: scheme=%s metric=%s backend=%s n_nodes=%s seed=%s",
        spec.scheme,
        spec.metric,
        spec.backend,
        n_nodes,
        seed,
    )
    if spec.scheme == "knn" and spec.k is not None and int(spec.k) <= 1:
        logger.warning("Graph spec k is very small: k=%s", spec.k)
    if spec.scheme == "epsilon" and spec.radius is not None and float(spec.radius) <= 0:
        logger.warning("Graph spec radius is non-positive: radius=%s", spec.radius)

    edge_index, distances = build_raw_edges(
        X_arr,
        spec=spec,
        seed=int(seed),
        work_dir=work_dir,
        resume=bool(resume),
    )

    # Turn distances into weights
    edge_weight = compute_edge_weights(
        distances=distances, weights=spec.weights, metric=spec.metric
    )

    # Post-process graph
    if spec.symmetrize != "none":
        edge_index, edge_weight = symmetrize_edges(
            n_nodes=n_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            mode=spec.symmetrize,
        )

    if spec.self_loops:
        edge_index, edge_weight = add_self_loops(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight
        )

    if spec.normalize != "none":
        edge_weight = normalize_edge_weights(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode=spec.normalize
        )

    if edge_weight is not None and not np.isfinite(edge_weight).all():
        raise GraphValidationError(
            "Non-finite edge weights detected (check input features and spec)"
        )

    graph = GraphArtifact(
        n_nodes=n_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        directed=(spec.symmetrize == "none"),
        meta={
            "fingerprint": g_fp,
            "dataset_fingerprint": ds_fp,
            "preprocess_fingerprint": preprocess_fingerprint,
            "spec_fingerprint": spec_fp,
            "seed": int(seed),
        },
    )

    if cache:
        manifest = {
            "fingerprint": g_fp,
            "dataset_fingerprint": ds_fp,
            "preprocess_fingerprint": preprocess_fingerprint,
            "spec": spec.to_dict(),
            "spec_fingerprint": spec_fp,
            "seed": int(seed),
        }
        cache_store.save(fingerprint=g_fp, graph=graph, manifest=manifest, overwrite=True)

    duration = perf_counter() - start
    logger.info(
        "Graph build done: fingerprint=%s n_nodes=%s n_edges=%s duration_s=%.3f",
        g_fp,
        n_nodes,
        int(edge_index.shape[1]),
        duration,
    )
    if logger.isEnabledFor(logging.DEBUG) and edge_index.size and edge_index.shape[1] <= 5_000_000:
        min_deg, mean_deg, max_deg, zero_deg = _degree_summary(edge_index, n_nodes)
        logger.debug(
            "Graph degrees: min=%s mean=%.2f max=%s zero=%s",
            min_deg,
            mean_deg,
            max_deg,
            zero_deg,
        )
        if n_nodes and zero_deg / float(n_nodes) > 0.2:
            logger.warning("Graph has many isolated nodes: zero_degree=%s", zero_deg)

    return graph
