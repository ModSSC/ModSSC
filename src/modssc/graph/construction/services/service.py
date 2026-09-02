from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ...artifacts import GraphArtifact
from ...cache import (
    GraphCache,
    GraphCacheError,
    graph_content_sha256,
    graph_implementation_identity,
)
from ...errors import GraphValidationError
from ...fingerprint import fingerprint_array, fingerprint_dict
from ...specs import GraphBuilderSpec
from ...validation import validate_builder_spec, validate_features
from ..builder import build_raw_edges, resolve_graph_backend
from ..ops.diagonal import zero_diagonal_edges
from ..ops.normalize import normalize_edge_weights
from ..ops.self_loops import add_self_loops
from ..ops.symmetrize import symmetrize_edges
from ..ops.weights import compute_edge_weights

logger = logging.getLogger(__name__)


def _degree_summary(edge_index: np.ndarray, n_nodes: int) -> tuple[int, float, int, int]:
    src = edge_index[0]
    deg = np.bincount(src, minlength=int(n_nodes))
    return int(deg.min()), float(deg.mean()), int(deg.max()), int((deg == 0).sum())


def _graph_fingerprint(
    *,
    dataset_fingerprint: str,
    preprocess_fingerprint: str | None,
    features_fingerprint: str,
    spec: GraphBuilderSpec,
    seed: int,
    producer_identity: dict[str, Any],
) -> str:
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "preprocess_fingerprint": preprocess_fingerprint,
        "features_fingerprint": features_fingerprint,
        "spec": spec.fingerprint_payload(),
        "seed": int(seed),
        "producer_identity": producer_identity,
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
    require_cache_hit: bool = False,
    expected_fingerprint: str | None = None,
    expected_preprocess_fingerprint: str | None = None,
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
        Optional upstream dataset identity. The exact feature-array fingerprint
        is always computed separately and remains part of the graph key, so a
        stale declared dataset identity cannot hide changed graph inputs.
    preprocess_fingerprint:
        Optional fingerprint of the preprocessing pipeline.
    cache:
        Whether to cache the built graph on disk.
    cache_dir:
        Override the default cache directory.
    require_cache_hit:
        If True, require a complete pre-existing cache entry and never build a
        graph in this process. Used by frozen scientific paper profiles.
    expected_fingerprint:
        Optional immutable pin for the graph fingerprint computed from the
        dataset, preprocessing, graph specification, and seed.
    expected_preprocess_fingerprint:
        Optional immutable pin for the preprocessing fingerprint used to build
        the graph.
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

    features_fp = fingerprint_array(X_arr)
    ds_fp = dataset_fingerprint or features_fp
    spec_fp = fingerprint_dict(spec.fingerprint_payload())
    resolved_backend = resolve_graph_backend(spec)
    producer_identity = graph_implementation_identity(
        component="construction", backend=resolved_backend
    )
    if (
        expected_preprocess_fingerprint is not None
        and preprocess_fingerprint != expected_preprocess_fingerprint
    ):
        raise GraphValidationError(
            "Graph preprocessing fingerprint differs from "
            "expected_preprocess_fingerprint: "
            f"computed {preprocess_fingerprint!r}, expected "
            f"{expected_preprocess_fingerprint!r}"
        )
    g_fp = _graph_fingerprint(
        dataset_fingerprint=ds_fp,
        preprocess_fingerprint=preprocess_fingerprint,
        features_fingerprint=features_fp,
        spec=spec,
        seed=int(seed),
        producer_identity=producer_identity,
    )
    if expected_fingerprint is not None and g_fp != expected_fingerprint:
        raise GraphValidationError(
            "Graph fingerprint differs from expected_fingerprint: "
            f"computed {g_fp}, expected {expected_fingerprint}"
        )

    cache_store = GraphCache(
        root=Path(cache_dir) if cache_dir is not None else GraphCache.default().root,
        edge_shard_size=edge_shard_size,
    )

    if require_cache_hit and not cache:
        raise GraphCacheError("require_cache_hit=True requires cache=True")

    if cache:
        entry_exists = cache_store.entry_dir(g_fp).exists()
        try:
            graph, _ = cache_store.load(
                g_fp,
                expected_manifest={
                    "dataset_fingerprint": ds_fp,
                    "preprocess_fingerprint": preprocess_fingerprint,
                    "features_fingerprint": features_fp,
                    "spec_fingerprint": spec_fp,
                    "seed": int(seed),
                    "producer_identity": producer_identity,
                    "resolved_backend": resolved_backend,
                },
            )
        except GraphCacheError as exc:
            if require_cache_hit:
                state = "invalid" if entry_exists else "missing"
                raise GraphCacheError(
                    f"Frozen graph cache entry is {state}: {cache_store.entry_dir(g_fp)}. {exc}"
                ) from exc
            if entry_exists:
                logger.warning(
                    "Ignoring invalid graph cache entry and rebuilding: fingerprint=%s error=%s",
                    g_fp,
                    exc,
                )
        else:
            logger.info(
                "Graph cached: fingerprint=%s n_nodes=%s n_edges=%s duration_s=%.3f",
                g_fp,
                graph.n_nodes,
                int(graph.edge_index.shape[1]),
                perf_counter() - start,
            )
            return graph

    # Resumable chunks live outside the immutable published entry. Multiple
    # same-key builders may safely converge because chunk files and publication
    # are atomic, while a late builder can never recreate files in a live entry.
    work_dir: Path | None = None
    if cache and resume:
        work_dir = cache_store.work_dir(g_fp)
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
        distances=distances,
        weights=spec.weights,
        metric=spec.metric,
        edge_index=edge_index,
        n_nodes=n_nodes,
        dtype=spec.edge_weight_dtype,
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

    if spec.diagonal_policy == "zero":
        edge_index, edge_weight = zero_diagonal_edges(
            edge_index=edge_index,
            edge_weight=edge_weight,
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
            "features_fingerprint": features_fp,
            "spec_fingerprint": spec_fp,
            "seed": int(seed),
            "edge_weight_dtype": spec.edge_weight_dtype,
            "resolved_backend": resolved_backend,
            "producer_identity": producer_identity,
        },
    )
    graph.meta["graph_content_sha256"] = graph_content_sha256(graph)

    if cache:
        manifest = {
            "fingerprint": g_fp,
            "dataset_fingerprint": ds_fp,
            "preprocess_fingerprint": preprocess_fingerprint,
            "features_fingerprint": features_fp,
            "spec": spec.to_dict(),
            "spec_fingerprint": spec_fp,
            "seed": int(seed),
            "resolved_backend": resolved_backend,
            "producer_identity": producer_identity,
            "graph_content_sha256": graph.meta["graph_content_sha256"],
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
