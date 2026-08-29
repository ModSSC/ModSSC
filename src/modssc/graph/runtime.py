"""Native graph materialization from resolved preprocessing outputs."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from modssc.preprocess.types import PreprocessResult

from .artifacts import GraphArtifact
from .construction.api import build_graph
from .errors import GraphValidationError
from .fingerprint import fingerprint_dict
from .specs import GraphBuilderSpec

_LOGGER = logging.getLogger(__name__)

GRAPH_FEATURE_IDENTITY_SCHEMA_VERSION = 2
_GRAPH_FEATURE_IDENTITY_PREFIX = "preprocess-content-v2:"


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _array_sha256(value: Any) -> str:
    """Hash every feature byte together with its dtype and shape."""

    array = np.ascontiguousarray(_to_numpy(value))
    if array.dtype.hasobject:
        raise GraphValidationError("Graph feature content hashing requires a numeric array")
    digest = hashlib.sha256()
    # This deliberately matches the VAE output attestation format. It is a full
    # content hash, unlike the bounded fallback used for generic dataset caches.
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(int(dim) for dim in array.shape)).encode("utf-8"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _feature_from_artifacts(pre: PreprocessResult, key: str, *, split: str) -> Any:
    if split == "train":
        if pre.train_artifacts.has(key):
            return pre.train_artifacts.get(key)
        return pre.dataset.train.X
    if split == "test":
        if pre.test_artifacts is not None and pre.test_artifacts.has(key):
            return pre.test_artifacts.get(key)
        if pre.dataset.test is None:
            return None
        return pre.dataset.test.X
    raise ValueError(f"Unknown split: {split}")


def _feature_producer_identity(
    pre: PreprocessResult,
    *,
    key: str,
    split: str,
    feature_array: np.ndarray,
) -> dict[str, Any]:
    """Validate an optional producer attestation and return a content identity."""

    store = pre.train_artifacts if split == "train" else pre.test_artifacts
    info = store.get(f"{key}.info") if store is not None else None
    output = info.get("output") if isinstance(info, Mapping) else None
    declared_sha256 = output.get("content_sha256") if isinstance(output, Mapping) else None
    actual_sha256 = _array_sha256(feature_array)
    if declared_sha256 is not None and str(declared_sha256) != actual_sha256:
        raise GraphValidationError(
            f"Graph feature artifact {key!r} on split {split!r} changed after "
            "preprocessing: its declared content_sha256 no longer matches the "
            "actual feature bytes"
        )
    producer_fingerprint = (
        output.get("identity_fingerprint") if isinstance(output, Mapping) else None
    )
    return {
        "content_sha256": actual_sha256,
        "shape": [int(dim) for dim in feature_array.shape],
        "dtype": str(feature_array.dtype),
        "producer_identity_fingerprint": (
            None if producer_fingerprint is None else str(producer_fingerprint)
        ),
    }


def _content_bound_preprocess_identity(
    pre: PreprocessResult,
    *,
    feature_field: str,
    include_test: bool,
    train_features: np.ndarray,
    test_features: np.ndarray | None,
    combined_features: np.ndarray,
) -> tuple[str, dict[str, Any]]:
    """Bind the semantic plan identity to the exact graph input content."""

    split_identities: dict[str, Any] = {
        "train": _feature_producer_identity(
            pre,
            key=feature_field,
            split="train",
            feature_array=train_features,
        )
    }
    if test_features is not None:
        split_identities["test"] = _feature_producer_identity(
            pre,
            key=feature_field,
            split="test",
            feature_array=test_features,
        )
    payload = {
        "kind": "graph_preprocess_features",
        "version": GRAPH_FEATURE_IDENTITY_SCHEMA_VERSION,
        "semantic_preprocess_fingerprint": str(pre.preprocess_fingerprint),
        "feature_field": str(feature_field),
        "include_test": bool(include_test),
        "combined_content_sha256": _array_sha256(combined_features),
        "combined_shape": [int(dim) for dim in combined_features.shape],
        "combined_dtype": str(combined_features.dtype),
        "splits": split_identities,
    }
    identity = _GRAPH_FEATURE_IDENTITY_PREFIX + fingerprint_dict(payload)
    return identity, payload


def _connected_component_stats(graph: GraphArtifact) -> dict[str, Any]:
    n_nodes = int(graph.n_nodes)
    if n_nodes <= 0:
        return {"connected_components": 0, "largest_component_fraction": 0.0}

    edge_index = np.asarray(graph.edge_index, dtype=np.int64)
    if edge_index.size == 0:
        return {
            "connected_components": n_nodes,
            "largest_component_fraction": 1.0 / float(n_nodes),
        }

    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        data = np.ones(int(edge_index.shape[1]), dtype=np.int8)
        adjacency = coo_matrix(
            (data, (edge_index[0], edge_index[1])),
            shape=(n_nodes, n_nodes),
        )
        n_components, labels = connected_components(
            adjacency,
            directed=False,
            return_labels=True,
        )
        counts = np.bincount(labels, minlength=int(n_components))
    except (ImportError, ValueError, TypeError, IndexError):
        parent = np.arange(n_nodes, dtype=np.int64)
        sizes = np.ones(n_nodes, dtype=np.int64)

        def find(node: int) -> int:
            while int(parent[node]) != node:
                parent[node] = parent[int(parent[node])]
                node = int(parent[node])
            return node

        def union(source: int, destination: int) -> None:
            source_root = find(source)
            destination_root = find(destination)
            if source_root == destination_root:
                return
            if int(sizes[source_root]) < int(sizes[destination_root]):
                source_root, destination_root = destination_root, source_root
            parent[destination_root] = source_root
            sizes[source_root] += sizes[destination_root]

        for source, destination in edge_index.T:
            union(int(source), int(destination))
        roots = np.asarray([find(node) for node in range(n_nodes)], dtype=np.int64)
        _, inverse = np.unique(roots, return_inverse=True)
        counts = np.bincount(inverse)

    largest = int(counts.max()) if counts.size else 0
    return {
        "connected_components": int(counts.size),
        "largest_component_fraction": float(largest / float(n_nodes)),
    }


def summarize_graph(
    graph: GraphArtifact,
    spec_dict: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return graph diagnostics without coupling them to a benchmark runner."""

    spec = dict(spec_dict or {})
    weights = spec.get("weights") if isinstance(spec.get("weights"), Mapping) else {}
    info: dict[str, Any] = {
        "n_nodes": int(graph.n_nodes),
        "n_edges": int(graph.n_edges),
        "directed": bool(graph.directed),
        "k": spec.get("k"),
        "metric": spec.get("metric"),
        "scheme": spec.get("scheme"),
        "symmetrize": spec.get("symmetrize"),
        "weights": dict(weights),
        "normalize": spec.get("normalize"),
        "self_loops": spec.get("self_loops"),
        "include_self_in_knn": spec.get("include_self_in_knn", False),
        "edge_weight_dtype": spec.get("edge_weight_dtype", "float32"),
        "backend": spec.get("backend"),
        "annoy_n_trees": spec.get("annoy_n_trees"),
        "annoy_query_k": spec.get("annoy_query_k"),
        "annoy_search_k": spec.get("annoy_search_k"),
        "annoy_rerank": spec.get("annoy_rerank"),
        "precomputed_sha256": spec.get("precomputed_sha256"),
        "feature_field": spec.get("feature_field"),
        "preprocess_fingerprint": graph.meta.get("preprocess_fingerprint"),
        "preprocess_semantic_fingerprint": graph.meta.get("preprocess_semantic_fingerprint"),
        "feature_content_sha256": graph.meta.get("feature_content_sha256"),
        "feature_identity_schema_version": graph.meta.get("feature_identity_schema_version"),
    }
    info.update(_connected_component_stats(graph))
    return info


def build_graph_from_preprocess(
    pre: PreprocessResult,
    *,
    spec: GraphBuilderSpec,
    seed: int,
    dataset_fingerprint: str | None,
    cache: bool,
    require_cache_hit: bool,
    cache_dir: Path | None,
    include_test: bool,
    expected_fingerprint: str | None = None,
    expected_preprocess_fingerprint: str | None = None,
) -> GraphArtifact:
    """Build a graph whose cache identity commits to every input feature byte.

    ``expected_preprocess_fingerprint`` pins the content-bound v2 identity exposed
    as ``graph.meta['preprocess_fingerprint']``. Pins for the former semantic-only
    identity intentionally fail instead of reusing an ambiguously keyed graph.
    """

    start = perf_counter()
    X_train = _to_numpy(_feature_from_artifacts(pre, spec.feature_field, split="train"))
    X_test = (
        _to_numpy(_feature_from_artifacts(pre, spec.feature_field, split="test"))
        if include_test
        else None
    )
    X = np.concatenate([X_train, X_test], axis=0) if X_test is not None else X_train
    preprocess_identity, feature_identity = _content_bound_preprocess_identity(
        pre,
        feature_field=spec.feature_field,
        include_test=include_test,
        train_features=X_train,
        test_features=X_test,
        combined_features=X,
    )
    _LOGGER.info(
        "Graph start: include_test=%s cache=%s n_nodes=%s feature_sha256=%s",
        bool(include_test),
        bool(cache),
        int(X.shape[0]),
        feature_identity["combined_content_sha256"],
    )
    graph = build_graph(
        X,
        spec=spec,
        seed=int(seed),
        dataset_fingerprint=dataset_fingerprint,
        preprocess_fingerprint=preprocess_identity,
        cache=bool(cache),
        require_cache_hit=bool(require_cache_hit),
        expected_fingerprint=expected_fingerprint,
        expected_preprocess_fingerprint=expected_preprocess_fingerprint,
        cache_dir=cache_dir,
    )
    graph.meta.update(
        {
            "preprocess_semantic_fingerprint": str(pre.preprocess_fingerprint),
            "feature_content_sha256": feature_identity["combined_content_sha256"],
            "feature_identity_schema_version": GRAPH_FEATURE_IDENTITY_SCHEMA_VERSION,
            "feature_identity": feature_identity,
        }
    )
    _LOGGER.info(
        "Graph built: fingerprint=%s n_nodes=%s n_edges=%s duration_s=%.3f",
        graph.meta.get("fingerprint"),
        graph.n_nodes,
        graph.n_edges,
        perf_counter() - start,
    )
    return graph


__all__ = ["build_graph_from_preprocess", "summarize_graph"]
