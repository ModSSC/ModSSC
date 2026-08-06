from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from modssc.graph.artifacts import GraphArtifact
from modssc.graph.construction.api import build_graph
from modssc.graph.specs import GraphBuilderSpec
from modssc.preprocess.types import PreprocessResult

_LOGGER = logging.getLogger(__name__)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


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


def _spec_from_dict(obj: Mapping[str, Any]) -> GraphBuilderSpec:
    return GraphBuilderSpec.from_dict(dict(obj))


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
        adj = coo_matrix((data, (edge_index[0], edge_index[1])), shape=(n_nodes, n_nodes))
        n_components, labels = connected_components(
            adj,
            directed=False,
            return_labels=True,
        )
        counts = np.bincount(labels, minlength=int(n_components))
        largest = int(counts.max()) if counts.size else 0
        return {
            "connected_components": int(n_components),
            "largest_component_fraction": float(largest / float(n_nodes)),
        }
    except Exception:
        parent = np.arange(n_nodes, dtype=np.int64)
        size = np.ones(n_nodes, dtype=np.int64)

        def find(x: int) -> int:
            while int(parent[x]) != x:
                parent[x] = parent[int(parent[x])]
                x = int(parent[x])
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if int(size[ra]) < int(size[rb]):
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        for src, dst in edge_index.T:
            union(int(src), int(dst))

        roots = np.asarray([find(i) for i in range(n_nodes)], dtype=np.int64)
        _, inverse = np.unique(roots, return_inverse=True)
        counts = np.bincount(inverse)
        largest = int(counts.max()) if counts.size else 0
        return {
            "connected_components": int(counts.size),
            "largest_component_fraction": float(largest / float(n_nodes)),
        }


def summarize_graph(graph: GraphArtifact, spec_dict: Mapping[str, Any] | None) -> dict[str, Any]:
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
        "precomputed_sha256": spec.get("precomputed_sha256"),
        "feature_field": spec.get("feature_field"),
    }
    info.update(_connected_component_stats(graph))
    return info


def build(
    pre: PreprocessResult,
    *,
    spec_dict: Mapping[str, Any],
    seed: int,
    dataset_fingerprint: str | None,
    cache: bool,
    require_cache_hit: bool,
    cache_dir: str | None,
    include_test: bool,
    expected_fingerprint: str | None = None,
    expected_preprocess_fingerprint: str | None = None,
) -> GraphArtifact:
    start = perf_counter()
    spec = _spec_from_dict(spec_dict)
    key = spec.feature_field
    cache_root = Path(cache_dir).expanduser().resolve() if cache_dir else None

    X_train = _feature_from_artifacts(pre, key, split="train")
    X_test = _feature_from_artifacts(pre, key, split="test") if include_test else None

    _LOGGER.info(
        "Graph start: include_test=%s cache=%s",
        bool(include_test),
        bool(cache),
    )
    _LOGGER.debug(
        "Graph spec: scheme=%s metric=%s k=%s radius=%s backend=%s feature_field=%s seed=%s",
        spec.scheme,
        spec.metric,
        spec.k,
        spec.radius,
        spec.backend,
        spec.feature_field,
        int(seed),
    )
    if cache_root is not None:
        _LOGGER.debug("Graph cache_dir: %s", str(cache_root))

    if X_test is not None:
        X = np.concatenate([_to_numpy(X_train), _to_numpy(X_test)], axis=0)
    else:
        X = _to_numpy(X_train)
    _LOGGER.debug("Graph features: shape=%s", tuple(np.asarray(X).shape))

    graph = build_graph(
        X,
        spec=spec,
        seed=int(seed),
        dataset_fingerprint=dataset_fingerprint,
        preprocess_fingerprint=pre.preprocess_fingerprint,
        cache=bool(cache),
        require_cache_hit=bool(require_cache_hit),
        expected_fingerprint=expected_fingerprint,
        expected_preprocess_fingerprint=expected_preprocess_fingerprint,
        cache_dir=cache_root,
    )
    _LOGGER.info(
        "Graph built: fingerprint=%s n_nodes=%s n_edges=%s duration_s=%.3f",
        graph.meta.get("fingerprint"),
        graph.n_nodes,
        int(graph.edge_index.shape[1]),
        perf_counter() - start,
    )
    return graph
