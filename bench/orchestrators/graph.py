from __future__ import annotations

import logging
from collections.abc import Mapping
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


def build(
    pre: PreprocessResult,
    *,
    spec_dict: Mapping[str, Any],
    seed: int,
    dataset_fingerprint: str | None,
    cache: bool,
    include_test: bool,
) -> GraphArtifact:
    start = perf_counter()
    spec = _spec_from_dict(spec_dict)
    key = spec.feature_field

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
    )
    _LOGGER.info(
        "Graph built: fingerprint=%s n_nodes=%s n_edges=%s duration_s=%.3f",
        graph.meta.get("fingerprint"),
        graph.n_nodes,
        int(graph.edge_index.shape[1]),
        perf_counter() - start,
    )
    return graph
