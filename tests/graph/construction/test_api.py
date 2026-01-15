from __future__ import annotations

import logging

import numpy as np

import modssc.graph.construction.api as graph_api
from modssc.graph import GraphBuilderSpec, GraphWeightsSpec, build_graph


def test_build_graph_debug_degree_summary(monkeypatch, caplog, tmp_path) -> None:
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    distances = np.array([0.1, 0.1], dtype=np.float32)

    monkeypatch.setattr(
        graph_api, "build_raw_edges", lambda *args, **kwargs: (edge_index, distances)
    )

    X = np.zeros((5, 2), dtype=np.float32)
    spec = GraphBuilderSpec(
        scheme="knn",
        metric="cosine",
        k=1,
        backend="numpy",
        normalize="none",
        symmetrize="none",
        self_loops=False,
        weights=GraphWeightsSpec(kind="binary"),
    )

    with caplog.at_level(logging.DEBUG, logger=graph_api.__name__):
        g = build_graph(
            X,
            spec=spec,
            seed=0,
            cache=False,
            cache_dir=tmp_path,
            resume=False,
        )

    assert g.n_nodes == 5


def test_build_graph_debug_degree_summary_no_warning(monkeypatch, caplog, tmp_path) -> None:
    edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
    distances = np.full(edge_index.shape[1], 0.1, dtype=np.float32)

    monkeypatch.setattr(
        graph_api, "build_raw_edges", lambda *args, **kwargs: (edge_index, distances)
    )

    X = np.zeros((5, 2), dtype=np.float32)
    spec = GraphBuilderSpec(
        scheme="knn",
        metric="cosine",
        k=1,
        backend="numpy",
        normalize="none",
        symmetrize="none",
        self_loops=False,
        weights=GraphWeightsSpec(kind="binary"),
    )

    with caplog.at_level(logging.DEBUG, logger=graph_api.__name__):
        g = build_graph(
            X,
            spec=spec,
            seed=0,
            cache=False,
            cache_dir=tmp_path,
            resume=False,
        )

    assert g.n_nodes == 5


def test_build_graph_debug_no_edges(monkeypatch, caplog, tmp_path) -> None:
    edge_index = np.zeros((2, 0), dtype=np.int64)
    distances = np.zeros((0,), dtype=np.float32)

    monkeypatch.setattr(
        graph_api, "build_raw_edges", lambda *args, **kwargs: (edge_index, distances)
    )

    X = np.zeros((3, 2), dtype=np.float32)
    spec = GraphBuilderSpec(
        scheme="knn",
        metric="cosine",
        k=1,
        backend="numpy",
        normalize="none",
        symmetrize="none",
        self_loops=False,
        weights=GraphWeightsSpec(kind="binary"),
    )

    with caplog.at_level(logging.DEBUG, logger=graph_api.__name__):
        g = build_graph(
            X,
            spec=spec,
            seed=0,
            cache=False,
            cache_dir=tmp_path,
            resume=False,
        )

    assert g.n_nodes == 3


def test_build_graph_warns_on_nonpositive_radius(monkeypatch, tmp_path) -> None:
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    distances = np.array([0.1, 0.1], dtype=np.float32)

    monkeypatch.setattr(graph_api, "validate_builder_spec", lambda spec: None)
    monkeypatch.setattr(
        graph_api, "build_raw_edges", lambda *args, **kwargs: (edge_index, distances)
    )

    X = np.zeros((3, 2), dtype=np.float32)
    spec = GraphBuilderSpec(
        scheme="epsilon",
        metric="cosine",
        k=1,
        radius=0.0,
        backend="numpy",
        normalize="none",
        symmetrize="none",
        self_loops=False,
        weights=GraphWeightsSpec(kind="binary"),
    )

    logger = logging.getLogger(graph_api.__name__)
    handlers = list(logger.handlers)
    prev_propagate = logger.propagate
    for handler in handlers:
        logger.removeHandler(handler)
    logger.propagate = False
    try:
        build_graph(
            X,
            spec=spec,
            seed=0,
            cache=False,
            cache_dir=tmp_path,
            resume=False,
        )
    finally:
        logger.propagate = prev_propagate
        for handler in handlers:
            logger.addHandler(handler)
