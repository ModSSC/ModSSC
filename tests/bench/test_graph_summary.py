from __future__ import annotations

import numpy as np
import pytest

from bench.orchestrators import graph as graph_orchestrator
from modssc.data_loader.types import LoadedDataset, Split
from modssc.graph.artifacts import GraphArtifact
from modssc.graph.runtime import summarize_graph
from modssc.preprocess.store import ArtifactStore
from modssc.preprocess.types import PreprocessResult, ResolvedPlan


def test_summarize_graph_reports_connectivity_and_spec() -> None:
    graph = GraphArtifact(
        n_nodes=4,
        edge_index=np.asarray([[0, 2], [1, 3]], dtype=np.int64),
        edge_weight=np.ones(2, dtype=np.float32),
    )

    info = summarize_graph(
        graph,
        {
            "scheme": "knn",
            "metric": "euclidean",
            "k": 10,
            "symmetrize": "mutual",
            "weights": {"kind": "heat", "sigma": 1.0},
            "normalize": "rw",
            "self_loops": True,
            "backend": "annoy",
            "annoy_n_trees": 10,
            "annoy_query_k": 30,
            "annoy_search_k": -1,
            "annoy_rerank": False,
            "feature_field": "features.X",
        },
    )

    assert info["n_nodes"] == 4
    assert info["n_edges"] == 2
    assert info["k"] == 10
    assert info["metric"] == "euclidean"
    assert info["backend"] == "annoy"
    assert info["annoy_n_trees"] == 10
    assert info["annoy_query_k"] == 30
    assert info["annoy_search_k"] == -1
    assert info["annoy_rerank"] is False
    assert info["connected_components"] == 2
    assert info["largest_component_fraction"] == 0.5


def test_graph_orchestrator_forwards_fingerprint_pins(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    expected_graph = GraphArtifact(
        n_nodes=2,
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        edge_weight=np.ones(2, dtype=np.float32),
        meta={"fingerprint": "graph:expected"},
    )

    def fake_build_graph_from_preprocess(_pre, **kwargs):
        captured.update(kwargs)
        return expected_graph

    monkeypatch.setattr(
        graph_orchestrator,
        "build_graph_from_preprocess",
        fake_build_graph_from_preprocess,
    )
    dataset = LoadedDataset(
        train=Split(X=np.zeros((2, 3), dtype=np.float32), y=np.array([0, 1])),
        meta={"dataset_fingerprint": "dataset:expected"},
    )
    pre = PreprocessResult(
        dataset=dataset,
        plan=ResolvedPlan(steps=()),
        preprocess_fingerprint="preprocess:expected",
        train_artifacts=ArtifactStore(),
    )

    result = graph_orchestrator.build(
        pre,
        spec_dict={"scheme": "knn", "k": 1, "feature_field": "features.X"},
        seed=7,
        dataset_fingerprint="dataset:expected",
        cache=True,
        require_cache_hit=True,
        cache_dir=str(tmp_path),
        include_test=False,
        expected_fingerprint="graph:expected",
        expected_preprocess_fingerprint="preprocess:expected",
    )

    assert result is expected_graph
    assert captured["spec"].feature_field == "features.X"
    assert captured["expected_fingerprint"] == "graph:expected"
    assert captured["expected_preprocess_fingerprint"] == "preprocess:expected"
