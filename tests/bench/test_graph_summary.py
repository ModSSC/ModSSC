from __future__ import annotations

import numpy as np

from bench.orchestrators.graph import summarize_graph
from modssc.graph.artifacts import GraphArtifact


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
            "metric": "cosine",
            "k": 10,
            "symmetrize": "mutual",
            "weights": {"kind": "heat", "sigma": 1.0},
            "normalize": "rw",
            "self_loops": True,
            "backend": "sklearn",
            "feature_field": "features.X",
        },
    )

    assert info["n_nodes"] == 4
    assert info["n_edges"] == 2
    assert info["k"] == 10
    assert info["metric"] == "cosine"
    assert info["connected_components"] == 2
    assert info["largest_component_fraction"] == 0.5
