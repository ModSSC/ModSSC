from __future__ import annotations

import numpy as np

from modssc.preprocess.steps.graph.sparse_adjacency import SparseAdjacencyStep
from modssc.preprocess.store import ArtifactStore


def test_sparse_adjacency_includes_edge_weight_and_features():
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_weight = np.array([0.5, 0.5], dtype=np.float32)
    feats = np.array([[1.0], [2.0]], dtype=np.float32)
    store = ArtifactStore(
        {
            "graph.edge_index": edge_index,
            "graph.edge_weight": edge_weight,
            "features.X": feats,
        }
    )
    out = SparseAdjacencyStep().transform(store, rng=None)
    packed = out["features.X"]
    assert np.array_equal(packed["x"], feats)
    assert np.array_equal(packed["edge_index"], edge_index)
    assert np.array_equal(packed["edge_weight"], edge_weight)


def test_sparse_adjacency_falls_back_to_raw():
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    raw = np.array([[3.0], [4.0]], dtype=np.float32)
    store = ArtifactStore({"graph.edge_index": edge_index, "raw.X": raw})
    out = SparseAdjacencyStep().transform(store, rng=None)
    packed = out["features.X"]
    assert np.array_equal(packed["x"], raw)
    assert "edge_weight" not in packed
