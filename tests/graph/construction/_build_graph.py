from __future__ import annotations

import numpy as np

from modssc.graph import GraphBuilderSpec, GraphWeightsSpec, build_graph
from modssc.graph.cache import GraphCache


def test_build_knn_graph_shapes_and_meta(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6)).astype(np.float32)

    spec = GraphBuilderSpec(
        scheme="knn",
        metric="cosine",
        k=5,
        symmetrize="none",
        self_loops=False,
        normalize="none",
        weights=GraphWeightsSpec(kind="binary"),
        backend="numpy",
        chunk_size=16,
    )

    g = build_graph(X, spec=spec, seed=0, cache=True, cache_dir=tmp_path)
    assert g.n_nodes == 40
    assert g.edge_index.shape[0] == 2
    assert g.edge_index.shape[1] == 40 * 5
    assert g.edge_weight is not None
    assert g.edge_weight.shape[0] == g.edge_index.shape[1]
    assert g.meta["fingerprint"]
    assert g.directed is True

    g2 = build_graph(X, spec=spec, seed=0, cache=True, cache_dir=tmp_path)
    assert g2.meta["fingerprint"] == g.meta["fingerprint"]
    np.testing.assert_array_equal(g2.edge_index, g.edge_index)
    np.testing.assert_allclose(g2.edge_weight, g.edge_weight)


def test_build_epsilon_graph(tmp_path) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4)).astype(np.float32)

    spec = GraphBuilderSpec(
        scheme="epsilon",
        metric="euclidean",
        radius=1.0,
        symmetrize="or",
        self_loops=False,
        normalize="none",
        weights=GraphWeightsSpec(kind="heat", sigma=1.0),
        backend="numpy",
        chunk_size=10,
    )

    g = build_graph(X, spec=spec, seed=0, cache=False, cache_dir=tmp_path)
    assert g.n_nodes == 30
    assert g.edge_index.shape[0] == 2
    assert g.edge_weight is not None
    assert np.isfinite(g.edge_weight).all()


def test_build_anchor_graph(tmp_path) -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 5)).astype(np.float32)

    spec = GraphBuilderSpec(
        scheme="anchor",
        metric="cosine",
        k=6,
        n_anchors=12,
        anchors_k=3,
        candidate_limit=80,
        symmetrize="none",
        self_loops=False,
        normalize="none",
        weights=GraphWeightsSpec(kind="binary"),
        backend="numpy",
        chunk_size=20,
    )

    g = build_graph(X, spec=spec, seed=123, cache=False, cache_dir=tmp_path)
    assert g.n_nodes == 50
    assert g.edge_index.shape[0] == 2

    assert g.edge_index.shape[1] <= 50 * 6
    assert g.edge_weight is not None
    assert g.edge_weight.shape[0] == g.edge_index.shape[1]


def test_build_with_sharded_edge_storage(tmp_path) -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 6)).astype(np.float32)

    spec = GraphBuilderSpec(
        scheme="knn",
        metric="cosine",
        k=8,
        symmetrize="none",
        self_loops=False,
        normalize="none",
        weights=GraphWeightsSpec(kind="binary"),
        backend="numpy",
        chunk_size=20,
    )

    g = build_graph(X, spec=spec, seed=0, cache=True, cache_dir=tmp_path, edge_shard_size=50)
    fp = g.meta["fingerprint"]

    store = GraphCache(root=tmp_path, edge_shard_size=50)
    assert store.exists(fp)

    d = store.entry_dir(fp)

    assert not (d / "edge_index.npy").exists()
    assert any(p.name.startswith("edges_") and p.suffix == ".npz" for p in d.iterdir())

    g2, manifest = store.load(fp)
    assert manifest["_storage"]["edge"]["kind"] == "sharded"
    np.testing.assert_array_equal(g2.edge_index, g.edge_index)
    np.testing.assert_allclose(g2.edge_weight, g.edge_weight)
