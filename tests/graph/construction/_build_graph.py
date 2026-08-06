from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from modssc.graph import GraphBuilderSpec, GraphWeightsSpec, build_graph
from modssc.graph.cache import GraphCache, GraphCacheError
from modssc.graph.errors import GraphValidationError


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

    frozen = build_graph(
        X,
        spec=spec,
        seed=0,
        cache=True,
        require_cache_hit=True,
        cache_dir=tmp_path,
    )
    assert frozen.meta["fingerprint"] == g.meta["fingerprint"]


def test_frozen_graph_requires_prebuilt_cache(tmp_path) -> None:
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    spec = GraphBuilderSpec(
        scheme="knn",
        metric="euclidean",
        k=2,
        weights=GraphWeightsSpec(kind="binary"),
        backend="numpy",
    )
    with pytest.raises(GraphCacheError, match="requires cache=True"):
        build_graph(X, spec=spec, cache=False, require_cache_hit=True, cache_dir=tmp_path)
    with pytest.raises(GraphCacheError, match="Frozen graph cache entry is missing"):
        build_graph(X, spec=spec, cache=True, require_cache_hit=True, cache_dir=tmp_path)


def test_graph_rejects_fingerprint_pin_mismatches(tmp_path) -> None:
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    spec = GraphBuilderSpec(
        scheme="knn",
        metric="euclidean",
        k=2,
        weights=GraphWeightsSpec(kind="binary"),
        backend="numpy",
    )
    graph = build_graph(
        X,
        spec=spec,
        preprocess_fingerprint="preprocess:expected",
        cache=False,
        cache_dir=tmp_path,
    )
    pinned = build_graph(
        X,
        spec=spec,
        preprocess_fingerprint="preprocess:expected",
        expected_preprocess_fingerprint="preprocess:expected",
        expected_fingerprint=graph.meta["fingerprint"],
        cache=False,
        cache_dir=tmp_path,
    )
    assert pinned.meta["fingerprint"] == graph.meta["fingerprint"]

    with pytest.raises(GraphValidationError, match="preprocessing fingerprint differs"):
        build_graph(
            X,
            spec=spec,
            preprocess_fingerprint="preprocess:actual",
            expected_preprocess_fingerprint="preprocess:expected",
            cache=False,
            cache_dir=tmp_path,
        )
    with pytest.raises(GraphValidationError, match="Graph fingerprint differs"):
        build_graph(
            X,
            spec=spec,
            preprocess_fingerprint="preprocess:expected",
            expected_preprocess_fingerprint="preprocess:expected",
            expected_fingerprint="graph-wrong",
            cache=False,
            cache_dir=tmp_path,
        )


def test_precomputed_graph_identity_depends_on_content_not_path(tmp_path) -> None:
    first_path = tmp_path / "first" / "knn.npz"
    second_path = tmp_path / "second" / "knn.npz"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    np.savez(
        first_path,
        I=np.repeat(np.arange(3, dtype=np.int64)[:, None], 3, axis=1),
        J=np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]], dtype=np.int64),
        D=np.array(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 1.5], [0.0, 0.5, 2.0]],
            dtype=np.float64,
        ),
    )
    artifact = first_path.read_bytes()
    second_path.write_bytes(artifact)
    digest = hashlib.sha256(artifact).hexdigest()

    def _spec(path: Path) -> GraphBuilderSpec:
        return GraphBuilderSpec(
            scheme="knn",
            metric="euclidean",
            k=2,
            symmetrize="none",
            weights=GraphWeightsSpec(kind="binary"),
            normalize="none",
            self_loops=False,
            backend="precomputed",
            include_self_in_knn=True,
            precomputed_path=str(path),
            precomputed_sha256=digest,
        )

    X = np.zeros((3, 1), dtype=np.float32)
    first = build_graph(
        X,
        spec=_spec(first_path),
        dataset_fingerprint="dataset:fixed",
        cache=False,
    )
    second = build_graph(
        X,
        spec=_spec(second_path),
        dataset_fingerprint="dataset:fixed",
        cache=False,
    )

    assert first.meta["fingerprint"] == second.meta["fingerprint"]
    assert first.meta["spec_fingerprint"] == second.meta["spec_fingerprint"]
    np.testing.assert_array_equal(first.edge_index, second.edge_index)
    np.testing.assert_array_equal(first.edge_weight, second.edge_weight)


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
