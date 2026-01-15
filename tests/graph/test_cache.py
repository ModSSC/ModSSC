import contextlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.graph.artifacts import DatasetViews, GraphArtifact
from modssc.graph.cache import (
    GraphCache,
    GraphCacheError,
    ViewsCache,
    _safe_write_json,
)
from modssc.graph.construction.backends.sklearn_backend import (
    epsilon_edges_sklearn,
    knn_edges_sklearn,
)


def test_graph_cache_sharded(tmp_path):
    cache = GraphCache(root=tmp_path, edge_shard_size=2)

    n_nodes = 10
    edge_index = np.zeros((2, 5), dtype=np.int64)
    edge_weight = np.random.rand(5).astype(np.float32)
    graph = GraphArtifact(n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight)

    fingerprint = "test_sharded"
    cache.save(fingerprint=fingerprint, graph=graph, manifest={"n_nodes": n_nodes})

    d = cache.entry_dir(fingerprint)
    assert (d / "edges_0000.npz").exists()
    assert (d / "edges_0001.npz").exists()
    assert (d / "edges_0002.npz").exists()
    assert not (d / "edge_index.npy").exists()

    loaded, _ = cache.load(fingerprint)
    assert loaded.n_nodes == n_nodes
    assert np.array_equal(loaded.edge_index, edge_index)
    assert np.array_equal(loaded.edge_weight, edge_weight)


def test_graph_cache_save_no_overwrite(tmp_path):
    cache = GraphCache(root=tmp_path)
    graph = GraphArtifact(
        n_nodes=10,
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_weight=None,
        directed=False,
        meta={},
    )
    manifest = {"n_nodes": 10}

    d = cache.save(fingerprint="fp1", graph=graph, manifest=manifest)
    assert d.exists()

    with patch("modssc.graph.cache.GraphCache._clear_entry_dir") as mock_clear:
        cache.save(fingerprint="fp1", graph=graph, manifest=manifest, overwrite=False)
        mock_clear.assert_not_called()

        cache.save(fingerprint="fp1", graph=graph, manifest=manifest, overwrite=True)
        mock_clear.assert_called()


def test_graph_cache_missing_root_methods(tmp_path):
    root = tmp_path / "non_existent"
    cache = GraphCache(root=root)

    assert cache.list() == []
    assert cache.purge() == 0


def test_graph_cache_errors(tmp_path):
    cache = GraphCache(root=tmp_path)

    with pytest.raises(GraphCacheError, match="Missing cached graph manifest"):
        cache.load("missing")

    d = cache.entry_dir("corrupt")
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("not json")
    with pytest.raises(GraphCacheError, match="Invalid json payload"):
        cache.load("corrupt")

    d = cache.entry_dir("missing_edges")
    d.mkdir(parents=True)
    _safe_write_json(d / "manifest.json", {"n_nodes": 10})
    with pytest.raises(GraphCacheError, match="Missing cached edge_index.npy"):
        cache.load("missing_edges")

    d = cache.entry_dir("corrupt_weight")
    d.mkdir(parents=True)
    _safe_write_json(d / "manifest.json", {"n_nodes": 10})
    np.save(d / "edge_index.npy", np.zeros((2, 5)))
    (d / "edge_weight.npy").write_text("bad")
    with pytest.raises(GraphCacheError, match="Corrupted cached edge_weight.npy"):
        cache.load("corrupt_weight")


def test_graph_cache_sharded_errors(tmp_path):
    cache = GraphCache(root=tmp_path)
    d = cache.entry_dir("sharded_err")
    d.mkdir(parents=True)

    manifest = {"n_nodes": 10, "_storage": {"edge": {"kind": "sharded", "num_shards": 2}}}
    _safe_write_json(d / "manifest.json", manifest)

    with pytest.raises(GraphCacheError, match="Missing edge shard"):
        cache.load("sharded_err")

    np.savez_compressed(d / "edges_0000.npz", other="data")
    with pytest.raises(GraphCacheError, match="Shard missing edge_index"):
        cache.load("sharded_err")


def test_graph_cache_sharded_inconsistent_edge_weight(tmp_path):
    cache = GraphCache(root=tmp_path)
    d = cache.entry_dir("sharded_inconsistent")
    d.mkdir(parents=True)

    manifest = {"n_nodes": 2, "_storage": {"edge": {"kind": "sharded", "num_shards": 2}}}
    _safe_write_json(d / "manifest.json", manifest)

    np.savez_compressed(d / "edges_0000.npz", edge_index=np.zeros((2, 1)), edge_weight=np.zeros(1))
    np.savez_compressed(d / "edges_0001.npz", edge_index=np.zeros((2, 1)))

    with pytest.raises(GraphCacheError, match="Inconsistent edge_weight"):
        cache.load("sharded_inconsistent")


def test_graph_cache_sharded_total_zero(tmp_path):
    cache = GraphCache(root=tmp_path)
    d = cache.entry_dir("sharded_zero")
    d.mkdir(parents=True)

    np.savez_compressed(d / "edges_0000.npz", edge_index=np.zeros((2, 0)), edge_weight=np.zeros(0))
    edge_index, edge_weight = cache._load_edges_sharded(d, num_shards=1)
    assert edge_index.shape == (2, 0)
    assert edge_weight.shape == (0,)


def test_graph_cache_sharded_skips_empty_shard(tmp_path):
    cache = GraphCache(root=tmp_path)
    d = cache.entry_dir("sharded_skip")
    d.mkdir(parents=True)

    np.savez_compressed(d / "edges_0000.npz", edge_index=np.zeros((2, 0)), edge_weight=np.zeros(0))
    np.savez_compressed(d / "edges_0001.npz", edge_index=np.zeros((2, 1)), edge_weight=np.ones(1))
    edge_index, edge_weight = cache._load_edges_sharded(d, num_shards=2)
    assert edge_index.shape == (2, 1)
    assert edge_weight.shape == (1,)


def test_graph_cache_sharded_missing_edge_index_second_pass(tmp_path, monkeypatch):
    cache = GraphCache(root=tmp_path)
    d = cache.entry_dir("sharded_missing_second")
    d.mkdir(parents=True)
    (d / "edges_0000.npz").touch()

    class DummyNPZ(dict):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    calls = {"n": 0}

    def fake_load(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return DummyNPZ({"edge_index": np.zeros((2, 1)), "edge_weight": np.zeros(1)})
        return DummyNPZ({})

    monkeypatch.setattr("modssc.graph.cache.np.load", fake_load)
    with pytest.raises(GraphCacheError, match="Shard missing edge_index"):
        cache._load_edges_sharded(d, num_shards=1)


def test_graph_cache_sharded_missing_edge_weight_second_pass(tmp_path, monkeypatch):
    cache = GraphCache(root=tmp_path)
    d = cache.entry_dir("sharded_missing_weight_second")
    d.mkdir(parents=True)
    (d / "edges_0000.npz").touch()

    class DummyNPZ(dict):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    calls = {"n": 0}

    def fake_load(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return DummyNPZ({"edge_index": np.zeros((2, 1)), "edge_weight": np.zeros(1)})
        return DummyNPZ({"edge_index": np.zeros((2, 1))})

    monkeypatch.setattr("modssc.graph.cache.np.load", fake_load)
    with pytest.raises(GraphCacheError, match="Shard missing edge_weight"):
        cache._load_edges_sharded(d, num_shards=1)


def test_views_cache_missing_root_methods(tmp_path):
    root = tmp_path / "non_existent"
    cache = ViewsCache(root=root)

    assert cache.list() == []


def test_views_cache(tmp_path):
    cache = ViewsCache(root=tmp_path)

    views = DatasetViews(
        views={"view_a": np.zeros((5, 2))},
        y=np.zeros(5),
        masks={"train": np.zeros(5, dtype=bool)},
        meta={"m": 1},
    )

    fingerprint = "views_fp"
    cache.save(fingerprint=fingerprint, views=views, manifest={"orig": "data"})

    assert cache.exists(fingerprint)
    assert "views_fp" in cache.list()

    loaded, manifest = cache.load(fingerprint, y=views.y, masks=views.masks)
    assert np.array_equal(loaded.views["view_a"], views.views["view_a"])
    assert manifest["orig"] == "data"
    assert manifest["meta"]["m"] == 1


def test_views_cache_errors(tmp_path):
    cache = ViewsCache(root=tmp_path)

    with pytest.raises(GraphCacheError, match="Missing cached views manifest"):
        cache.load("missing", y=np.array([]), masks={})

    d = cache.entry_dir("missing_npz")
    d.mkdir(parents=True)
    _safe_write_json(d / "manifest.json", {})
    with pytest.raises(GraphCacheError, match="Missing cached views.npz"):
        cache.load("missing_npz", y=np.array([]), masks={})


def test_safe_write_json_cleanup(tmp_path):
    p = tmp_path / "test.json"

    with patch("builtins.open", side_effect=OSError("fail")), contextlib.suppress(OSError):
        _safe_write_json(p, {})

    assert len(list(tmp_path.glob("*.tmp"))) == 0


def test_graph_cache_purge(tmp_path):
    cache = GraphCache(root=tmp_path)
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    (tmp_path / "f1").touch()

    n = cache.purge()
    assert n == 2
    assert not (tmp_path / "d1").exists()
    assert (tmp_path / "f1").exists()


def test_graph_cache_internals(tmp_path):
    cache = GraphCache(root=tmp_path)

    with (
        patch("builtins.open", side_effect=OSError("Disk full")),
        pytest.raises(OSError),
    ):
        from modssc.graph.cache import _safe_write_json

        _safe_write_json(tmp_path / "test.json", {})

    p = tmp_path / "list.json"
    with open(p, "w") as f:
        json.dump([1, 2, 3], f)

    with pytest.raises(GraphCacheError, match="Invalid json payload"):
        from modssc.graph.cache import _safe_read_json

        _safe_read_json(p)

    d = tmp_path / "race_condition"
    d.mkdir()
    f = d / "file.txt"
    f.touch()

    with patch.object(Path, "unlink", side_effect=FileNotFoundError):
        cache._clear_entry_dir(d)

    d_single = cache.entry_dir("single_missing")
    d_single.mkdir()
    (d_single / "manifest.json").write_text("{}")

    with pytest.raises(GraphCacheError, match="Missing cached edge_index.npy"):
        cache._load_edges_single(d_single)

    d_corrupt = cache.entry_dir("single_corrupt")
    d_corrupt.mkdir()
    np.save(d_corrupt / "edge_index.npy", np.zeros((2, 0)))
    (d_corrupt / "edge_weight.npy").write_text("garbage")

    with pytest.raises(GraphCacheError, match="Corrupted cached edge_weight.npy"):
        cache._load_edges_single(d_corrupt)

    d_sharded = cache.entry_dir("sharded_missing")
    d_sharded.mkdir()

    with pytest.raises(GraphCacheError, match="Missing edge shard"):
        cache._load_edges_sharded(d_sharded, num_shards=2)

    d_sharded_bad = cache.entry_dir("sharded_bad")
    d_sharded_bad.mkdir()
    np.savez(d_sharded_bad / "edges_0000.npz", other="data")

    with pytest.raises(GraphCacheError, match="Shard missing edge_index"):
        cache._load_edges_sharded(d_sharded_bad, num_shards=1)


def test_cache_defaults():
    gc = GraphCache.default()
    assert isinstance(gc.root, Path)
    assert "modssc" in str(gc.root)

    vc = ViewsCache.default()
    assert isinstance(vc.root, Path)
    assert "modssc" in str(vc.root)


def test_cache_list_purge(tmp_path):
    cache = GraphCache(root=tmp_path)
    assert cache.list() == []
    assert cache.purge() == 0

    cache.entry_dir("e1").mkdir()
    cache.entry_dir("e2").mkdir()
    (tmp_path / "file.txt").touch()

    assert set(cache.list()) == {"e1", "e2"}

    n = cache.purge()
    assert n == 2
    assert cache.list() == []
    assert (tmp_path / "file.txt").exists()

    vc = ViewsCache(root=tmp_path)
    assert vc.list() == []


def test_graph_cache_more_internals(tmp_path):
    cache = GraphCache(root=tmp_path)

    with (
        patch("builtins.open", side_effect=OSError("Disk full")),
        patch("os.remove", side_effect=OSError("Cannot remove")),
        pytest.raises(OSError),
    ):
        from modssc.graph.cache import _safe_write_json

        _safe_write_json(tmp_path / "test.json", {})

    assert not cache.exists("missing")
    d = cache.entry_dir("exists")
    d.mkdir()
    (d / "manifest.json").touch()
    assert cache.exists("exists")

    cache._clear_entry_dir(tmp_path / "non_existent")

    d_sub = tmp_path / "with_subdir"
    d_sub.mkdir()
    (d_sub / "subdir").mkdir()
    cache._clear_entry_dir(d_sub)
    assert not (d_sub / "subdir").exists()

    cache_sharded = GraphCache(root=tmp_path, edge_shard_size=100)
    graph_small = GraphArtifact(n_nodes=10, edge_index=np.zeros((2, 50)))
    cache_sharded.save(fingerprint="small_sharded", graph=graph_small, manifest={})

    assert (cache_sharded.entry_dir("small_sharded") / "edge_index.npy").exists()
    assert not (cache_sharded.entry_dir("small_sharded") / "edges_0000.npz").exists()

    graph_no_w = GraphArtifact(n_nodes=10, edge_index=np.zeros((2, 10)), edge_weight=None)
    cache.save(fingerprint="no_weights", graph=graph_no_w, manifest={})
    assert not (cache.entry_dir("no_weights") / "edge_weight.npy").exists()

    g_loaded, _ = cache.load("no_weights")
    assert g_loaded.edge_weight is None

    d_legacy = cache.entry_dir("legacy")
    d_legacy.mkdir()
    np.save(d_legacy / "edge_index.npy", np.zeros((2, 10)))
    (d_legacy / "manifest.json").write_text(json.dumps({"n_nodes": 10}))

    g_legacy, _ = cache.load("legacy")
    assert g_legacy.n_nodes == 10


def test_views_cache_more(tmp_path):
    vc = ViewsCache(root=tmp_path)

    assert not vc.exists("missing")

    d = vc.entry_dir("overwrite")
    d.mkdir(parents=True)
    (d / "old_file.txt").touch()

    views = DatasetViews(views={"v": np.zeros((5, 2))}, y=np.zeros(5))
    vc.save(fingerprint="overwrite", views=views, manifest={})

    assert not (d / "old_file.txt").exists()
    assert (d / "views.npz").exists()

    d_broken = vc.entry_dir("broken")
    d_broken.mkdir()
    (d_broken / "manifest.json").write_text("{}")

    with pytest.raises(GraphCacheError, match="Missing cached views.npz"):
        vc.load("broken", y=np.zeros(5), masks={})


def test_graph_cache_sharded_no_weights(tmp_path):
    cache = GraphCache(root=tmp_path, edge_shard_size=2)
    edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    graph = GraphArtifact(n_nodes=5, edge_index=edge_index, edge_weight=None)

    cache.save(fingerprint="sharded_no_w", graph=graph, manifest={})

    d = cache.entry_dir("sharded_no_w")
    assert (d / "edges_0000.npz").exists()

    g_loaded, _ = cache.load("sharded_no_w")
    assert np.array_equal(g_loaded.edge_index, edge_index)
    assert g_loaded.edge_weight is None


def test_views_cache_overwrite_logic(tmp_path):
    vc = ViewsCache(root=tmp_path)
    d = vc.entry_dir("overwrite_logic")
    d.mkdir(parents=True)
    (d / "keep_me.txt").touch()
    (d / "subdir").mkdir()

    views = DatasetViews(views={"v": np.zeros((5, 2))}, y=np.zeros(5))

    vc.save(fingerprint="overwrite_logic", views=views, manifest={}, overwrite=False)
    assert (d / "keep_me.txt").exists()
    assert (d / "subdir").exists()

    vc.save(fingerprint="overwrite_logic", views=views, manifest={}, overwrite=True)
    assert not (d / "keep_me.txt").exists()
    assert (d / "subdir").exists()


def test_graph_cache_small_shard_fallback(tmp_path):
    cache = GraphCache(root=tmp_path / "cache_fallback", edge_shard_size=100)
    edge_index = np.zeros((2, 5), dtype=np.int64)
    edge_weight = np.random.rand(5).astype(np.float32)
    graph = GraphArtifact(n_nodes=10, edge_index=edge_index, edge_weight=edge_weight, directed=True)

    cache.save(fingerprint="small_shard", graph=graph, manifest={"n_nodes": 10, "directed": True})

    entry_dir = cache.entry_dir("small_shard")
    assert (entry_dir / "edge_index.npy").exists()
    assert not (entry_dir / "edges_0000.npz").exists()

    g_loaded, _ = cache.load("small_shard")
    assert g_loaded.edge_index.shape[1] == 5


def test_graph_cache_corrupted_shards_missing_file(tmp_path):
    cache = GraphCache(root=tmp_path / "cache_corrupt", edge_shard_size=2)
    edge_index = np.zeros((2, 5), dtype=np.int64)
    graph = GraphArtifact(n_nodes=10, edge_index=edge_index, edge_weight=None, directed=True)

    cache.save(fingerprint="corrupt_fp", graph=graph, manifest={"n_nodes": 10, "directed": True})

    (cache.entry_dir("corrupt_fp") / "edges_0001.npz").unlink()

    with pytest.raises(GraphCacheError, match="Missing edge shard"):
        cache.load("corrupt_fp")


def test_graph_cache_corrupted_shards_missing_key(tmp_path):
    cache = GraphCache(root=tmp_path / "cache_corrupt_key", edge_shard_size=2)
    edge_index = np.zeros((2, 5), dtype=np.int64)
    graph = GraphArtifact(n_nodes=10, edge_index=edge_index, edge_weight=None, directed=True)

    cache.save(fingerprint="corrupt_key", graph=graph, manifest={"n_nodes": 10, "directed": True})

    shard_path = cache.entry_dir("corrupt_key") / "edges_0000.npz"
    np.savez_compressed(shard_path, wrong_key=np.zeros(1))

    with pytest.raises(GraphCacheError, match="Shard missing edge_index"):
        cache.load("corrupt_key")


def test_views_cache_overwrite_explicit(tmp_path):
    cache = ViewsCache(root=tmp_path / "views_overwrite")
    views = DatasetViews(views={"view_a": np.zeros((5, 2))}, y=np.zeros(5), masks={}, meta={})

    cache.save(fingerprint="fp1", views=views, manifest={})

    (cache.entry_dir("fp1") / "dummy.txt").touch()

    cache.save(fingerprint="fp1", views=views, manifest={}, overwrite=True)

    assert not (cache.entry_dir("fp1") / "dummy.txt").exists()


def test_graph_cache_sharded_with_weights(tmp_path):
    cache = GraphCache(root=tmp_path, edge_shard_size=2)
    edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    edge_weight = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    graph = GraphArtifact(n_nodes=5, edge_index=edge_index, edge_weight=edge_weight)

    cache.save(fingerprint="sharded_w", graph=graph, manifest={})

    g_loaded, _ = cache.load("sharded_w")
    assert np.array_equal(g_loaded.edge_index, edge_index)
    assert np.allclose(g_loaded.edge_weight, edge_weight)


def test_views_cache_no_overwrite(tmp_path):
    vc = ViewsCache(root=tmp_path)
    d = vc.entry_dir("no_overwrite")
    d.mkdir(parents=True)
    (d / "old_file.txt").touch()

    views = DatasetViews(views={"v": np.zeros((5, 2))}, y=np.zeros(5))
    vc.save(fingerprint="no_overwrite", views=views, manifest={}, overwrite=False)

    assert (d / "old_file.txt").exists()


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        pytest.skip(f"sklearn unavailable: {exc}")


def test_knn_edges_sklearn_include_self():
    with patch("modssc.graph.construction.backends.sklearn_backend.optional_import") as mock_import:
        mock_sklearn = MagicMock()
        mock_import.return_value = mock_sklearn

        mock_nn = MagicMock()
        mock_sklearn.NearestNeighbors.return_value = mock_nn

        X = np.array([[0], [1], [2]])
        dist = np.array([[0.0], [0.0], [0.0]])
        idx = np.array([[0], [1], [2]])
        mock_nn.kneighbors.return_value = (dist, idx)

        edge_index, _ = knn_edges_sklearn(X, k=1, metric="euclidean", include_self=True)
        assert edge_index.shape[1] == 3

        edge_index, _ = knn_edges_sklearn(X, k=1, metric="euclidean", include_self=False)
        assert edge_index.shape[1] == 0

        dist = np.array([[1.0]])
        idx = np.array([[1]])
        mock_nn.kneighbors.return_value = (dist, idx)

        X = np.array([[0]])
        mock_nn.kneighbors.return_value = (dist, idx)

        edge_index, _ = knn_edges_sklearn(X, k=1, metric="euclidean", include_self=False)
        assert edge_index.shape[1] == 1
        assert edge_index[0, 0] == 0
        assert edge_index[1, 0] == 1


def test_knn_edges_sklearn_empty_mocked():
    with patch("modssc.graph.construction.backends.sklearn_backend.optional_import") as mock_import:
        mock_import.return_value = MagicMock()
        X = np.zeros((0, 2))
        edge_index, dist = knn_edges_sklearn(X, k=1, metric="euclidean")
        assert edge_index.shape == (2, 0)
        assert dist.shape == (0,)


def test_epsilon_edges_sklearn_mocked():
    with patch("modssc.graph.construction.backends.sklearn_backend.optional_import") as mock_import:
        mock_sklearn = MagicMock()
        mock_nn = MagicMock()
        mock_sklearn.NearestNeighbors.return_value = mock_nn
        mock_nn.radius_neighbors.return_value = (
            [np.array([0.0, 1.0]), np.array([0.0, 1.0])],
            [np.array([0, 1]), np.array([1, 0])],
        )
        mock_import.return_value = mock_sklearn

        X = np.array([[0, 0], [0, 1]], dtype=np.float32)
        edge_index, dist = epsilon_edges_sklearn(
            X, radius=1.0, metric="euclidean", include_self=False
        )
        assert edge_index.shape[1] == 2
        assert dist.shape[0] == 2


def test_epsilon_edges_sklearn_no_neighbors():
    with patch("modssc.graph.construction.backends.sklearn_backend.optional_import") as mock_import:
        mock_sklearn = MagicMock()
        mock_nn = MagicMock()
        mock_sklearn.NearestNeighbors.return_value = mock_nn
        mock_nn.radius_neighbors.return_value = (
            [np.array([]), np.array([])],
            [np.array([], dtype=np.int64), np.array([], dtype=np.int64)],
        )
        mock_import.return_value = mock_sklearn

        X = np.array([[0, 0], [0, 1]], dtype=np.float32)
        edge_index, dist = epsilon_edges_sklearn(
            X, radius=1.0, metric="euclidean", include_self=False
        )
        assert edge_index.shape == (2, 0)
        assert dist.shape == (0,)


def test_epsilon_edges_sklearn_self_masked():
    with patch("modssc.graph.construction.backends.sklearn_backend.optional_import") as mock_import:
        mock_sklearn = MagicMock()
        mock_nn = MagicMock()
        mock_sklearn.NearestNeighbors.return_value = mock_nn
        mock_nn.radius_neighbors.return_value = (
            [np.array([0.0]), np.array([0.0])],
            [np.array([0]), np.array([1])],
        )
        mock_import.return_value = mock_sklearn

        X = np.array([[0, 0], [0, 1]], dtype=np.float32)
        edge_index, dist = epsilon_edges_sklearn(
            X, radius=1.0, metric="euclidean", include_self=False
        )
        assert edge_index.shape[1] == 0
        assert dist.shape[0] == 0


def test_epsilon_edges_sklearn_include_self_mocked():
    with patch("modssc.graph.construction.backends.sklearn_backend.optional_import") as mock_import:
        mock_sklearn = MagicMock()
        mock_nn = MagicMock()
        mock_sklearn.NearestNeighbors.return_value = mock_nn
        mock_nn.radius_neighbors.return_value = (
            [np.array([0.0, 0.1]), np.array([0.0])],
            [np.array([0, 1]), np.array([1])],
        )
        mock_import.return_value = mock_sklearn

        X = np.array([[0, 0], [0, 1]], dtype=np.float32)
        edge_index, dist = epsilon_edges_sklearn(
            X, radius=1.0, metric="euclidean", include_self=True
        )
        assert edge_index.shape == (2, 3)
        assert np.array_equal(edge_index, np.array([[0, 0, 1], [0, 1, 1]]))


def test_epsilon_edges_sklearn_empty_mocked():
    with patch("modssc.graph.construction.backends.sklearn_backend.optional_import") as mock_import:
        mock_import.return_value = MagicMock()
        X = np.zeros((0, 2))
        edge_index, dist = epsilon_edges_sklearn(X, radius=1.0, metric="euclidean")
        assert edge_index.shape == (2, 0)
        assert dist.shape == (0,)


def test_epsilon_edges_sklearn():
    _require_sklearn()
    X_empty = np.zeros((0, 2))
    edge_index, dist = epsilon_edges_sklearn(X_empty, radius=1.0, metric="euclidean")
    assert edge_index.shape == (2, 0)
    assert dist.shape == (0,)

    X = np.array([[0, 0], [0, 0.5], [0, 2]])

    edge_index, dist = epsilon_edges_sklearn(X, radius=1.0, metric="euclidean", include_self=False)

    assert edge_index.shape[1] == 2
    assert np.allclose(dist, [0.5, 0.5])

    edge_index, dist = epsilon_edges_sklearn(X, radius=1.0, metric="euclidean", include_self=True)

    assert edge_index.shape[1] == 5
