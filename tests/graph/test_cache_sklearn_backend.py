import contextlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.graph.artifacts import DatasetViews, GraphArtifact
from modssc.graph.cache import (
    GraphCache,
    GraphCacheError,
    ViewsCache,
    _safe_read_json,
    _safe_write_json,
)
from modssc.graph.construction.backends.sklearn_backend import (
    epsilon_edges_sklearn,
    knn_edges_sklearn,
)


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        pytest.skip(f"sklearn unavailable: {exc}")


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


def test_graph_cache_existing_root_methods(tmp_path):
    cache = GraphCache(root=tmp_path)

    (tmp_path / "fp1").mkdir()
    (tmp_path / "fp2").mkdir()
    (tmp_path / "file.txt").touch()

    entries = cache.list()
    assert "fp1" in entries
    assert "fp2" in entries
    assert "file.txt" not in entries

    n = cache.purge()
    assert n == 2
    assert not (tmp_path / "fp1").exists()
    assert (tmp_path / "file.txt").exists()


def test_views_cache_missing_root_methods(tmp_path):
    root = tmp_path / "non_existent"
    cache = ViewsCache(root=root)

    assert cache.list() == []


def test_views_cache_existing_root_methods(tmp_path):
    cache = ViewsCache(root=tmp_path)
    (tmp_path / "fp1").mkdir()

    assert cache.list() == ["fp1"]


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


def test_views_cache(tmp_path):
    cache = ViewsCache(root=tmp_path)
    assert ViewsCache.default().root.name == "graph_views"

    views = DatasetViews(
        views={"view_a": np.array([[1, 2], [3, 4]])},
        y=np.array([0, 1]),
        masks={"train": np.array([True, False])},
        meta={"foo": "bar"},
    )
    manifest = {"dataset": "test"}

    d = cache.save(fingerprint="fp1", views=views, manifest=manifest)
    assert d.exists()
    assert (d / "views.npz").exists()
    assert (d / "manifest.json").exists()

    assert cache.exists("fp1")
    assert not cache.exists("fp2")

    assert cache.list() == ["fp1"]

    loaded_views, loaded_manifest = cache.load("fp1", y=views.y, masks=views.masks)
    assert np.allclose(loaded_views.views["view_a"], views.views["view_a"])
    assert loaded_manifest["dataset"] == "test"
    assert loaded_manifest["meta"]["foo"] == "bar"

    with pytest.raises(GraphCacheError, match="Missing cached views manifest"):
        cache.load("fp2", y=views.y, masks=views.masks)

    (d / "views.npz").unlink()
    with pytest.raises(GraphCacheError, match="Missing cached views.npz"):
        cache.load("fp1", y=views.y, masks=views.masks)


def test_safe_read_json_errors(tmp_path):
    p = tmp_path / "bad.json"

    p.write_text("{", encoding="utf-8")
    with pytest.raises(GraphCacheError, match="Invalid json payload"):
        _safe_read_json(p)

    p.write_text("[]", encoding="utf-8")
    with pytest.raises(GraphCacheError, match="Invalid json payload"):
        _safe_read_json(p)


def test_knn_edges_sklearn_empty():
    _require_sklearn()
    X = np.zeros((0, 2))
    edge_index, dist = knn_edges_sklearn(X, k=1, metric="euclidean")
    assert edge_index.shape == (2, 0)
    assert dist.shape == (0,)


def test_graph_cache_load_single(tmp_path):
    cache = GraphCache(root=tmp_path)
    graph = GraphArtifact(
        n_nodes=3,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_weight=np.array([0.5, 0.8], dtype=np.float32),
        directed=True,
        meta={"foo": "bar"},
    )
    manifest = {"dataset": "test"}

    cache.save(fingerprint="fp1", graph=graph, manifest=manifest)

    loaded_graph, loaded_manifest = cache.load("fp1")
    assert loaded_graph.n_nodes == 3
    assert np.allclose(loaded_graph.edge_index, graph.edge_index)
    assert np.allclose(loaded_graph.edge_weight, graph.edge_weight)
    assert loaded_graph.meta == graph.meta
    assert loaded_manifest["dataset"] == "test"


def test_graph_cache_sharded(tmp_path):
    cache = GraphCache(root=tmp_path, edge_shard_size=1)
    graph = GraphArtifact(
        n_nodes=3,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_weight=np.array([0.5, 0.8], dtype=np.float32),
        directed=True,
        meta={},
    )
    manifest = {}

    cache.save(fingerprint="fp_sharded", graph=graph, manifest=manifest)

    d = cache.entry_dir("fp_sharded")
    assert (d / "edges_0000.npz").exists()
    assert (d / "edges_0001.npz").exists()

    loaded_graph, _ = cache.load("fp_sharded")
    assert np.allclose(loaded_graph.edge_index, graph.edge_index)
    assert np.allclose(loaded_graph.edge_weight, graph.edge_weight)


def test_graph_cache_load_errors(tmp_path):
    cache = GraphCache(root=tmp_path)
    (tmp_path / "fp_bad").mkdir()

    with pytest.raises(GraphCacheError, match="Missing cached graph manifest"):
        cache.load("fp_bad")

    (tmp_path / "fp_bad" / "manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(GraphCacheError, match="Invalid json payload"):
        cache.load("fp_bad")

    graph = GraphArtifact(
        n_nodes=1,
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_weight=None,
        directed=False,
        meta={},
    )
    cache.save(fingerprint="fp_single", graph=graph, manifest={})
    (tmp_path / "fp_single" / "edge_index.npy").unlink()
    with pytest.raises(GraphCacheError, match="Missing cached edge_index.npy"):
        cache.load("fp_single")

    cache = GraphCache(root=tmp_path, edge_shard_size=1)

    graph = GraphArtifact(
        n_nodes=3,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_weight=None,
        directed=False,
        meta={},
    )
    cache.save(fingerprint="fp_sharded_bad", graph=graph, manifest={})
    (tmp_path / "fp_sharded_bad" / "edges_0000.npz").unlink()
    with pytest.raises(GraphCacheError, match="Missing edge shard"):
        cache.load("fp_sharded_bad")


def test_safe_write_json_errors(tmp_path):
    p = tmp_path / "fail.json"
    with patch("builtins.open", side_effect=OSError("Fail")):
        with contextlib.suppress(OSError):
            _safe_write_json(p, {})

        pass


def test_graph_cache_corrupted_files(tmp_path):
    cache = GraphCache(root=tmp_path)
    graph = GraphArtifact(
        n_nodes=3,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_weight=np.array([1.0, 1.0], dtype=np.float32),
        directed=False,
        meta={},
    )
    cache.save(fingerprint="fp_corrupt", graph=graph, manifest={})

    (tmp_path / "fp_corrupt" / "edge_weight.npy").write_text("bad", encoding="utf-8")
    with pytest.raises(GraphCacheError, match="Corrupted cached edge_weight.npy"):
        cache.load("fp_corrupt")


def test_graph_cache_sharded_corrupt(tmp_path):
    cache = GraphCache(root=tmp_path, edge_shard_size=1)
    graph = GraphArtifact(
        n_nodes=3,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_weight=None,
        directed=False,
        meta={},
    )
    cache.save(fingerprint="fp_sharded_corrupt", graph=graph, manifest={})

    np.savez(tmp_path / "fp_sharded_corrupt" / "edges_0000.npz", foo=np.array([1]))

    with pytest.raises(GraphCacheError, match="Shard missing edge_index"):
        cache.load("fp_sharded_corrupt")


def test_graph_cache_default():
    cache = GraphCache.default()
    assert cache.root.name == "graphs"


def test_graph_cache_clear_entry_dir_edge_cases(tmp_path):
    cache = GraphCache(root=tmp_path)
    d = tmp_path / "fp_clear"

    cache._clear_entry_dir(d)

    d.mkdir()
    (d / "subdir").mkdir()
    (d / "file.txt").touch()

    with patch("pathlib.Path.unlink", side_effect=FileNotFoundError):
        cache._clear_entry_dir(d)

    assert not (d / "subdir").exists()
    assert (d / "file.txt").exists()


def test_graph_cache_load_no_weights(tmp_path):
    cache = GraphCache(root=tmp_path)
    graph = GraphArtifact(
        n_nodes=3,
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_weight=None,
        directed=False,
        meta={},
    )
    cache.save(fingerprint="fp_no_weights", graph=graph, manifest={})

    loaded_graph, _ = cache.load("fp_no_weights")
    assert loaded_graph.edge_weight is None


def test_safe_write_json_cleanup_error(tmp_path):
    p = tmp_path / "cleanup_fail.json"

    with (
        patch("os.remove", side_effect=OSError("Fail")),
        patch("os.replace", side_effect=OSError("Replace Fail")),
        contextlib.suppress(OSError),
    ):
        _safe_write_json(p, {})


def test_graph_cache_exists_false(tmp_path):
    cache = GraphCache(root=tmp_path)
    assert not cache.exists("fp_missing")


def test_graph_cache_sharded_no_weights(tmp_path):
    cache = GraphCache(root=tmp_path, edge_shard_size=1)
    graph = GraphArtifact(
        n_nodes=3,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_weight=None,
        directed=False,
        meta={},
    )
    cache.save(fingerprint="fp_sharded_no_w", graph=graph, manifest={})

    loaded_graph, _ = cache.load("fp_sharded_no_w")
    assert loaded_graph.edge_weight is None


def test_views_cache_overwrite(tmp_path):
    cache = ViewsCache(root=tmp_path)
    views = DatasetViews(views={"view_a": np.array([[1]])}, y=np.array([0]), masks={}, meta={})

    cache.save(fingerprint="fp_views", views=views, manifest={})

    (tmp_path / "fp_views" / "dummy.txt").touch()

    cache.save(fingerprint="fp_views", views=views, manifest={})

    assert not (tmp_path / "fp_views" / "dummy.txt").exists()
