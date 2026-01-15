from unittest.mock import patch

import numpy as np
import pytest

from modssc.graph.construction.backends.numpy_backend import (
    _save_npz_atomic,
    epsilon_edges_numpy,
    knn_edges_numpy,
)


def test_save_npz_atomic_cleanup_error(tmp_path):
    p = tmp_path / "test.npz"

    with (
        patch("os.remove", side_effect=OSError("fail")),
        patch("os.path.exists", return_value=True),
    ):
        _save_npz_atomic(p, x=np.array([1]))


def test_knn_numpy_empty():
    X = np.zeros((0, 10))
    edge_index, dist = knn_edges_numpy(X, k=5, metric="cosine")
    assert edge_index.shape == (2, 0)
    assert dist.shape == (0,)


def test_knn_numpy_euclidean():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    edge_index, dist = knn_edges_numpy(X, k=1, metric="euclidean")
    assert edge_index.shape[1] == 3

    assert np.allclose(dist, 1.0)


def test_knn_numpy_include_self():
    X = np.array([[0.0], [1.0]])

    edge_index, dist = knn_edges_numpy(X, k=1, metric="euclidean", include_self=True)
    assert edge_index.shape[1] == 2

    assert np.all(edge_index[0] == edge_index[1])
    assert np.allclose(dist, 0.0)


def test_knn_numpy_k_eff_zero():
    X = np.array([[0.0]])

    edge_index, dist = knn_edges_numpy(X, k=1, metric="euclidean", include_self=False)
    assert edge_index.shape == (2, 0)


def test_knn_numpy_work_dir(tmp_path):
    X = np.array([[0.0], [1.0], [2.0]])
    wd = tmp_path / "work"
    edge_index, dist = knn_edges_numpy(X, k=1, metric="euclidean", work_dir=wd, chunk_size=1)

    assert len(list(wd.glob("*.npz"))) > 0
    assert edge_index.shape[1] == 3

    edge_index2, dist2 = knn_edges_numpy(
        X, k=1, metric="euclidean", work_dir=wd, chunk_size=1, resume=True
    )
    assert np.array_equal(edge_index, edge_index2)


def test_knn_numpy_cosine():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    edge_index, dist = knn_edges_numpy(X, k=1, metric="cosine")
    assert edge_index.shape[1] == 3
    assert np.allclose(dist[0], 1.0 - 0.70710678)


def test_epsilon_numpy_empty():
    X = np.zeros((0, 10))
    edge_index, dist = epsilon_edges_numpy(X, radius=0.5, metric="cosine")
    assert edge_index.shape == (2, 0)


def test_epsilon_numpy_invalid_radius():
    X = np.array([[0.0]])
    with pytest.raises(ValueError, match="radius must be > 0"):
        epsilon_edges_numpy(X, radius=0.0, metric="cosine")


def test_epsilon_numpy_euclidean():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    edge_index, dist = epsilon_edges_numpy(X, radius=1.1, metric="euclidean")

    assert edge_index.shape[1] == 4


def test_epsilon_numpy_include_self():
    X = np.array([[0.0]])
    edge_index, dist = epsilon_edges_numpy(X, radius=0.1, metric="euclidean", include_self=True)
    assert edge_index.shape[1] == 1
    assert edge_index[0, 0] == 0
    assert edge_index[1, 0] == 0


def test_epsilon_numpy_no_edges():
    X = np.array([[0.0], [10.0]])
    edge_index, dist = epsilon_edges_numpy(X, radius=1.0, metric="euclidean")
    assert edge_index.shape[1] == 0


def test_epsilon_numpy_work_dir(tmp_path):
    X = np.array([[0.0], [0.1], [0.2]])
    wd = tmp_path / "work_eps"

    edge_index, dist = epsilon_edges_numpy(
        X, radius=0.15, metric="euclidean", work_dir=wd, chunk_size=1
    )

    assert len(list(wd.glob("*.npz"))) > 0

    edge_index2, dist2 = epsilon_edges_numpy(
        X, radius=0.15, metric="euclidean", work_dir=wd, chunk_size=1, resume=True
    )
    assert np.array_equal(edge_index, edge_index2)


def test_epsilon_numpy_cosine():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    edge_index, dist = epsilon_edges_numpy(X, radius=0.3, metric="cosine")

    assert edge_index.shape[1] == 4
    assert np.all(dist < 0.3)


def test_knn_numpy_chunked_no_workdir():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])

    edge_index, dist = knn_edges_numpy(X, k=1, metric="euclidean", chunk_size=2)
    assert edge_index.shape[1] == 4


def test_knn_numpy_nan():
    X = np.array([[np.nan, np.nan], [1.0, 1.0]])

    edge_index, dist = knn_edges_numpy(X, k=1, metric="euclidean", include_self=False)

    assert edge_index.shape[1] == 0


def test_knn_numpy_resume(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 4)).astype(np.float32)

    ei1, d1 = knn_edges_numpy(
        X,
        k=4,
        metric="cosine",
        include_self=False,
        chunk_size=7,
        work_dir=tmp_path,
        resume=False,
    )
    assert any(p.name.startswith("knn_") for p in tmp_path.iterdir())

    ei2, d2 = knn_edges_numpy(
        X,
        k=4,
        metric="cosine",
        include_self=False,
        chunk_size=7,
        work_dir=tmp_path,
        resume=True,
    )
    np.testing.assert_array_equal(ei2, ei1)
    np.testing.assert_allclose(d2, d1)


def test_epsilon_numpy_resume(tmp_path) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3)).astype(np.float32)

    ei1, d1 = epsilon_edges_numpy(
        X,
        radius=1.2,
        metric="euclidean",
        include_self=False,
        chunk_size=6,
        work_dir=tmp_path,
        resume=False,
    )
    assert any(p.name.startswith("eps_") for p in tmp_path.iterdir())

    ei2, d2 = epsilon_edges_numpy(
        X,
        radius=1.2,
        metric="euclidean",
        include_self=False,
        chunk_size=6,
        work_dir=tmp_path,
        resume=True,
    )
    np.testing.assert_array_equal(ei2, ei1)
    np.testing.assert_allclose(d2, d1)
