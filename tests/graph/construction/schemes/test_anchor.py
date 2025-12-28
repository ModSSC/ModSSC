from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.graph.construction.schemes.anchor import (
    AnchorParams,
    _choose_anchors,
    _derive_seed,
    _knn_query_numpy,
    anchor_edges,
)
from modssc.graph.errors import GraphValidationError


def test_anchor_edges_empty_X():
    X = np.zeros((0, 5))
    params = AnchorParams()
    edge_index, dist = anchor_edges(
        X=X, params=params, seed=42, backend="numpy", k=5, metric="euclidean"
    )
    assert edge_index.shape == (2, 0)
    assert dist.shape == (0,)


def test_anchor_edges_empty_candidate_set():
    X = np.zeros((1, 5))
    params = AnchorParams(n_anchors=1, anchors_k=1)

    edge_index, dist = anchor_edges(
        X=X, params=params, seed=42, backend="numpy", include_self=False, k=5, metric="euclidean"
    )

    assert edge_index.shape == (2, 0)


def test_anchor_edges_validate_x_dim():
    with pytest.raises(GraphValidationError, match="X must be 2D"):
        anchor_edges(
            X=np.zeros((10,), dtype=np.float32),
            k=5,
            metric="cosine",
            backend="numpy",
            seed=0,
            params=AnchorParams(),
        )


def test_anchor_edges_include_self_true():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    params = AnchorParams(n_anchors=2, anchors_k=1)

    with patch("modssc.graph.construction.schemes.anchor._choose_anchors", return_value=X):
        edge_index, dist = anchor_edges(
            X=X,
            k=1,
            metric="cosine",
            backend="numpy",
            seed=0,
            params=params,
            include_self=True,
        )
        assert edge_index.shape[1] == 2
        assert dist.shape[0] == 2


def test_anchor_edges_kk_zero():
    X = np.zeros((5, 5))
    params = AnchorParams(n_anchors=2, anchors_k=1)

    edge_index, dist = anchor_edges(
        X=X,
        params=params,
        seed=42,
        backend="numpy",
        k=0,
        metric="euclidean",
    )
    assert edge_index.shape == (2, 0)


def test_anchor_edges_cosine_metric():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    params = AnchorParams(n_anchors=2, anchors_k=2)

    edge_index, dist = anchor_edges(
        X=X, params=params, seed=42, backend="numpy", metric="cosine", k=1
    )
    assert edge_index.shape[1] > 0


def test_anchor_edges_candidate_limit():
    X = np.random.rand(10, 5)
    params = AnchorParams(n_anchors=1, anchors_k=1, candidate_limit=2)

    edge_index, dist = anchor_edges(
        X=X, params=params, seed=42, backend="numpy", k=5, metric="euclidean"
    )

    degrees = np.bincount(edge_index[0])
    assert np.all(degrees <= 2)


def test_anchor_edges_default_anchors():
    X = np.random.randn(10, 2).astype(np.float32)
    params = AnchorParams(n_anchors=None)

    edge_index, dist = anchor_edges(
        X=X, k=2, metric="cosine", backend="numpy", seed=0, params=params
    )
    assert edge_index.shape[1] > 0
    assert dist.shape[0] > 0


def test_derive_seed_determinism():
    s1 = _derive_seed(42, 100)
    s2 = _derive_seed(42, 100)
    assert s1 == s2
    assert isinstance(s1, int)

    s3 = _derive_seed(43, 100)
    assert s1 != s3


def test_choose_anchors_kmeans():
    X = np.random.randn(10, 5).astype(np.float32)

    with patch("modssc.graph.construction.schemes.anchor.optional_import") as mock_import:
        mock_kmeans_cls = MagicMock()
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.cluster_centers_ = np.zeros((2, 5), dtype=np.float32)
        mock_kmeans_cls.return_value = mock_kmeans_instance

        mock_module = MagicMock()
        mock_module.MiniBatchKMeans = mock_kmeans_cls
        mock_import.return_value = mock_module

        anchors = _choose_anchors(X, n_anchors=2, method="kmeans", seed=42)

        assert anchors.shape == (2, 5)
        mock_kmeans_cls.assert_called_once()
        mock_kmeans_instance.fit.assert_called_once()


def test_choose_anchors_unknown():
    X = np.zeros((5, 2))
    with pytest.raises(GraphValidationError, match="Unknown anchor method"):
        _choose_anchors(X, n_anchors=2, method="invalid_method", seed=42)


def test_knn_query_numpy_chunking():
    query = np.random.randn(5, 3).astype(np.float32)
    ref = np.random.randn(10, 3).astype(np.float32)

    idx, dist = _knn_query_numpy(query, ref, k=2, metric="euclidean", chunk_size=2)

    assert idx.shape == (5, 2)
    assert dist.shape == (5, 2)

    idx_full, dist_full = _knn_query_numpy(query, ref, k=2, metric="euclidean", chunk_size=100)
    np.testing.assert_array_equal(idx, idx_full)
    np.testing.assert_allclose(dist, dist_full, rtol=1e-5)


def test_knn_query_numpy_cosine():
    query = np.array([[1, 0], [0, 1]], dtype=np.float32)
    ref = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)

    idx, dist = _knn_query_numpy(query, ref, k=1, metric="cosine", chunk_size=10)

    assert idx[0, 0] == 0
    assert idx[1, 0] == 1
    assert dist[0, 0] < 1e-6
    assert dist[1, 0] < 1e-6


def test_anchor_edges_integration_kmeans():
    X = np.random.randn(20, 4).astype(np.float32)
    params = AnchorParams(n_anchors=5, method="kmeans")

    with patch("modssc.graph.construction.schemes.anchor.optional_import") as mock_import:
        mock_kmeans_cls = MagicMock()
        mock_kmeans_instance = MagicMock()

        mock_kmeans_instance.cluster_centers_ = np.random.randn(5, 4).astype(np.float32)
        mock_kmeans_cls.return_value = mock_kmeans_instance

        mock_module = MagicMock()
        mock_module.MiniBatchKMeans = mock_kmeans_cls
        mock_import.return_value = mock_module

        edge_index, dist = anchor_edges(
            X=X, params=params, seed=123, backend="numpy", k=3, metric="euclidean"
        )
        assert edge_index.shape[1] > 0
