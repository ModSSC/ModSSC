from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.graph.construction.backends.faiss_backend import (
    FaissParams,
    _as_float32_contiguous,
    knn_edges_faiss,
    knn_search_faiss,
)


def test_as_float32_contiguous_non_contiguous():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    non_contig = arr[:, ::2]
    assert not non_contig.flags["C_CONTIGUOUS"]

    res = _as_float32_contiguous(non_contig)
    assert res.flags["C_CONTIGUOUS"]
    assert np.allclose(res, non_contig)


@pytest.fixture
def mock_faiss():
    with patch("modssc.graph.construction.backends.faiss_backend.optional_import") as mock_import:
        faiss_module = MagicMock()
        faiss_module.METRIC_INNER_PRODUCT = 1
        faiss_module.METRIC_L2 = 2

        faiss_module.IndexFlatIP = MagicMock()
        faiss_module.IndexFlatL2 = MagicMock()
        faiss_module.IndexHNSWFlat = MagicMock()

        mock_import.return_value = faiss_module
        yield faiss_module


def test_knn_search_faiss_exact_cosine(mock_faiss):
    X = np.random.rand(10, 5).astype(np.float32)
    params = FaissParams(exact=True)

    index_mock = mock_faiss.IndexFlatIP.return_value
    index_mock.search.return_value = (
        np.zeros((10, 3), dtype=np.float32),
        np.zeros((10, 3), dtype=np.int64),
    )

    knn_search_faiss(X, X, k=3, metric="cosine", params=params)

    mock_faiss.IndexFlatIP.assert_called_once()
    mock_faiss.IndexFlatL2.assert_not_called()


def test_knn_search_faiss_exact_euclidean(mock_faiss):
    X = np.random.rand(10, 5).astype(np.float32)
    params = FaissParams(exact=True)

    index_mock = mock_faiss.IndexFlatL2.return_value
    index_mock.search.return_value = (
        np.zeros((10, 3), dtype=np.float32),
        np.zeros((10, 3), dtype=np.int64),
    )

    knn_search_faiss(X, X, k=3, metric="euclidean", params=params)

    mock_faiss.IndexFlatL2.assert_called_once()
    mock_faiss.IndexFlatIP.assert_not_called()


def test_knn_search_faiss_remove_diagonal(mock_faiss):
    X = np.array([[1.0]], dtype=np.float32)

    D = np.array([[0.0, 0.1]], dtype=np.float32)
    indices = np.array([[0, 1]], dtype=np.int64)

    index_mock = mock_faiss.IndexHNSWFlat.return_value
    index_mock.search.return_value = (D, indices)

    indices, dists = knn_search_faiss(X, X, k=1, metric="euclidean", include_self=False)

    assert indices.shape == (1, 1)
    assert indices[0, 0] == 1
    assert np.isclose(dists[0, 0], np.sqrt(0.1))


def test_knn_search_faiss_remove_diagonal_empty_selection(mock_faiss):
    X = np.array([[1.0]], dtype=np.float32)

    D = np.array([[0.0, 0.0]], dtype=np.float32)
    indices = np.array([[0, 0]], dtype=np.int64)

    index_mock = mock_faiss.IndexHNSWFlat.return_value
    index_mock.search.return_value = (D, indices)

    indices, dists = knn_search_faiss(X, X, k=1, metric="euclidean", include_self=False)

    assert indices[0, 0] == -1
    assert dists[0, 0] == np.inf


def test_knn_edges_faiss(mock_faiss):
    X = np.random.rand(5, 2).astype(np.float32)

    D = np.zeros((5, 2), dtype=np.float32)
    indices = np.zeros((5, 2), dtype=np.int64)

    indices[:] = np.arange(2)

    index_mock = mock_faiss.IndexHNSWFlat.return_value
    index_mock.search.return_value = (D, indices)

    edge_index, edge_weights = knn_edges_faiss(X, k=2, metric="euclidean")

    assert edge_index.shape[0] == 2

    assert edge_index.shape[1] == 8
    assert len(edge_weights) == 8


def test_knn_search_faiss_different_query_ref(mock_faiss):
    X = np.random.rand(5, 2).astype(np.float32)
    Y = np.random.rand(3, 2).astype(np.float32)

    D = np.zeros((3, 2), dtype=np.float32)
    indices = np.zeros((3, 2), dtype=np.int64)

    index_mock = mock_faiss.IndexHNSWFlat.return_value
    index_mock.search.return_value = (D, indices)

    knn_search_faiss(Y, X, k=2, metric="euclidean")

    args, _ = index_mock.search.call_args
    assert args[1] == 2
