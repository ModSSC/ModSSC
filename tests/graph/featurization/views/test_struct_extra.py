from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.graph.featurization.views.struct import (
    StructParams,
    _ppmi_matrix_dense,
    struct_embeddings,
)


def _require_scipy() -> None:
    try:
        import scipy  # noqa: F401
    except Exception as exc:
        pytest.skip(f"scipy unavailable: {exc}")


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        pytest.skip(f"sklearn unavailable: {exc}")


def test_struct_embeddings_inconsistent_counts_coverage():
    with (
        patch("modssc.graph.featurization.views.struct._cooccurrence_counts") as mock_counts,
        patch("modssc.graph.featurization.views.struct.optional_import") as mock_import,
    ):
        counts = {(0, 1): 1}

        row_arr = np.array([0, 0], dtype=np.int64)

        col_arr = np.array([0, 0], dtype=np.int64)
        total = 1

        mock_counts.return_value = (counts, row_arr, col_arr, total)

        mock_scipy = MagicMock()
        mock_sklearn = MagicMock()

        def import_side_effect(name, extra=None):
            if "scipy" in name:
                return mock_scipy
            if "sklearn" in name:
                return mock_sklearn
            return MagicMock()

        mock_import.side_effect = import_side_effect

        mock_svd = MagicMock()
        mock_sklearn.TruncatedSVD.return_value = mock_svd

        mock_svd.fit_transform.return_value = np.zeros((2, 2), dtype=np.float32)

        mock_csr = MagicMock()
        mock_scipy.csr_matrix.return_value = mock_csr

        params = StructParams(
            dim=2, max_dense_nodes=0, window_size=1, num_walks_per_node=1, walk_length=1
        )

        struct_embeddings(edge_index=np.array([[0], [1]]), n_nodes=2, params=params, seed=42)

        mock_scipy.csr_matrix.assert_called_once()

        call_args = mock_scipy.csr_matrix.call_args
        data_tuple = call_args[0][0]
        data_list, (rows_list, cols_list) = data_tuple

        assert len(data_list) == 0
        assert len(rows_list) == 0
        assert len(cols_list) == 0


def test_struct_embeddings_sparse_counts_path():
    _require_scipy()
    _require_sklearn()
    params = MagicMock()
    params.dim = 2
    params.window_size = 2
    params.max_dense_nodes = 0
    params.p = 1.0
    params.q = 1.0

    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    n_nodes = 2
    seed = 42

    with (
        patch("modssc.graph.featurization.views.struct._cooccurrence_counts") as mock_counts,
        patch("modssc.graph.featurization.views.struct._random_walks") as mock_walks,
        patch("modssc.graph.featurization.views.struct._build_neighbors") as mock_neighbors,
    ):
        counts = {(0, 1): 2}
        row_arr = np.array([10, 10])
        col_arr = np.array([10, 10])
        total = 100

        mock_counts.return_value = (counts, row_arr, col_arr, total)
        mock_walks.return_value = []
        mock_neighbors.return_value = (None, None)

        emb = struct_embeddings(edge_index=edge_index, n_nodes=n_nodes, params=params, seed=seed)

        assert emb.shape == (2, 2)


def test_struct_embeddings_sparse_empty_counts(monkeypatch):
    edge_index = np.array([[0, 1], [1, 0]])
    params = StructParams(dim=2, max_dense_nodes=1)

    monkeypatch.setattr(
        "modssc.graph.featurization.views.struct._cooccurrence_counts",
        lambda *a, **k: ({}, np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), 0),
    )

    mock_scipy = MagicMock()
    mock_sklearn = MagicMock()
    mock_svd_cls = MagicMock()
    mock_svd_inst = MagicMock()
    mock_svd_inst.fit_transform.return_value = np.zeros((2, 2), dtype=np.float32)
    mock_svd_cls.return_value = mock_svd_inst
    mock_sklearn.TruncatedSVD = mock_svd_cls

    with patch("modssc.graph.featurization.views.struct.optional_import") as mock_import:
        mock_import.side_effect = (
            lambda name, extra=None: mock_scipy if "scipy" in name else mock_sklearn
        )
        emb = struct_embeddings(edge_index=edge_index, n_nodes=2, params=params, seed=42)
        assert emb.shape == (2, 2)


def test_struct_embeddings_sparse_missing_row_col_masks():
    with (
        patch("modssc.graph.featurization.views.struct._cooccurrence_counts") as mock_counts,
        patch("modssc.graph.featurization.views.struct.optional_import") as mock_import,
    ):
        counts = {(0, 1): 1}
        row_arr = np.asarray([], dtype=np.int64)
        col_arr = np.asarray([], dtype=np.int64)
        total = 1

        mock_counts.return_value = (counts, row_arr, col_arr, total)

        mock_scipy = MagicMock()
        mock_sklearn = MagicMock()

        def import_side_effect(name, extra=None):
            if "scipy" in name:
                return mock_scipy
            if "sklearn" in name:
                return mock_sklearn
            return MagicMock()

        mock_import.side_effect = import_side_effect

        mock_svd = MagicMock()
        mock_sklearn.TruncatedSVD.return_value = mock_svd
        mock_svd.fit_transform.return_value = np.zeros((2, 2), dtype=np.float32)

        mock_csr = MagicMock()
        mock_scipy.csr_matrix.return_value = mock_csr

        params = StructParams(
            dim=2, max_dense_nodes=0, window_size=1, num_walks_per_node=1, walk_length=1
        )

        emb = struct_embeddings(edge_index=np.array([[0], [1]]), n_nodes=2, params=params, seed=42)

        assert emb.shape == (2, 2)
        mock_scipy.csr_matrix.assert_called_once()
        data, (rows, cols) = mock_scipy.csr_matrix.call_args[0][0]
        assert len(rows) == 0
        assert len(cols) == 0


def test_ppmi_dense_skips_missing_row_col():
    counts = {(0, 0): 1}
    row_sum = np.asarray([], dtype=np.float64)
    col_sum = np.asarray([], dtype=np.float64)

    M = _ppmi_matrix_dense(n_nodes=2, counts=counts, row_sum=row_sum, col_sum=col_sum, total=1)
    assert np.allclose(M, 0.0)


def test_ppmi_dense_negative_pmi():
    counts = {(0, 1): 1}
    row_sum = np.array([10.0, 10.0], dtype=np.float64)
    col_sum = np.array([10.0, 10.0], dtype=np.float64)

    M = _ppmi_matrix_dense(n_nodes=2, counts=counts, row_sum=row_sum, col_sum=col_sum, total=1)
    assert np.allclose(M, 0.0)
