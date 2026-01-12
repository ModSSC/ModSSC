from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.graph.featurization.views.struct import (
    StructParams,
    _build_neighbors,
    _cooccurrence_counts,
    _derive_seed,
    _iter_random_walks,
    _ppmi_matrix_dense,
    _random_walks,
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


class TestStructEmbeddings:
    def test_derive_seed(self):
        s1 = _derive_seed(42, 1)
        s2 = _derive_seed(42, 1)
        assert isinstance(s1, int)
        assert s1 == s2
        assert s1 != _derive_seed(42, 2)

    def test_build_neighbors(self):
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        n_nodes = 3
        neigh_arr, neigh_set = _build_neighbors(edge_index, n_nodes=n_nodes)

        assert len(neigh_arr) == 3
        assert len(neigh_set) == 3

        assert 1 in neigh_set[0]

        assert 2 in neigh_set[1]

        assert 0 in neigh_set[2]

        edge_index_bad = np.array([[0, 5], [1, 0]])
        neigh_arr_bad, _ = _build_neighbors(edge_index_bad, n_nodes=3)
        assert len(neigh_arr_bad[0]) == 1

    def test_random_walks_deepwalk(self):
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        n_nodes = 3
        neigh_arr, neigh_set = _build_neighbors(edge_index, n_nodes=n_nodes)

        params = StructParams(method="deepwalk", walk_length=5, num_walks_per_node=2)

        walks = _random_walks(neighbors=neigh_arr, neighbor_sets=neigh_set, params=params, seed=42)

        assert len(walks) == n_nodes * 2
        for walk in walks:
            assert len(walk) == 5

            for i in range(len(walk) - 1):
                u, v = walk[i], walk[i + 1]
                if u != v:
                    assert v in neigh_set[u]

    def test_random_walks_node2vec(self):
        edge_index = np.array([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        n_nodes = 3
        neigh_arr, neigh_set = _build_neighbors(edge_index, n_nodes=n_nodes)

        params = StructParams(method="node2vec", p=0.5, q=2.0, walk_length=5, num_walks_per_node=2)

        walks = _random_walks(neighbors=neigh_arr, neighbor_sets=neigh_set, params=params, seed=42)
        assert len(walks) == n_nodes * 2

    def test_random_walks_node2vec_zero_weights(self, monkeypatch):
        seen_p = []

        class DummyRng:
            def choice(self, a, p=None):
                seen_p.append(p)
                return int(a[0])

        monkeypatch.setattr(np.random, "default_rng", lambda *_args, **_kwargs: DummyRng())

        params = StructParams(method="node2vec", p=1.0, q=-1.0, walk_length=3, num_walks_per_node=1)
        neighbors = [np.array([1]), np.array([0, 2]), np.array([1])]
        neighbor_sets = [set(), set(), set()]

        walk = next(
            _iter_random_walks(
                neighbors=neighbors, neighbor_sets=neighbor_sets, params=params, seed=42
            )
        )

        assert walk.shape == (3,)
        assert seen_p
        assert all(p is None for p in seen_p)

    def test_random_walks_isolated_node(self):
        edge_index = np.array([[], []], dtype=np.int64)
        n_nodes = 1
        neigh_arr, neigh_set = _build_neighbors(edge_index, n_nodes=n_nodes)

        params = StructParams(walk_length=3, num_walks_per_node=1)
        walks = _random_walks(neighbors=neigh_arr, neighbor_sets=neigh_set, params=params, seed=42)
        assert len(walks) == 1
        assert np.all(walks[0] == 0)

    def test_cooccurrence_counts(self):
        walks = [np.array([0, 1, 2])]
        counts, row_sum, col_sum, total = _cooccurrence_counts(walks, window_size=1)

        assert counts[(0, 1)] == 1
        assert counts[(1, 0)] == 1
        assert counts[(1, 2)] == 1
        assert counts[(2, 1)] == 1
        assert total == 4

        assert row_sum[1] == 2
        assert col_sum[1] == 2

    def test_cooccurrence_counts_nonempty(self):
        walks = [np.array([0, 1, 2]), np.array([1, 2, 3])]
        counts, row_sum, col_sum, total = _cooccurrence_counts(walks, window_size=2)

        assert total > 0
        assert len(counts) > 0
        assert len(row_sum) > 0
        assert len(col_sum) > 0

    def test_cooccurrence_counts_no_pairs(self):
        walks = [np.array([0, 5], dtype=np.int64)]
        counts, row_sum, col_sum, total = _cooccurrence_counts(walks, window_size=1, n_nodes=2)
        assert counts == {}
        assert total == 0
        assert row_sum.shape == (2,)
        assert col_sum.shape == (2,)

    def test_ppmi_matrix_dense(self):
        n_nodes = 2
        counts = {(0, 1): 10}
        row_sum = np.array([10, 0])
        col_sum = np.array([0, 10])
        total = 100

        M = _ppmi_matrix_dense(
            n_nodes=n_nodes, counts=counts, row_sum=row_sum, col_sum=col_sum, total=total
        )

        expected = np.log(10.0)
        assert np.isclose(M[0, 1], expected)
        assert M[1, 0] == 0

    def test_ppmi_matrix_dense_zeros(self):
        counts = {}
        row_sum = np.zeros(5)
        col_sum = np.zeros(5)

        mat = _ppmi_matrix_dense(
            counts=counts, n_nodes=5, row_sum=row_sum, col_sum=col_sum, total=0
        )
        assert np.all(mat == 0)

    def test_ppmi_matrix_dense_no_counts(self):
        mat = _ppmi_matrix_dense(
            counts={}, n_nodes=3, row_sum=np.zeros(3), col_sum=np.zeros(3), total=1
        )
        assert np.all(mat == 0)

    def test_ppmi_dense_skips_missing_row_col(self):
        counts = {(0, 0): 1}
        row_sum = np.asarray([], dtype=np.float64)
        col_sum = np.asarray([], dtype=np.float64)

        M = _ppmi_matrix_dense(n_nodes=2, counts=counts, row_sum=row_sum, col_sum=col_sum, total=1)
        assert np.allclose(M, 0.0)

    def test_ppmi_dense_negative_pmi(self):
        counts = {(0, 1): 1}
        row_sum = np.array([10.0, 10.0], dtype=np.float64)
        col_sum = np.array([10.0, 10.0], dtype=np.float64)

        M = _ppmi_matrix_dense(n_nodes=2, counts=counts, row_sum=row_sum, col_sum=col_sum, total=1)
        assert np.allclose(M, 0.0)

    def test_struct_embeddings_dense_integration(self):
        edge_index = np.array([[0, 1], [1, 0]])
        n_nodes = 2
        params = StructParams(dim=2, max_dense_nodes=10)

        emb = struct_embeddings(edge_index=edge_index, n_nodes=n_nodes, params=params, seed=42)

        assert emb.shape == (2, 2)
        assert not np.all(emb == 0)

    def test_struct_embeddings_sparse_integration(self):
        edge_index = np.array([[0, 1], [1, 0]])
        n_nodes = 2
        params = StructParams(dim=2, max_dense_nodes=0)

        mock_sparse = MagicMock()
        mock_csr = MagicMock()
        mock_sparse.csr_matrix.return_value = mock_csr

        mock_sklearn = MagicMock()
        mock_svd = MagicMock()
        mock_sklearn.TruncatedSVD.return_value = mock_svd
        mock_svd.fit_transform.return_value = np.ones((2, 2), dtype=np.float32)

        with patch("modssc.graph.featurization.views.struct.optional_import") as mock_import:

            def side_effect(name, extra=None):
                if name == "scipy.sparse":
                    return mock_sparse
                if name == "sklearn.decomposition":
                    return mock_sklearn
                return MagicMock()

            mock_import.side_effect = side_effect

            emb = struct_embeddings(edge_index=edge_index, n_nodes=n_nodes, params=params, seed=42)

            assert emb.shape == (2, 2)
            assert np.all(emb == 1.0)
            mock_sparse.csr_matrix.assert_called()
            mock_sklearn.TruncatedSVD.assert_called()

    def test_struct_embeddings_inconsistent_counts_coverage(self):
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

    def test_struct_embeddings_sparse_counts_path(self):
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

            emb = struct_embeddings(
                edge_index=edge_index, n_nodes=n_nodes, params=params, seed=seed
            )

            assert emb.shape == (2, 2)

    def test_struct_embeddings_sparse_empty_counts(self, monkeypatch):
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

    def test_struct_embeddings_sparse_missing_row_col_masks(self):
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

            emb = struct_embeddings(
                edge_index=np.array([[0], [1]]), n_nodes=2, params=params, seed=42
            )

            assert emb.shape == (2, 2)
            mock_scipy.csr_matrix.assert_called_once()
            data, (rows, cols) = mock_scipy.csr_matrix.call_args[0][0]
            assert len(rows) == 0
            assert len(cols) == 0

    def test_struct_embeddings_empty_graph(self):
        emb = struct_embeddings(
            edge_index=np.zeros((2, 0)), n_nodes=0, params=StructParams(), seed=42
        )
        assert emb.shape == (0, 64)

    def test_struct_embeddings_dim_truncation(self):
        edge_index = np.array([[0, 1], [1, 0]])
        n_nodes = 2
        params = StructParams(dim=10, max_dense_nodes=100)

        emb = struct_embeddings(edge_index=edge_index, n_nodes=n_nodes, params=params, seed=42)

        assert emb.shape == (2, 10)

        assert np.all(emb[:, 2:] == 0)

    def test_node2vec_weights_logic(self):
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        n_nodes = 3
        neigh_arr, neigh_set = _build_neighbors(edge_index, n_nodes=n_nodes)

        params = StructParams(
            method="node2vec",
            p=0.1,
            q=10.0,
            walk_length=3,
            num_walks_per_node=1,
        )

        walks = _random_walks(neighbors=neigh_arr, neighbor_sets=neigh_set, params=params, seed=42)
        assert len(walks) == 3

    def test_cooccurrence_empty(self):
        counts, _, _, total = _cooccurrence_counts([], window_size=5)
        assert total == 0
        assert counts == {}

    def test_ppmi_zeros(self):
        M = _ppmi_matrix_dense(
            n_nodes=2,
            counts={(0, 1): 1},
            row_sum=np.array([0, 0]),
            col_sum=np.array([0, 0]),
            total=10,
        )
        assert np.all(M == 0)

    def test_num_walks_zero(self):
        edge_index = np.array([[0, 1], [1, 0]])
        n_nodes = 2
        neigh_arr, neigh_set = _build_neighbors(edge_index, n_nodes=n_nodes)
        params = StructParams(num_walks_per_node=0)
        walks = _random_walks(neighbors=neigh_arr, neighbor_sets=neigh_set, params=params, seed=42)
        assert len(walks) == 0

    def test_iter_random_walks_requires_neighbor_sets(self):
        params = StructParams(method="node2vec", num_walks_per_node=1, walk_length=2)
        with pytest.raises(ValueError, match="neighbor_sets required"):
            list(
                _iter_random_walks(
                    neighbors=[np.array([1])], neighbor_sets=None, params=params, seed=42
                )
            )

    def test_ppmi_out_of_bounds(self):
        M = _ppmi_matrix_dense(
            n_nodes=2,
            counts={(0, 5): 1, (5, 0): 1},
            row_sum=np.array([1, 0, 0, 0, 0, 1]),
            col_sum=np.array([1, 0, 0, 0, 0, 1]),
            total=10,
        )

        assert np.all(M == 0)

    def test_sparse_out_of_bounds(self):
        edge_index = np.array([[0, 1], [1, 0]])
        n_nodes = 2
        params = StructParams(dim=2, max_dense_nodes=0)

        with patch("modssc.graph.featurization.views.struct._cooccurrence_counts") as mock_counts:
            mock_counts.return_value = (
                {(0, 5): 1},
                np.array([1, 0, 0, 0, 0, 1]),
                np.array([1, 0, 0, 0, 0, 1]),
                10,
            )

            mock_sparse = MagicMock()
            mock_csr = MagicMock()
            mock_sparse.csr_matrix.return_value = mock_csr

            mock_sklearn = MagicMock()
            mock_svd = MagicMock()
            mock_sklearn.TruncatedSVD.return_value = mock_svd
            mock_svd.fit_transform.return_value = np.zeros((2, 2))

            with patch("modssc.graph.featurization.views.struct.optional_import") as mock_import:

                def side_effect(name, extra=None):
                    if name == "scipy.sparse":
                        return mock_sparse
                    if name == "sklearn.decomposition":
                        return mock_sklearn
                    return MagicMock()

                mock_import.side_effect = side_effect

                struct_embeddings(edge_index=edge_index, n_nodes=n_nodes, params=params, seed=42)

                call_args = mock_sparse.csr_matrix.call_args
                data, (rows, cols) = call_args[0][0]
                assert len(rows) == 0
