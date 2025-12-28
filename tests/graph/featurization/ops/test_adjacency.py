from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.graph.featurization.ops.adjacency import adjacency_from_edge_index


class TestAdjacencyCoverage:
    def test_dense_no_weights(self):
        edge_index = np.array([[0, 1], [1, 0]])
        n_nodes = 2
        adj = adjacency_from_edge_index(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=None, format="dense"
        )
        assert isinstance(adj, np.ndarray)
        assert adj.shape == (2, 2)
        assert adj[0, 1] == 1.0
        assert adj[1, 0] == 1.0
        assert adj[0, 0] == 0.0

    def test_dense_with_weights(self):
        edge_index = np.array([[0, 1], [1, 0]])
        edge_weight = np.array([0.5, 0.8])
        n_nodes = 2
        adj = adjacency_from_edge_index(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, format="dense"
        )
        assert adj[0, 1] == 0.5
        assert adj[1, 0] == 0.8

    def test_dense_empty_edges(self):
        edge_index = np.zeros((2, 0), dtype=np.int64)
        n_nodes = 3
        adj = adjacency_from_edge_index(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=None, format="dense"
        )
        assert np.allclose(adj, np.zeros((3, 3), dtype=np.float32))

    def test_csr_no_weights(self):
        edge_index = np.array([[0, 1], [1, 0]])
        n_nodes = 2

        mock_sparse = MagicMock()
        mock_coo = MagicMock()
        mock_csr = MagicMock()
        mock_sparse.coo_matrix.return_value = mock_coo
        mock_coo.tocsr.return_value = mock_csr

        with patch("modssc.graph.featurization.ops.adjacency.optional_import") as mock_import:
            mock_import.return_value = mock_sparse

            adj = adjacency_from_edge_index(
                n_nodes=n_nodes, edge_index=edge_index, edge_weight=None, format="csr"
            )

            assert adj == mock_csr
            mock_sparse.coo_matrix.assert_called()

            call_args = mock_sparse.coo_matrix.call_args
            data_tuple = call_args[0][0]
            data, (row, col) = data_tuple
            assert np.all(data == 1.0)
            assert np.all(row == edge_index[0])
            assert np.all(col == edge_index[1])

    def test_csr_with_weights(self):
        edge_index = np.array([[0], [1]])
        edge_weight = np.array([0.5])
        n_nodes = 2

        mock_sparse = MagicMock()
        mock_coo = MagicMock()
        mock_sparse.coo_matrix.return_value = mock_coo
        mock_coo.tocsr.return_value = "csr_matrix"

        with patch("modssc.graph.featurization.ops.adjacency.optional_import") as mock_import:
            mock_import.return_value = mock_sparse

            adj = adjacency_from_edge_index(
                n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, format="csr"
            )

            assert adj == "csr_matrix"
            call_args = mock_sparse.coo_matrix.call_args
            data_tuple = call_args[0][0]
            data, _ = data_tuple
            assert data[0] == 0.5

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Unknown format"):
            adjacency_from_edge_index(
                n_nodes=2, edge_index=np.zeros((2, 0)), edge_weight=None, format="invalid"
            )
