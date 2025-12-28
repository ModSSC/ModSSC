from unittest.mock import MagicMock, patch

import numpy as np

from modssc.graph.adapters.pyg import to_pyg_data
from modssc.graph.artifacts import GraphArtifact, NodeDataset


def test_to_pyg_data():
    mock_torch = MagicMock()
    mock_pyg = MagicMock()

    def as_tensor(data, dtype=None):
        return MagicMock(data=data, dtype=dtype)

    mock_torch.as_tensor.side_effect = as_tensor

    mock_data_cls = MagicMock()
    mock_pyg.Data = mock_data_cls

    with patch("modssc.graph.adapters.pyg.optional_import") as mock_import:

        def import_side_effect(name, extra=None):
            if name == "torch":
                return mock_torch
            if name == "torch_geometric.data":
                return mock_pyg
            raise ImportError(name)

        mock_import.side_effect = import_side_effect

        n_nodes = 5
        edge_index = np.array([[0, 1], [1, 2]])
        edge_weight = np.array([0.5, 0.8])
        X = np.random.randn(n_nodes, 3)
        y = np.random.randint(0, 2, size=n_nodes)
        masks = {"train": np.zeros(n_nodes, dtype=bool)}

        graph = GraphArtifact(n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight)
        dataset = NodeDataset(X=X, y=y, graph=graph, masks=masks)

        data = to_pyg_data(dataset)

        mock_data_cls.assert_called_once()
        _, kwargs = mock_data_cls.call_args
        assert kwargs["x"].data is X
        assert np.array_equal(kwargs["edge_index"].data, edge_index)
        assert kwargs["y"].data is y

        assert np.allclose(data.edge_weight.data, edge_weight)
        assert data.train_mask.data is masks["train"]


def test_to_pyg_data_no_weights():
    mock_torch = MagicMock()
    mock_pyg = MagicMock()

    def as_tensor(x, dtype=None):
        return x

    mock_torch.as_tensor = as_tensor

    class MockData:
        def __init__(self, x, edge_index, y):
            self.x = x
            self.edge_index = edge_index
            self.y = y
            self.edge_weight = None

    mock_pyg.Data = MockData

    with patch("modssc.graph.adapters.pyg.optional_import") as mock_import:

        def side_effect(name, extra=None):
            if name == "torch":
                return mock_torch
            if name == "torch_geometric.data":
                return mock_pyg
            raise ImportError(name)

        mock_import.side_effect = side_effect

        graph = GraphArtifact(
            n_nodes=3,
            edge_index=np.array([[0, 1], [1, 2]]),
            edge_weight=None,
            directed=True,
        )
        dataset = NodeDataset(
            X=np.zeros((3, 2)),
            y=np.zeros(3),
            masks={"train": np.zeros(3, dtype=bool)},
            graph=graph,
        )

        data = to_pyg_data(dataset)

        assert data.edge_weight is None
        assert hasattr(data, "train_mask")
