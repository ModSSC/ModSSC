import numpy as np
import pytest

from modssc.graph.artifacts import DatasetViews, GraphArtifact, NodeDataset
from modssc.graph.errors import GraphValidationError


def test_graph_artifact_validation():
    with pytest.raises(GraphValidationError, match="edge_index must have shape"):
        GraphArtifact(n_nodes=10, edge_index=np.zeros((3, 5)))

    with pytest.raises(GraphValidationError, match="outside"):
        GraphArtifact(n_nodes=5, edge_index=np.array([[0], [5]]))

    with pytest.raises(GraphValidationError, match="edge_weight must have shape"):
        GraphArtifact(n_nodes=5, edge_index=np.zeros((2, 3)), edge_weight=np.zeros((4,)))


def test_node_dataset_validation():
    graph = GraphArtifact(n_nodes=5, edge_index=np.zeros((2, 0)))

    with pytest.raises(GraphValidationError, match="X must have shape"):
        NodeDataset(X=np.zeros((4, 2)), y=np.zeros(5), graph=graph)

    with pytest.raises(GraphValidationError, match="y must have the same first dimension"):
        NodeDataset(X=np.zeros((5, 2)), y=np.zeros(4), graph=graph)

    with pytest.raises(GraphValidationError, match="Mask 'train' must have shape"):
        NodeDataset(X=np.zeros((5, 2)), y=np.zeros(5), graph=graph, masks={"train": np.zeros(4)})


def test_dataset_views_validation():
    with pytest.raises(GraphValidationError, match="views cannot be empty"):
        DatasetViews(views={}, y=np.zeros(5))

    with pytest.raises(GraphValidationError, match="must be 2D"):
        DatasetViews(views={"view_a": np.zeros((5,))}, y=np.zeros(5))

    with pytest.raises(GraphValidationError, match="same number of samples"):
        DatasetViews(views={"view_a": np.zeros((5, 2)), "view_b": np.zeros((4, 2))}, y=np.zeros(5))

    with pytest.raises(GraphValidationError, match="y must have the same first dimension"):
        DatasetViews(views={"view_a": np.zeros((5, 2))}, y=np.zeros(4))

    with pytest.raises(GraphValidationError, match="Mask 'train' must have shape"):
        DatasetViews(
            views={"view_a": np.zeros((5, 2))}, y=np.zeros(5), masks={"train": np.zeros(4)}
        )


def test_artifacts_validation_extended():
    class NoShape:
        pass

    n_nodes = 5
    edge_index = np.array([[0, 1], [1, 2]])
    graph = GraphArtifact(n_nodes=n_nodes, edge_index=edge_index)

    with pytest.raises(GraphValidationError, match="X must expose a shape attribute"):
        NodeDataset(X=NoShape(), y=np.zeros(n_nodes), graph=graph)

    with pytest.raises(GraphValidationError, match="y must have shape"):
        NodeDataset(X=np.zeros((n_nodes, 2)), y=np.zeros((n_nodes, 2, 2)), graph=graph)

    with pytest.raises(GraphValidationError, match="must expose a shape attribute"):
        DatasetViews(views={"view_a": NoShape()}, y=np.zeros(n_nodes))

    with pytest.raises(
        GraphValidationError, match="All views must have the same number of samples"
    ):
        DatasetViews(
            views={"view_a": np.zeros((n_nodes, 2)), "view_b": np.zeros((n_nodes + 1, 2))},
            y=np.zeros(n_nodes),
        )
