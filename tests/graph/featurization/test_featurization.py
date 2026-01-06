from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.graph.artifacts import GraphArtifact, NodeDataset
from modssc.graph.errors import GraphValidationError, OptionalDependencyError
from modssc.graph.featurization.api import graph_to_views
from modssc.graph.featurization.ops.adjacency import adjacency_from_edge_index
from modssc.graph.featurization.views.diffusion import _to_dense, diffusion_view
from modssc.graph.fingerprint import _to_jsonable, fingerprint_array, fingerprint_spec
from modssc.graph.specs import GraphFeaturizerSpec


def _require_scipy() -> None:
    try:
        import scipy  # noqa: F401
    except Exception as exc:
        pytest.skip(f"scipy unavailable: {exc}")


def test_to_dense_sparse_matrix():
    mock_sparse = MagicMock()
    mock_sparse.toarray.return_value = np.array([[1, 0], [0, 1]])
    mock_sparse.shape = (2, 2)

    res = _to_dense(mock_sparse)
    assert isinstance(res, np.ndarray)
    assert np.allclose(res, np.eye(2))


def test_to_dense_too_large():
    mock_sparse = MagicMock()
    mock_sparse.toarray.return_value = np.zeros((1, 1))
    mock_sparse.shape = (5000, 5000)

    with pytest.raises(GraphValidationError, match="too large"):
        _to_dense(mock_sparse, max_elements=100)


def test_diffusion_view_sparse_fallback():
    X = np.zeros((5, 2))
    edge_index = np.array([[0, 1], [1, 0]])

    with patch("modssc.graph.featurization.views.diffusion.adjacency_from_edge_index") as mock_adj:
        mock_adj.side_effect = [OptionalDependencyError(extra="scipy"), np.eye(5)]

        res = diffusion_view(
            X=X, n_nodes=5, edge_index=edge_index, edge_weight=None, steps=10, alpha=0.1
        )
        assert res.shape == (5, 2)

        assert mock_adj.call_count == 2
        assert mock_adj.call_args_list[1].kwargs["format"] == "dense"


def test_diffusion_view_sparse_fallback_too_large():
    X = np.zeros((5000, 2))
    edge_index = np.zeros((2, 0), dtype=int)

    with patch("modssc.graph.featurization.views.diffusion.adjacency_from_edge_index") as mock_adj:
        mock_adj.side_effect = OptionalDependencyError(extra="scipy")

        with pytest.raises(OptionalDependencyError):
            diffusion_view(
                X=X, n_nodes=5000, edge_index=edge_index, edge_weight=None, steps=10, alpha=0.1
            )


def test_to_jsonable_types():
    assert _to_jsonable(np.int64(5)) == 5
    assert _to_jsonable(np.float32(1.5)) == 1.5
    assert _to_jsonable((1, 2)) == [1, 2]


def test_fingerprint_array_sparse():
    mock_sparse = MagicMock()
    mock_sparse.data = np.array([1, 1])
    mock_sparse.indices = np.array([0, 1])
    mock_sparse.indptr = np.array([0, 2])
    mock_sparse.shape = (2, 2)

    mock_sparse.__class__.__module__ = "scipy.sparse"

    fp = fingerprint_array(mock_sparse)
    assert isinstance(fp, str)
    assert len(fp) == 64


def test_fingerprint_array_large():
    arr = np.zeros((2000, 2000))
    fp1 = fingerprint_array(arr)

    arr[-1, -1] = 1
    fp2 = fingerprint_array(arr)

    assert isinstance(fp1, str)
    assert isinstance(fp2, str)


def test_to_dense_numpy_array():
    arr = np.array([[1, 2], [3, 4]])
    res = _to_dense(arr)
    assert res is arr


def test_diffusion_view_invalid_shape():
    X = np.zeros((5, 2))
    edge_index = np.array([[0, 1], [1, 0]])
    with pytest.raises(GraphValidationError, match="shape"):
        diffusion_view(X=X, n_nodes=3, edge_index=edge_index, edge_weight=None, steps=1, alpha=0.1)


def test_diffusion_view_with_weights():
    X = np.zeros((2, 2))
    edge_index = np.array([[0, 1], [1, 0]])
    edge_weight = np.array([0.5, 0.5])

    with patch("modssc.graph.featurization.views.diffusion.adjacency_from_edge_index") as mock_adj:
        mock_adj.return_value = np.eye(2)
        diffusion_view(
            X=X, n_nodes=2, edge_index=edge_index, edge_weight=edge_weight, steps=1, alpha=0.1
        )

        args, kwargs = mock_adj.call_args
        assert kwargs["edge_weight"] is not None


def test_diffusion_view_with_weights_empty_edges():
    X = np.zeros((2, 2))
    edge_index = np.zeros((2, 0), dtype=np.int64)
    edge_weight = np.array([], dtype=np.float32)

    with patch("modssc.graph.featurization.views.diffusion.adjacency_from_edge_index") as mock_adj:
        mock_adj.return_value = np.eye(2)
        res = diffusion_view(
            X=X, n_nodes=2, edge_index=edge_index, edge_weight=edge_weight, steps=1, alpha=0.1
        )

        assert res.shape == (2, 2)
        assert mock_adj.call_args.kwargs["edge_weight"] is not None


def test_diffusion_view_sparse_success():
    X = np.zeros((2, 2))
    edge_index = np.array([[0, 1], [1, 0]])

    with patch("modssc.graph.featurization.views.diffusion.adjacency_from_edge_index") as mock_adj:
        mock_csr = MagicMock()
        mock_csr.__matmul__.return_value = X
        mock_adj.return_value = mock_csr

        diffusion_view(X=X, n_nodes=2, edge_index=edge_index, edge_weight=None, steps=1, alpha=0.1)
        assert mock_adj.call_args.kwargs["format"] == "csr"


@dataclass
class MySpec:
    x: int


def test_to_jsonable_none():
    assert _to_jsonable(None) is None


def test_to_jsonable_dict():
    assert _to_jsonable({"a": 1}) == {"a": 1}


def test_to_jsonable_dataclass():
    assert _to_jsonable(MySpec(1)) == {"x": 1}


def test_fingerprint_spec_dict():
    assert isinstance(fingerprint_spec({"a": 1}), str)


def test_fingerprint_spec_dataclass():
    assert isinstance(fingerprint_spec(MySpec(1)), str)


def test_fingerprint_spec_invalid():
    with pytest.raises(TypeError):
        fingerprint_spec(123)


def test_to_dense_fallback():
    data = [[1, 2], [3, 4]]
    res = _to_dense(data)
    assert isinstance(res, np.ndarray)
    assert np.array_equal(res, np.array(data))


def test_to_jsonable_type_error():
    class NotSerializable:
        pass

    obj = NotSerializable()
    with pytest.raises(TypeError, match="is not JSON-serializable"):
        _to_jsonable(obj)


def test_fingerprint_numpy_small():
    arr = np.array([1, 2, 3])
    fp = fingerprint_array(arr)
    assert isinstance(fp, str)
    assert len(fp) == 64


def test_api_graph_to_views_missing_meta():
    graph = MagicMock(spec=GraphArtifact)
    graph.meta = None
    graph.n_nodes = 10
    graph.edge_index = np.zeros((2, 0), dtype=int)
    graph.edge_weight = None

    dataset = MagicMock(spec=NodeDataset)
    dataset.graph = graph
    dataset.y = np.zeros(10)
    dataset.masks = {}
    dataset.X = np.zeros((10, 5))

    spec = MagicMock(spec=GraphFeaturizerSpec)
    spec.views = ["attr"]
    spec.cache = False
    spec.to_dict.return_value = {}

    views = graph_to_views(dataset, spec=spec)
    assert "attr" in views.views


def test_api_graph_to_views_unknown_view():
    dataset = MagicMock()
    dataset.graph.n_nodes = 10
    dataset.graph.meta = {}

    dataset.graph.edge_index = np.zeros((2, 10), dtype=int)

    spec = MagicMock(spec=GraphFeaturizerSpec)
    spec.views = ["unknown_view"]
    spec.cache = False
    spec.to_dict.return_value = {}

    with pytest.raises(ValueError, match="Unknown view"):
        graph_to_views(dataset, spec=spec)


def test_adjacency_ops():
    _require_scipy()
    edge_index = np.array([[0, 1], [1, 0]])
    edge_weight = np.array([0.5, 0.5])
    n_nodes = 2

    adj = adjacency_from_edge_index(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, format="csr"
    )
    assert adj.shape == (2, 2)
    assert adj[0, 1] == 0.5

    adj_dense = adjacency_from_edge_index(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=None, format="dense"
    )
    assert isinstance(adj_dense, np.ndarray)
    assert adj_dense.shape == (2, 2)
    assert adj_dense[0, 1] == 1.0

    with pytest.raises(ValueError, match="Unknown format"):
        adjacency_from_edge_index(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=None, format="invalid"
        )
