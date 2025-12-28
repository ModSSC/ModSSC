from collections import defaultdict
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.graph.artifacts import GraphArtifact, NodeDataset
from modssc.graph.errors import GraphValidationError, OptionalDependencyError
from modssc.graph.featurization.api import graph_to_views
from modssc.graph.featurization.ops.adjacency import adjacency_from_edge_index
from modssc.graph.featurization.views.diffusion import _to_dense, diffusion_view
from modssc.graph.featurization.views.struct import (
    StructParams,
    _build_neighbors,
    _cooccurrence_counts,
    _iter_random_walks,
    _ppmi_matrix_dense,
    _random_walks,
    struct_embeddings,
)
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


def test_build_neighbors_out_of_bounds():
    edge_index = np.array([[0], [100]])
    neigh_arr, neigh_set = _build_neighbors(edge_index, n_nodes=5)

    assert len(neigh_arr) == 5

    assert len(neigh_arr[0]) == 0


def test_random_walks_no_walks():
    params = StructParams(num_walks_per_node=0)
    walks = _random_walks(
        neighbors=[np.array([1]), np.array([0])], neighbor_sets=[{1}, {0}], params=params, seed=42
    )
    assert len(walks) == 0


def test_random_walks_isolated_node():
    params = StructParams(num_walks_per_node=1, walk_length=3)
    walks = _random_walks(
        neighbors=[np.array([], dtype=int)], neighbor_sets=[set()], params=params, seed=42
    )

    assert len(walks) == 1
    assert np.all(walks[0] == 0)


def test_random_walks_node2vec_weights():
    params = StructParams(method="node2vec", p=0.1, q=10.0, num_walks_per_node=1, walk_length=3)

    neighbors = [np.array([1]), np.array([0, 2]), np.array([1])]
    neighbor_sets = [{1}, {0, 2}, {1}]

    walks = _random_walks(neighbors=neighbors, neighbor_sets=neighbor_sets, params=params, seed=42)
    assert len(walks) == 3
    for w in walks:
        assert len(w) == 3


def test_random_walks_node2vec_zero_weights():
    pass


def test_iter_random_walks_requires_neighbor_sets():
    params = StructParams(method="node2vec", num_walks_per_node=1, walk_length=2)
    with pytest.raises(ValueError, match="neighbor_sets required"):
        list(
            _iter_random_walks(
                neighbors=[np.array([1])], neighbor_sets=None, params=params, seed=42
            )
        )


def test_ppmi_matrix_dense_zeros():
    counts = defaultdict(int)

    mat = _ppmi_matrix_dense(
        counts=counts, n_nodes=5, row_sum=np.zeros(5), col_sum=np.zeros(5), total=0
    )
    assert np.all(mat == 0)


def test_ppmi_matrix_dense_no_counts():
    mat = _ppmi_matrix_dense(
        counts={}, n_nodes=3, row_sum=np.zeros(3), col_sum=np.zeros(3), total=1
    )
    assert np.all(mat == 0)


def test_ppmi_matrix_dense_positive_pmi():
    counts = {(0, 1): 2}
    row_sum = np.array([1, 1], dtype=np.int64)
    col_sum = np.array([1, 1], dtype=np.int64)
    mat = _ppmi_matrix_dense(counts=counts, n_nodes=2, row_sum=row_sum, col_sum=col_sum, total=2)
    assert mat[0, 1] > 0


def test_struct_embeddings_svd_fallback():
    edge_index = np.array([[0, 1], [1, 0]])

    with patch("modssc.graph.featurization.views.struct.optional_import") as mock_imp:
        mock_imp.side_effect = OptionalDependencyError(extra="sklearn")

        emb = struct_embeddings(
            edge_index=edge_index, n_nodes=2, params=StructParams(dim=2), seed=42
        )
        assert emb.shape == (2, 2)


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


def test_random_walks_deepwalk():
    params = StructParams(method="deepwalk", num_walks_per_node=1, walk_length=3)
    neighbors = [np.array([1]), np.array([0])]
    neighbor_sets = [{1}, {0}]
    walks = _random_walks(neighbors=neighbors, neighbor_sets=neighbor_sets, params=params, seed=42)
    assert len(walks) == 2


def test_cooccurrence_counts_no_pairs():
    walks = [np.array([0, 5], dtype=np.int64)]
    counts, row_sum, col_sum, total = _cooccurrence_counts(walks, window_size=1, n_nodes=2)
    assert counts == {}
    assert total == 0


def test_ppmi_matrix_dense_out_of_bounds():
    counts = {(0, 100): 1}
    mat = _ppmi_matrix_dense(
        counts=counts, n_nodes=5, row_sum=np.ones(5), col_sum=np.ones(5), total=1
    )
    assert np.all(mat == 0)


def test_ppmi_matrix_dense_zero_sums():
    counts = {(0, 1): 1}
    row_sum = np.zeros(5)
    col_sum = np.ones(5)
    mat = _ppmi_matrix_dense(counts=counts, n_nodes=5, row_sum=row_sum, col_sum=col_sum, total=1)
    assert mat[0, 1] == 0


def test_struct_embeddings_empty_nodes():
    emb = struct_embeddings(edge_index=np.array([]), n_nodes=0, params=StructParams(), seed=42)
    assert emb.shape == (0, 64)


def test_struct_embeddings_svd_fewer_components():
    edge_index = np.array([[0, 1], [1, 0]])
    params = StructParams(dim=10, max_dense_nodes=100)

    with patch("numpy.linalg.svd") as mock_svd:
        mock_svd.return_value = (np.eye(2), np.ones(2), np.eye(2))

        emb = struct_embeddings(edge_index=edge_index, n_nodes=2, params=params, seed=42)

        assert emb.shape == (2, 10)
        assert np.all(emb[:, 2:] == 0)


def test_struct_embeddings_large_graph_success():
    edge_index = np.array([[0, 1], [1, 0]])
    params = StructParams(dim=2, max_dense_nodes=1)

    with patch("modssc.graph.featurization.views.struct.optional_import") as mock_imp:
        mock_scipy = MagicMock()
        mock_sklearn = MagicMock()
        mock_svd_cls = MagicMock()
        mock_svd_inst = MagicMock()

        mock_svd_inst.fit_transform.return_value = np.zeros((2, 2))
        mock_svd_cls.return_value = mock_svd_inst
        mock_sklearn.TruncatedSVD = mock_svd_cls

        def side_effect(name, extra=None):
            if "scipy" in name:
                return mock_scipy
            if "sklearn" in name:
                return mock_sklearn
            return MagicMock()

        mock_imp.side_effect = side_effect

        emb = struct_embeddings(edge_index=edge_index, n_nodes=2, params=params, seed=42)
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

    with patch("modssc.graph.featurization.views.struct.optional_import") as mock_imp:
        mock_imp.side_effect = (
            lambda name, extra=None: mock_scipy if "scipy" in name else mock_sklearn
        )
        emb = struct_embeddings(edge_index=edge_index, n_nodes=2, params=params, seed=42)
        assert emb.shape == (2, 2)


def test_struct_embeddings_sparse_mask_fill(monkeypatch):
    edge_index = np.array([[0, 1], [1, 0]])
    params = StructParams(dim=2, max_dense_nodes=1)

    monkeypatch.setattr(
        "modssc.graph.featurization.views.struct._cooccurrence_counts",
        lambda *a, **k: ({(0, 1): 1}, np.array([1, 1]), np.array([1, 1]), 1),
    )

    mock_scipy = MagicMock()
    mock_sklearn = MagicMock()
    mock_svd_cls = MagicMock()
    mock_svd_inst = MagicMock()
    mock_svd_inst.fit_transform.return_value = np.zeros((2, 2), dtype=np.float32)
    mock_svd_cls.return_value = mock_svd_inst
    mock_sklearn.TruncatedSVD = mock_svd_cls

    with patch("modssc.graph.featurization.views.struct.optional_import") as mock_imp:
        mock_imp.side_effect = (
            lambda name, extra=None: mock_scipy if "scipy" in name else mock_sklearn
        )
        emb = struct_embeddings(edge_index=edge_index, n_nodes=2, params=params, seed=42)
        assert emb.shape == (2, 2)


def test_struct_embeddings_sparse_missing_row_col(monkeypatch):
    edge_index = np.array([[0, 1], [1, 0]])
    params = StructParams(dim=2, max_dense_nodes=1)

    monkeypatch.setattr(
        "modssc.graph.featurization.views.struct._cooccurrence_counts",
        lambda *a, **k: (
            {(0, 1): 1},
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            1,
        ),
    )

    mock_scipy = MagicMock()
    mock_sklearn = MagicMock()
    mock_svd_cls = MagicMock()
    mock_svd_inst = MagicMock()
    mock_svd_inst.fit_transform.return_value = np.zeros((2, 2), dtype=np.float32)
    mock_svd_cls.return_value = mock_svd_inst
    mock_sklearn.TruncatedSVD = mock_svd_cls

    with patch("modssc.graph.featurization.views.struct.optional_import") as mock_imp:
        mock_imp.side_effect = (
            lambda name, extra=None: mock_scipy if "scipy" in name else mock_sklearn
        )
        emb = struct_embeddings(edge_index=edge_index, n_nodes=2, params=params, seed=42)
        assert emb.shape == (2, 2)


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


def test_struct_embeddings_dense_dim_check():
    edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    params = StructParams(dim=10, max_dense_nodes=100)
    emb = struct_embeddings(edge_index=edge_index, n_nodes=5, params=params, seed=42)
    assert emb.shape == (5, 10)
    assert np.all(emb[:, 5:] == 0)

    edge_index = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
    params = StructParams(dim=5, max_dense_nodes=100)
    emb = struct_embeddings(edge_index=edge_index, n_nodes=10, params=params, seed=42)
    assert emb.shape == (10, 5)


def test_random_walks_node2vec_backtrack():
    edge_index = np.array([[0, 1], [1, 0]])
    neighbors, neighbor_sets = _build_neighbors(edge_index, n_nodes=2)

    params = StructParams(method="node2vec", walk_length=10, num_walks_per_node=1, p=0.5, q=2.0)

    walks = _random_walks(neighbors=neighbors, neighbor_sets=neighbor_sets, params=params, seed=42)
    assert len(walks) == 2


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
