from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.graph.construction.api import build_graph
from modssc.graph.construction.backends.faiss_backend import FaissParams, knn_search_faiss
from modssc.graph.construction.backends.sklearn_backend import (
    epsilon_edges_sklearn,
    knn_edges_sklearn,
)
from modssc.graph.construction.builder import _pick_backend, build_raw_edges
from modssc.graph.construction.ops.normalize import normalize_edge_weights
from modssc.graph.construction.ops.self_loops import add_self_loops
from modssc.graph.construction.ops.symmetrize import symmetrize_edges
from modssc.graph.construction.ops.weights import compute_edge_weights
from modssc.graph.construction.schemes.anchor import (
    AnchorParams,
    _choose_anchors,
    _knn_query_numpy,
    anchor_edges,
)
from modssc.graph.errors import GraphValidationError
from modssc.graph.specs import GraphBuilderSpec, GraphWeightsSpec


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        pytest.skip(f"sklearn unavailable: {exc}")


def test_normalize_edge_weights():
    n_nodes = 3
    edge_index = np.array([[0, 1, 2], [1, 2, 0]])
    edge_weight = np.array([1.0, 2.0, 4.0], dtype=np.float32)

    assert (
        normalize_edge_weights(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="none"
        )
        is edge_weight
    )

    assert (
        normalize_edge_weights(n_nodes=n_nodes, edge_index=edge_index, edge_weight=None, mode="rw")
        is None
    )

    w_rw = normalize_edge_weights(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="rw"
    )
    assert np.allclose(w_rw, [1.0, 1.0, 1.0])

    w_sym = normalize_edge_weights(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="sym"
    )
    assert np.allclose(w_sym, [1.0 / np.sqrt(2), 2.0 / np.sqrt(8), 4.0 / 2.0])

    with pytest.raises(ValueError, match="Unknown normalization mode"):
        normalize_edge_weights(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="invalid"
        )


def test_normalize_edge_weights_negative_src():
    n_nodes = 3
    edge_index = np.array([[-1, 0], [1, 2]])
    edge_weight = np.array([1.0, 2.0], dtype=np.float32)
    w_rw = normalize_edge_weights(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="rw"
    )
    assert w_rw.shape == (2,)


def test_compute_edge_weights():
    d = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    w = compute_edge_weights(
        distances=d, metric="euclidean", weights=GraphWeightsSpec(kind="binary")
    )
    assert np.allclose(w, [1.0, 1.0, 1.0])

    w = compute_edge_weights(
        distances=d, metric="euclidean", weights=GraphWeightsSpec(kind="heat", sigma=1.0)
    )

    assert np.allclose(w, np.exp(-(d**2) / 2.0))

    with pytest.raises(GraphValidationError, match="sigma must be > 0"):
        compute_edge_weights(
            distances=d, metric="euclidean", weights=GraphWeightsSpec(kind="heat", sigma=0.0)
        )

    w = compute_edge_weights(distances=d, metric="cosine", weights=GraphWeightsSpec(kind="cosine"))
    assert np.allclose(w, [1.0, 0.0, -1.0])

    with pytest.raises(GraphValidationError, match="cosine weights require metric='cosine'"):
        compute_edge_weights(
            distances=d, metric="euclidean", weights=GraphWeightsSpec(kind="cosine")
        )

    with pytest.raises(GraphValidationError, match="distances must be 1D"):
        compute_edge_weights(
            distances=np.zeros((2, 2)), metric="euclidean", weights=GraphWeightsSpec(kind="binary")
        )


def test_self_loops():
    n_nodes = 3

    edge_index = np.array([[0, 1], [1, 1]])
    edge_weight = np.array([0.5, 0.8])

    ei2, ew2 = add_self_loops(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, weight=1.0
    )

    assert ei2.shape[1] == 4

    assert np.isclose(ew2.sum(), 3.3)


def test_symmetrize_edges():
    n_nodes = 3

    edge_index = np.array([[0, 1, 1], [1, 0, 2]])
    edge_weight = np.array([0.5, 0.3, 0.8])

    ei, ew = symmetrize_edges(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="or"
    )

    assert ei.shape[1] == 4

    w_dict = {(u, v): w for u, v, w in zip(ei[0], ei[1], ew, strict=False)}
    assert np.isclose(w_dict[(0, 1)], 0.4)
    assert np.isclose(w_dict[(1, 0)], 0.4)
    assert np.isclose(w_dict[(1, 2)], 0.8)
    assert np.isclose(w_dict[(2, 1)], 0.8)

    ei, ew = symmetrize_edges(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="mutual"
    )

    assert ei.shape[1] == 2
    w_dict = {(u, v): w for u, v, w in zip(ei[0], ei[1], ew, strict=False)}
    assert np.isclose(w_dict[(0, 1)], 0.4)
    assert np.isclose(w_dict[(1, 0)], 0.4)

    ei, ew = symmetrize_edges(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="none"
    )
    assert ew is edge_weight

    with pytest.raises(ValueError, match="Unknown symmetrization mode"):
        symmetrize_edges(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="invalid"
        )


def test_sklearn_backend():
    _require_sklearn()
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

    mock_sklearn = MagicMock()
    mock_nn = MagicMock()
    mock_sklearn.NearestNeighbors.return_value = mock_nn

    mock_nn.kneighbors.return_value = (
        np.array([[0, 1], [0, 1], [0, 1], [0, 1]]),
        np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
    )

    with patch(
        "modssc.graph.construction.backends.sklearn_backend.optional_import",
        return_value=mock_sklearn,
    ):
        ei, dist = knn_edges_sklearn(X, k=1, metric="euclidean", include_self=False)

        assert ei.shape[1] == 4
        assert np.allclose(dist, 1.0)

    mock_nn.radius_neighbors.return_value = (
        np.array([np.array([0.0, 1.0]), np.array([0.0])], dtype=object),
        np.array([np.array([0, 1]), np.array([1])], dtype=object),
    )

    with patch(
        "modssc.graph.construction.backends.sklearn_backend.optional_import",
        return_value=mock_sklearn,
    ):
        ei, dist = epsilon_edges_sklearn(X[:2], radius=1.5, metric="euclidean", include_self=False)

        assert ei.shape[1] == 1
        assert ei[0, 0] == 0
        assert ei[1, 0] == 1

    with patch(
        "modssc.graph.construction.backends.sklearn_backend.optional_import",
        return_value=mock_sklearn,
    ):
        ei, dist = epsilon_edges_sklearn(X[:2], radius=1.5, metric="euclidean", include_self=True)

        assert ei.shape[1] == 3
        assert (ei[:, 0] == np.array([0, 0])).all()
        assert (ei[:, 1] == np.array([0, 1])).all()
        assert (ei[:, 2] == np.array([1, 1])).all()

    ei, dist = knn_edges_sklearn(np.zeros((0, 2)), k=1, metric="euclidean")
    assert ei.shape[1] == 0

    ei, dist = epsilon_edges_sklearn(np.zeros((0, 2)), radius=1.0, metric="euclidean")
    assert ei.shape[1] == 0


def test_faiss_backend():
    X = np.array([[0, 0], [1, 1]], dtype=np.float32)

    mock_faiss = MagicMock()
    mock_index = MagicMock()
    mock_faiss.IndexFlatL2.return_value = mock_index
    mock_faiss.IndexFlatIP.return_value = mock_index
    mock_faiss.IndexHNSWFlat.return_value = mock_index
    mock_faiss.METRIC_L2 = 1
    mock_faiss.METRIC_INNER_PRODUCT = 2

    mock_index.search.return_value = (
        np.array([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32),
        np.array([[0, 1], [1, 0]], dtype=np.int64),
    )

    with patch(
        "modssc.graph.construction.backends.faiss_backend.optional_import", return_value=mock_faiss
    ):
        ei, dist = knn_search_faiss(X, X, k=1, metric="euclidean", include_self=False)

        assert ei.shape == (2, 1)
        assert np.allclose(dist, np.sqrt(2.0))

        mock_index.search.return_value = (
            np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            np.array([[0, 1], [1, 0]], dtype=np.int64),
        )
        ei, dist = knn_search_faiss(X, X, k=1, metric="cosine", include_self=False)
        assert np.allclose(dist, 1.0)

        params = FaissParams(exact=False)
        knn_search_faiss(X, X, k=1, metric="euclidean", params=params)
        mock_faiss.IndexHNSWFlat.assert_called()


def test_builder_dispatch():
    X = np.zeros((5, 2))
    spec = GraphBuilderSpec(scheme="knn", k=2, backend="auto")

    with patch("builtins.__import__", side_effect=ImportError):
        assert _pick_backend(spec) == "numpy"

    with patch("builtins.__import__"):
        assert _pick_backend(spec) == "sklearn"

    with patch("modssc.graph.construction.builder.knn_edges_sklearn") as mock_sklearn:
        mock_sklearn.return_value = (np.zeros((2, 0)), np.zeros(0))
        build_raw_edges(X, spec=GraphBuilderSpec(scheme="knn", k=2, backend="sklearn"), seed=0)
        mock_sklearn.assert_called()

    with patch("modssc.graph.construction.builder.knn_edges_faiss") as mock_faiss:
        mock_faiss.return_value = (np.zeros((2, 10)), np.zeros(10))
        build_raw_edges(X, spec=GraphBuilderSpec(scheme="knn", k=2, backend="faiss"), seed=0)
        mock_faiss.assert_called()

    with pytest.raises(GraphValidationError, match="k must be a positive integer"):
        build_raw_edges(X, spec=GraphBuilderSpec(scheme="knn", k=None), seed=0)

    with pytest.raises(GraphValidationError, match="radius must be > 0"):
        build_raw_edges(X, spec=GraphBuilderSpec(scheme="epsilon", radius=None), seed=0)

    with pytest.raises(GraphValidationError, match="faiss backend does not support epsilon"):
        build_raw_edges(
            X, spec=GraphBuilderSpec(scheme="epsilon", radius=1.0, backend="faiss"), seed=0
        )


def test_anchor_scheme():
    X = np.random.randn(10, 2)

    anchors = _choose_anchors(X, n_anchors=2, method="random", seed=42)
    assert anchors.shape == (2,)
    assert anchors.dtype == np.int64

    mock_sklearn = MagicMock()
    mock_km = MagicMock()
    mock_sklearn.MiniBatchKMeans.return_value = mock_km
    mock_km.cluster_centers_ = np.zeros((2, 2), dtype=np.float32)

    with patch(
        "modssc.graph.construction.schemes.anchor.optional_import", return_value=mock_sklearn
    ):
        anchors = _choose_anchors(X, n_anchors=2, method="kmeans", seed=42)
        assert anchors.shape == (2, 2)
        assert anchors.dtype == np.float32

    q = np.array([[0, 0]], dtype=np.float32)
    r = np.array([[0, 0], [1, 1]], dtype=np.float32)
    idx, dist = _knn_query_numpy(q, r, k=2, metric="euclidean", chunk_size=10)
    assert np.allclose(dist[0], [0.0, 1.41421356])

    q = np.array([[1, 0]], dtype=np.float32)
    r = np.array([[1, 0], [0, 1]], dtype=np.float32)
    idx, dist = _knn_query_numpy(q, r, k=2, metric="cosine", chunk_size=10)

    assert np.allclose(dist[0], [0.0, 1.0])


def test_api_non_finite_check():
    with patch("modssc.graph.construction.api.build_raw_edges") as mock_build:
        mock_build.return_value = (np.array([[0, 1], [1, 0]]), np.array([np.inf, 1.0]))

        with patch(
            "modssc.graph.construction.api.compute_edge_weights",
            return_value=np.array([np.inf, 1.0]),
        ):
            spec = GraphBuilderSpec(
                scheme="knn", k=1, normalize="none", symmetrize="none", self_loops=False
            )
            with pytest.raises(GraphValidationError, match="Non-finite edge weights"):
                build_graph(np.zeros((2, 2)), spec=spec)


def test_anchor_edges_integration():
    X = np.array([[0, 0], [0.1, 0], [10, 0], [10.1, 0]], dtype=np.float32)

    with patch(
        "modssc.graph.construction.schemes.anchor._choose_anchors",
        return_value=np.array([0, 2], dtype=np.int64),
    ):
        params = AnchorParams(n_anchors=2, anchors_k=1, candidate_limit=10, chunk_size=100)

        ei, dist = anchor_edges(
            X, k=1, metric="euclidean", backend="numpy", seed=42, params=params, include_self=False
        )

        assert ei.shape[1] == 4

        pairs = set(zip(ei[0], ei[1], strict=False))
        assert (0, 1) in pairs
        assert (1, 0) in pairs
        assert (2, 3) in pairs
        assert (3, 2) in pairs


def test_symmetrize_self_loops():
    ei = np.array([[0, 0], [0, 1]])
    ew = np.array([1.0, 0.5])

    ei_out, ew_out = symmetrize_edges(n_nodes=2, edge_index=ei, edge_weight=ew, mode="or")

    pairs = set(zip(ei_out[0], ei_out[1], strict=False))
    assert (0, 0) in pairs
    assert (0, 1) in pairs
    assert (1, 0) in pairs

    w_dict = {(u, v): w for u, v, w in zip(ei_out[0], ei_out[1], ew_out, strict=False)}
    assert w_dict[(0, 0)] == 1.0
    assert w_dict[(0, 1)] == 0.5
    assert w_dict[(1, 0)] == 0.5
