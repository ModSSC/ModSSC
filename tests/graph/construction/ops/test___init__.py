from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.graph.construction.ops.self_loops import add_self_loops
from modssc.graph.construction.ops.symmetrize import symmetrize_edges
from modssc.graph.construction.ops.weights import compute_edge_weights
from modssc.graph.construction.schemes.anchor import (
    AnchorParams,
    _choose_anchors,
    _derive_seed,
    anchor_edges,
)
from modssc.graph.errors import GraphValidationError
from modssc.graph.specs import GraphWeightsSpec


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        pytest.skip(f"sklearn unavailable: {exc}")


def test_self_loops_no_missing():
    n = 3
    edge_index = np.array([[0, 1, 2], [0, 1, 2]])
    edge_weight = np.array([1.0, 1.0, 1.0])

    ei, ew = add_self_loops(n_nodes=n, edge_index=edge_index, edge_weight=edge_weight)

    np.testing.assert_array_equal(ei, edge_index)
    np.testing.assert_array_equal(ew, edge_weight)


def test_self_loops_no_weights():
    n = 3
    edge_index = np.array([[0], [1]])

    ei, ew = add_self_loops(n_nodes=n, edge_index=edge_index, edge_weight=None)

    assert ew is None
    assert ei.shape[1] == 1 + 3

    src, dst = ei
    for i in range(n):
        mask = (src == i) & (dst == i)
        assert np.any(mask)


def test_symmetrize_no_weights():
    n = 3
    edge_index = np.array([[0], [1]])

    ei, ew = symmetrize_edges(n_nodes=n, edge_index=edge_index, edge_weight=None, mode="or")

    assert ew is None

    assert ei.shape[1] == 2

    pairs = set(zip(ei[0], ei[1], strict=False))
    assert (0, 1) in pairs
    assert (1, 0) in pairs


def test_symmetrize_with_self_loops():
    n = 2

    edge_index = np.array([[0, 0], [1, 0]])
    edge_weight = np.array([1.0, 0.5])

    ei, ew = symmetrize_edges(n_nodes=n, edge_index=edge_index, edge_weight=edge_weight, mode="or")

    assert ei.shape[1] == 3

    pairs = set(zip(ei[0], ei[1], strict=False))
    assert (0, 1) in pairs
    assert (1, 0) in pairs
    assert (0, 0) in pairs

    mask_loop = (ei[0] == 0) & (ei[1] == 0)
    assert np.allclose(ew[mask_loop], 0.5)


def test_symmetrize_weight_mismatch_truncates():
    n = 3
    edge_index = np.array([[0, 1, 2], [1, 2, 0]])
    edge_weight = np.array([1.0, 2.0], dtype=np.float32)

    ei, ew = symmetrize_edges(n_nodes=n, edge_index=edge_index, edge_weight=edge_weight, mode="or")

    assert ew is not None
    assert ew.size == ei.shape[1]


def test_symmetrize_only_loops():
    n = 3
    edge_index = np.array([[0, 1], [0, 1]])
    edge_weight = np.array([0.5, 1.5], dtype=np.float32)

    ei, ew = symmetrize_edges(n_nodes=n, edge_index=edge_index, edge_weight=edge_weight, mode="or")

    assert ei.shape[1] == 2
    assert np.allclose(ew, [0.5, 1.5])


def test_symmetrize_clips_nodes():
    n = 2
    edge_index = np.array([[5], [7]])
    edge_weight = np.array([1.0], dtype=np.float32)

    ei, ew = symmetrize_edges(n_nodes=n, edge_index=edge_index, edge_weight=edge_weight, mode="or")
    assert ei.max() <= n - 1
    assert ew is not None


def test_symmetrize_empty_edge_index():
    ei = np.zeros((2, 0), dtype=np.int64)
    ew = np.zeros((0,), dtype=np.float32)

    ei_out, ew_out = symmetrize_edges(n_nodes=10, edge_index=ei, edge_weight=ew, mode="or")
    assert ei_out.size == 0
    assert ew_out.size == 0


def test_symmetrize_mode_none_and_invalid():
    ei = np.array([[0, 1], [1, 0]], dtype=np.int64)
    ew = np.array([1.0, 1.0], dtype=np.float32)

    ei_out, ew_out = symmetrize_edges(n_nodes=2, edge_index=ei, edge_weight=ew, mode="none")
    assert ei_out is ei
    assert ew_out is ew

    with pytest.raises(ValueError, match="Unknown symmetrization mode"):
        symmetrize_edges(n_nodes=2, edge_index=ei, edge_weight=ew, mode="invalid_mode")


def test_weights_unknown_kind():
    spec = MagicMock(spec=GraphWeightsSpec)
    spec.kind = "invalid_kind"

    distances = np.array([0.1, 0.2], dtype=np.float32)

    with pytest.raises(GraphValidationError, match="Unknown weight kind"):
        compute_edge_weights(distances=distances, metric="cosine", weights=spec)


def test_derive_seed():
    s1 = _derive_seed(42, 0)
    s2 = _derive_seed(42, 1)
    assert isinstance(s1, int)
    assert s1 != s2
    assert 0 <= s1 <= 0xFFFFFFFF


def test_anchor_empty_graph():
    X = np.zeros((0, 10))
    params = AnchorParams()
    ei, dist = anchor_edges(X, k=5, metric="cosine", backend="numpy", seed=42, params=params)
    assert ei.shape == (2, 0)
    assert dist.shape == (0,)


def test_anchor_kmeans_method(monkeypatch):
    _require_sklearn()
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    X = np.random.rand(20, 5).astype(np.float32)
    params = AnchorParams(n_anchors=5, method="kmeans")

    ei, dist = anchor_edges(X, k=2, metric="cosine", backend="numpy", seed=42, params=params)

    assert ei.shape[0] == 2
    assert ei.shape[1] > 0


def test_anchor_sklearn_backend():
    _require_sklearn()
    X = np.random.rand(20, 5).astype(np.float32)
    params = AnchorParams(n_anchors=5)

    ei, dist = anchor_edges(X, k=2, metric="cosine", backend="sklearn", seed=42, params=params)

    assert ei.shape[0] == 2
    assert ei.shape[1] > 0


def test_anchor_sklearn_backend_mocked():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    params = AnchorParams(n_anchors=2)

    with patch("modssc.graph.construction.schemes.anchor.optional_import") as mock_import:
        mock_sklearn = MagicMock()
        mock_nn = MagicMock()
        mock_sklearn.NearestNeighbors.return_value = mock_nn
        mock_nn.kneighbors.return_value = (
            np.zeros((4, 2), dtype=np.float32),
            np.zeros((4, 2), dtype=np.int64),
        )
        mock_import.return_value = mock_sklearn

        ei, dist = anchor_edges(X, k=2, metric="cosine", backend="sklearn", seed=0, params=params)

    assert ei.shape[0] == 2
    assert dist.shape[0] == ei.shape[1]


def test_anchor_faiss_backend():
    X = np.random.rand(20, 5).astype(np.float32)
    params = AnchorParams(n_anchors=5)

    with patch("modssc.graph.construction.schemes.anchor.knn_search_faiss") as mock_search:
        mock_search.return_value = (
            np.zeros((20, 5), dtype=np.int64),
            np.zeros((20, 5), dtype=np.float32),
        )

        ei, dist = anchor_edges(X, k=2, metric="cosine", backend="faiss", seed=42, params=params)

        assert mock_search.called
        assert ei.shape[0] == 2


def test_anchor_euclidean_metric():
    X = np.random.rand(20, 5).astype(np.float32)
    params = AnchorParams(n_anchors=5)

    ei, dist = anchor_edges(X, k=2, metric="euclidean", backend="numpy", seed=42, params=params)

    assert ei.shape[0] == 2


def test_anchor_candidate_limit():
    X = np.random.rand(20, 5).astype(np.float32)

    params = AnchorParams(n_anchors=5, candidate_limit=2)

    ei, dist = anchor_edges(X, k=2, metric="cosine", backend="numpy", seed=42, params=params)

    assert ei.shape[0] == 2


def test_anchor_unknown_method():
    X = np.random.rand(10, 2)
    with pytest.raises(GraphValidationError, match="Unknown anchor method"):
        _choose_anchors(X, n_anchors=2, method="invalid", seed=42)


def test_anchor_unknown_backend():
    X = np.random.rand(10, 2)
    params = AnchorParams()
    with pytest.raises(GraphValidationError, match="Unknown backend"):
        anchor_edges(X, k=2, metric="cosine", backend="invalid", seed=42, params=params)


def test_self_loops_missing_with_weights():
    n = 2

    edge_index = np.array([[0], [1]])
    edge_weight = np.array([0.5])

    ei, ew = add_self_loops(n_nodes=n, edge_index=edge_index, edge_weight=edge_weight, weight=2.0)

    assert ei.shape[1] == 3
    assert ew.shape[0] == 3

    assert ew[0] == 0.5

    assert ew[1] == 2.0
    assert ew[2] == 2.0


def test_symmetrize_mode_none():
    n = 3
    edge_index = np.array([[0], [1]])
    edge_weight = np.array([1.0])

    ei, ew = symmetrize_edges(
        n_nodes=n, edge_index=edge_index, edge_weight=edge_weight, mode="none"
    )

    assert ei is edge_index
    assert ew is edge_weight


def test_symmetrize_mode_mutual():
    n = 3

    edge_index = np.array([[0, 1, 1], [1, 0, 2]])
    edge_weight = np.array([1.0, 0.5, 1.0])

    ei, ew = symmetrize_edges(
        n_nodes=n, edge_index=edge_index, edge_weight=edge_weight, mode="mutual"
    )

    assert ei.shape[1] == 2

    pairs = set(zip(ei[0], ei[1], strict=False))
    assert (0, 1) in pairs
    assert (1, 0) in pairs
    assert (1, 2) not in pairs

    assert np.allclose(ew, 0.75)


def test_symmetrize_loops_only_with_weights():
    n = 3
    edge_index = np.array([[0, 1], [0, 1]])
    edge_weight = np.array([0.2, 0.4], dtype=np.float32)

    ei, ew = symmetrize_edges(n_nodes=n, edge_index=edge_index, edge_weight=edge_weight, mode="or")

    assert np.array_equal(ei, edge_index)
    assert np.allclose(ew, edge_weight)


def test_symmetrize_loops_only_no_weights():
    n = 3
    edge_index = np.array([[0, 1], [0, 1]])

    ei, ew = symmetrize_edges(n_nodes=n, edge_index=edge_index, edge_weight=None, mode="or")

    assert np.array_equal(ei, edge_index)
    assert ew is None


def test_symmetrize_mutual_no_pairs():
    n = 4
    edge_index = np.array([[0, 1], [2, 3]])
    edge_weight = np.array([1.0, 2.0], dtype=np.float32)

    ei, ew = symmetrize_edges(
        n_nodes=n, edge_index=edge_index, edge_weight=edge_weight, mode="mutual"
    )

    assert ei.shape == (2, 0)
    assert ew is not None
    assert ew.size == 0


def test_symmetrize_invalid_mode():
    n = 3
    edge_index = np.array([[0], [1]])
    with pytest.raises(ValueError, match="Unknown symmetrization mode"):
        symmetrize_edges(n_nodes=n, edge_index=edge_index, edge_weight=None, mode="invalid")


def test_weights_binary():
    spec = GraphWeightsSpec(kind="binary")
    distances = np.array([0.1, 0.5, 0.9], dtype=np.float32)

    w = compute_edge_weights(distances=distances, metric="cosine", weights=spec)

    assert np.all(w == 1.0)
    assert w.dtype == np.float32


def test_weights_heat():
    spec = GraphWeightsSpec(kind="heat", sigma=1.0)
    distances = np.array([0.0, 1.0], dtype=np.float32)

    w = compute_edge_weights(distances=distances, metric="euclidean", weights=spec)

    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[1], np.exp(-0.5))


def test_weights_heat_invalid_sigma():
    spec = MagicMock(spec=GraphWeightsSpec)
    spec.kind = "heat"
    spec.sigma = 0.0

    distances = np.array([0.1], dtype=np.float32)

    with pytest.raises(GraphValidationError, match="sigma must be > 0"):
        compute_edge_weights(distances=distances, metric="euclidean", weights=spec)


def test_weights_cosine():
    spec = GraphWeightsSpec(kind="cosine")
    distances = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    w = compute_edge_weights(distances=distances, metric="cosine", weights=spec)

    assert np.allclose(w, [1.0, 0.5, 0.0])


def test_weights_cosine_metric_mismatch():
    spec = GraphWeightsSpec(kind="cosine")
    distances = np.array([0.1], dtype=np.float32)

    with pytest.raises(GraphValidationError, match="cosine weights require metric='cosine'"):
        compute_edge_weights(distances=distances, metric="euclidean", weights=spec)


def test_symmetrize_self_loops_no_weights():
    n = 2

    edge_index = np.array([[0, 0], [1, 0]])

    ei, ew = symmetrize_edges(n_nodes=n, edge_index=edge_index, edge_weight=None, mode="or")

    assert ei.shape[1] == 3
    assert ew is None

    pairs = set(zip(ei[0], ei[1], strict=False))
    assert (0, 0) in pairs
