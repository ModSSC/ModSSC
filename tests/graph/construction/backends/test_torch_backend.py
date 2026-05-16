import numpy as np
import pytest

from modssc.graph.construction.backends.torch_backend import knn_edges_torch


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception as exc:
        pytest.skip(f"torch unavailable: {exc}")


def test_knn_torch_empty() -> None:
    _require_torch()

    edge_index, dist = knn_edges_torch(np.zeros((0, 10)), k=5, metric="cosine", device="cpu")

    assert edge_index.shape == (2, 0)
    assert dist.shape == (0,)


def test_knn_torch_euclidean() -> None:
    _require_torch()
    X = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)

    edge_index, dist = knn_edges_torch(X, k=1, metric="euclidean", chunk_size=2, device="cpu")

    np.testing.assert_array_equal(edge_index, np.array([[0, 1, 2], [1, 0, 1]]))
    np.testing.assert_allclose(dist, np.array([1.0, 1.0, 2.0], dtype=np.float32))


def test_knn_torch_include_self() -> None:
    _require_torch()
    X = np.array([[0.0], [1.0]], dtype=np.float32)

    edge_index, dist = knn_edges_torch(X, k=1, metric="euclidean", include_self=True, device="cpu")

    assert edge_index.shape[1] == 2
    assert np.all(edge_index[0] == edge_index[1])
    np.testing.assert_allclose(dist, 0.0)


def test_knn_torch_cosine() -> None:
    _require_torch()
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    edge_index, dist = knn_edges_torch(X, k=1, metric="cosine", device="cpu")

    assert edge_index.shape[1] == 3
    np.testing.assert_allclose(dist[0], np.float32(1.0 - 0.70710678), atol=1e-6)


def test_knn_torch_k_eff_zero() -> None:
    _require_torch()
    X = np.array([[0.0]], dtype=np.float32)

    edge_index, dist = knn_edges_torch(X, k=1, metric="euclidean", include_self=False, device="cpu")

    assert edge_index.shape == (2, 0)
    assert dist.shape == (0,)


def test_knn_torch_non_contiguous_invalid_metric_and_nonfinite() -> None:
    _require_torch()
    X = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]

    edge_index, dist = knn_edges_torch(X, k=1, metric="euclidean", device="cpu")

    assert edge_index.shape == (2, 3)
    assert dist.shape == (3,)

    with pytest.raises(ValueError, match="Unknown metric"):
        knn_edges_torch(X, k=1, metric="manhattan", device="cpu")  # type: ignore[arg-type]

    edge_index, dist = knn_edges_torch(
        np.array([[np.nan], [np.nan]], dtype=np.float32),
        k=1,
        metric="euclidean",
        device="cpu",
    )
    assert edge_index.shape == (2, 0)
    assert dist.shape == (0,)
