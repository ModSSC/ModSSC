from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from modssc.graph.construction.backends.precomputed_backend import knn_edges_precomputed
from modssc.graph.errors import GraphValidationError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive(
    path: Path,
    *,
    rows: np.ndarray | None = None,
    neighbors: np.ndarray | None = None,
    distances: np.ndarray | None = None,
) -> str:
    default_rows = np.repeat(np.arange(3, dtype=np.int64)[:, None], 3, axis=1)
    default_neighbors = np.array(
        [
            [0, 1, 2],
            [1, 0, 2],
            [2, 1, 0],
        ],
        dtype=np.int64,
    )
    default_distances = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 1.5],
            [0.0, 0.5, 2.0],
        ],
        dtype=np.float64,
    )
    np.savez(
        path,
        I=default_rows if rows is None else rows,
        J=default_neighbors if neighbors is None else neighbors,
        D=default_distances if distances is None else distances,
    )
    return _sha256(path)


def test_precomputed_backend_preserves_k_including_self(tmp_path) -> None:
    path = tmp_path / "knn.npz"
    digest = _archive(path)
    edge_index, distances = knn_edges_precomputed(
        np.zeros((3, 4), dtype=np.float32),
        k=2,
        metric="euclidean",
        include_self=True,
        path=path,
        expected_sha256=digest,
    )

    np.testing.assert_array_equal(
        edge_index,
        np.array([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 2, 1]], dtype=np.int64),
    )
    np.testing.assert_allclose(distances, [0.0, 1.0, 0.0, 1.0, 0.0, 0.5])


def test_precomputed_backend_can_exclude_self(tmp_path) -> None:
    path = tmp_path / "knn.npz"
    digest = _archive(path)
    edge_index, distances = knn_edges_precomputed(
        np.zeros((3, 1), dtype=np.float32),
        k=2,
        metric="euclidean",
        include_self=False,
        path=path,
        expected_sha256=digest,
    )

    np.testing.assert_array_equal(
        edge_index,
        np.array([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 1, 0]], dtype=np.int64),
    )
    np.testing.assert_allclose(distances, [1.0, 2.0, 1.0, 1.5, 0.5, 2.0])


def test_precomputed_backend_handles_an_empty_graph(tmp_path) -> None:
    path = tmp_path / "empty.npz"
    digest = _archive(
        path,
        rows=np.zeros((0, 0), dtype=np.int64),
        neighbors=np.zeros((0, 0), dtype=np.int64),
        distances=np.zeros((0, 0), dtype=np.float64),
    )
    edge_index, distances = knn_edges_precomputed(
        np.zeros((0, 1), dtype=np.float32),
        k=1,
        metric="euclidean",
        include_self=True,
        path=path,
        expected_sha256=digest,
    )
    assert edge_index.shape == (2, 0)
    assert distances.shape == (0,)


def test_precomputed_backend_authenticates_before_loading(tmp_path) -> None:
    path = tmp_path / "knn.npz"
    _archive(path)
    with pytest.raises(GraphValidationError, match="SHA-256 differs"):
        knn_edges_precomputed(
            np.zeros((3, 1)),
            k=1,
            metric="euclidean",
            include_self=True,
            path=path,
            expected_sha256="0" * 64,
        )
    with pytest.raises(GraphValidationError, match="artifact is missing"):
        knn_edges_precomputed(
            np.zeros((3, 1)),
            k=1,
            metric="euclidean",
            include_self=True,
            path=tmp_path / "missing.npz",
            expected_sha256="0" * 64,
        )
    with pytest.raises(GraphValidationError, match="require euclidean"):
        knn_edges_precomputed(
            np.zeros((3, 1)),
            k=1,
            metric="cosine",
            include_self=True,
            path=path,
            expected_sha256=_sha256(path),
        )


def test_precomputed_backend_rejects_bad_archives(tmp_path) -> None:
    missing_key = tmp_path / "missing-key.npz"
    np.savez(missing_key, I=np.zeros((1, 1)), J=np.zeros((1, 1)))
    with pytest.raises(GraphValidationError, match="must contain I, J, and D"):
        knn_edges_precomputed(
            np.zeros((1, 1)),
            k=1,
            metric="euclidean",
            include_self=True,
            path=missing_key,
            expected_sha256=_sha256(missing_key),
        )

    malformed = tmp_path / "malformed.npz"
    malformed.write_bytes(b"not an npz")
    with pytest.raises(GraphValidationError, match="cannot load"):
        knn_edges_precomputed(
            np.zeros((1, 1)),
            k=1,
            metric="euclidean",
            include_self=True,
            path=malformed,
            expected_sha256=_sha256(malformed),
        )


@pytest.mark.parametrize(
    ("rows", "neighbors", "distances", "x_rows", "message"),
    [
        (
            np.zeros((3,), dtype=np.int64),
            np.zeros((3,), dtype=np.int64),
            np.zeros((3,), dtype=np.float64),
            3,
            "share one 2D shape",
        ),
        (
            np.zeros((2, 1), dtype=np.int64),
            np.zeros((2, 1), dtype=np.int64),
            np.zeros((2, 1), dtype=np.float64),
            3,
            "rows differ from input nodes",
        ),
        (
            np.repeat(np.arange(3)[:, None], 2, axis=1),
            np.array([[0, 3], [1, 0], [2, 1]]),
            np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            3,
            "neighbor ids are out of range",
        ),
        (
            np.repeat(np.arange(3)[:, None], 2, axis=1),
            np.array([[0, 1], [1, 0], [2, 1]]),
            np.array([[0.0, np.nan], [0.0, 1.0], [0.0, 1.0]]),
            3,
            "finite and non-negative",
        ),
        (
            np.zeros((3, 2), dtype=np.int64),
            np.array([[0, 1], [1, 0], [2, 1]]),
            np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            3,
            "do not identify their query",
        ),
        (
            np.repeat(np.arange(3)[:, None], 2, axis=1),
            np.array([[0, 1], [1, 0], [2, 1]]),
            np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            3,
            "not sorted",
        ),
    ],
)
def test_precomputed_backend_rejects_invalid_arrays(
    tmp_path,
    rows,
    neighbors,
    distances,
    x_rows,
    message,
) -> None:
    path = tmp_path / "invalid.npz"
    digest = _archive(
        path,
        rows=np.asarray(rows),
        neighbors=np.asarray(neighbors),
        distances=np.asarray(distances),
    )
    with pytest.raises(GraphValidationError, match=message):
        knn_edges_precomputed(
            np.zeros((x_rows, 1)),
            k=1,
            metric="euclidean",
            include_self=True,
            path=path,
            expected_sha256=digest,
        )


def test_precomputed_backend_rejects_missing_self_and_short_rows(tmp_path) -> None:
    missing_self = tmp_path / "missing-self.npz"
    digest = _archive(
        missing_self,
        neighbors=np.array([[1, 2, 0], [0, 2, 1], [1, 0, 2]]),
        distances=np.array([[0.0, 1.0, 2.0]] * 3),
    )
    with pytest.raises(GraphValidationError, match="first neighbor"):
        knn_edges_precomputed(
            np.zeros((3, 1)),
            k=1,
            metric="euclidean",
            include_self=True,
            path=missing_self,
            expected_sha256=digest,
        )

    short = tmp_path / "short.npz"
    digest = _archive(
        short,
        rows=np.arange(3, dtype=np.int64)[:, None],
        neighbors=np.arange(3, dtype=np.int64)[:, None],
        distances=np.zeros((3, 1), dtype=np.float64),
    )
    with pytest.raises(GraphValidationError, match="too few neighbors"):
        knn_edges_precomputed(
            np.zeros((3, 1)),
            k=2,
            metric="euclidean",
            include_self=True,
            path=short,
            expected_sha256=digest,
        )
    with pytest.raises(GraphValidationError, match="too few neighbors"):
        knn_edges_precomputed(
            np.zeros((3, 1)),
            k=1,
            metric="euclidean",
            include_self=False,
            path=short,
            expected_sha256=digest,
        )
