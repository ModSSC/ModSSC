from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np

from ...errors import GraphValidationError

Metric = Literal["cosine", "euclidean"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def knn_edges_precomputed(
    X: np.ndarray,
    *,
    k: int,
    metric: Metric,
    include_self: bool,
    path: str | Path,
    expected_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an immutable ``I/J/D`` kNN artifact.

    ``I``, ``J`` and ``D`` are the arrays emitted by GraphLearning's
    ``knnsearch`` routine. The file is authenticated before NumPy opens it.
    """

    if metric != "euclidean":
        raise GraphValidationError("precomputed GraphLearning kNN data require euclidean metric")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise GraphValidationError(f"precomputed kNN artifact is missing: {source}")
    actual_sha256 = _sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise GraphValidationError(
            "precomputed kNN artifact SHA-256 differs: "
            f"computed {actual_sha256}, expected {expected_sha256}"
        )

    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {"I", "J", "D"}:
                raise GraphValidationError("precomputed kNN artifact must contain I, J, and D")
            row_ids = np.asarray(archive["I"], dtype=np.int64)
            neighbors = np.asarray(archive["J"], dtype=np.int64)
            distances = np.asarray(archive["D"], dtype=np.float64)
    except GraphValidationError:
        raise
    except Exception as exc:
        raise GraphValidationError(f"cannot load precomputed kNN artifact: {source}") from exc

    if row_ids.ndim != 2 or neighbors.shape != row_ids.shape or distances.shape != row_ids.shape:
        raise GraphValidationError("precomputed I, J, and D arrays must share one 2D shape")
    n_nodes = int(np.asarray(X).shape[0])
    if row_ids.shape[0] != n_nodes:
        raise GraphValidationError(
            f"precomputed kNN rows differ from input nodes: {row_ids.shape[0]} != {n_nodes}"
        )
    if neighbors.size and (neighbors.min() < 0 or neighbors.max() >= n_nodes):
        raise GraphValidationError("precomputed neighbor ids are out of range")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
        raise GraphValidationError("precomputed distances must be finite and non-negative")
    expected_rows = np.arange(n_nodes, dtype=np.int64)[:, None]
    if not np.array_equal(row_ids, np.broadcast_to(expected_rows, row_ids.shape)):
        raise GraphValidationError("precomputed I rows do not identify their query vertex")
    if np.any(np.diff(distances, axis=1) < -1.0e-12):
        raise GraphValidationError("precomputed neighbors are not sorted by distance")

    selected_neighbors: list[np.ndarray] = []
    selected_distances: list[np.ndarray] = []
    for row in range(n_nodes):
        row_neighbors = neighbors[row]
        row_distances = distances[row]
        if include_self:
            if row_neighbors.size == 0 or int(row_neighbors[0]) != row:
                raise GraphValidationError(
                    "include_self_in_knn requires the query vertex as the first neighbor"
                )
            positions = np.arange(min(int(k), row_neighbors.size), dtype=np.int64)
        else:
            positions = np.flatnonzero(row_neighbors != row)[: int(k)]
        if positions.size != min(int(k), n_nodes if include_self else max(n_nodes - 1, 0)):
            raise GraphValidationError("precomputed artifact has too few neighbors")
        selected_neighbors.append(row_neighbors[positions])
        selected_distances.append(row_distances[positions])

    if not selected_neighbors:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    width = int(selected_neighbors[0].size)
    src = np.repeat(np.arange(n_nodes, dtype=np.int64), width)
    dst = np.concatenate(selected_neighbors).astype(np.int64, copy=False)
    distance = np.concatenate(selected_distances).astype(np.float64, copy=False)
    return np.vstack([src, dst]), distance
