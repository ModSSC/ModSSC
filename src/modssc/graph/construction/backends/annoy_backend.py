from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...errors import GraphValidationError
from ...optional import optional_import

Metric = Literal["euclidean"]


@dataclass(frozen=True)
class AnnoyParams:
    """Parameters controlling an Annoy index and its candidate search."""

    n_trees: int = 10
    search_k: int = -1
    query_k: int | None = None
    seed: int = 0
    rerank: bool = False


def _as_float32_contiguous(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float32)
    if not values.flags["C_CONTIGUOUS"]:
        values = np.ascontiguousarray(values)
    return values


def _candidate_width(*, k: int, include_self: bool, query_k: int | None, n: int) -> int:
    minimum = int(k) + (0 if include_self else 1)
    requested = max(2 * int(k), minimum) if query_k is None else int(query_k)
    if requested < minimum:
        raise GraphValidationError(
            "Annoy query_k must retrieve at least k candidates plus self when self is excluded"
        )
    return min(int(n), requested)


def knn_search_annoy(
    X: np.ndarray,
    *,
    k: int,
    metric: Metric,
    include_self: bool = False,
    params: AnnoyParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Search a seeded Annoy index with an independently configurable candidate width.

    ``query_k`` is the Annoy candidate-list width. It is deliberately separate
    from the final ``k`` so protocols can retrieve a wider approximate list and
    retain only its first ``k`` neighbors. Optional exact reranking is explicit;
    it is disabled by default to preserve Annoy's returned order and distances.
    """

    if metric != "euclidean":
        raise GraphValidationError("annoy backend currently requires metric='euclidean'")
    if int(k) <= 0:
        raise GraphValidationError("k must be a positive integer")

    values = np.asarray(X)
    if values.ndim != 2:
        raise GraphValidationError("X must be a 2D array")
    n, dimension = (int(values.shape[0]), int(values.shape[1]))
    output_width = min(int(k), n if include_self else max(0, n - 1))
    if n == 0:
        return (
            np.empty((0, output_width), dtype=np.int64),
            np.empty((0, output_width), dtype=np.float64),
        )
    if dimension <= 0:
        raise GraphValidationError("Annoy requires at least one feature column")
    if not np.isfinite(values).all():
        raise GraphValidationError("Annoy requires finite feature values")

    resolved = params or AnnoyParams()
    if int(resolved.n_trees) <= 0:
        raise GraphValidationError("Annoy n_trees must be > 0")
    if int(resolved.search_k) != -1 and int(resolved.search_k) <= 0:
        raise GraphValidationError("Annoy search_k must be -1 or > 0")
    query_width = _candidate_width(
        k=int(k),
        include_self=bool(include_self),
        query_k=resolved.query_k,
        n=n,
    )

    annoy = optional_import("annoy", extra="graph-annoy")
    index = annoy.AnnoyIndex(dimension, "euclidean")
    index.set_seed(int(resolved.seed))
    indexed_values = _as_float32_contiguous(values)
    for row in range(n):
        index.add_item(row, indexed_values[row])
    index.build(int(resolved.n_trees))

    indices = np.full((n, output_width), -1, dtype=np.int64)
    distances = np.full((n, output_width), np.inf, dtype=np.float64)
    for row in range(n):
        result = index.get_nns_by_item(
            row,
            query_width,
            search_k=int(resolved.search_k),
            include_distances=True,
        )
        candidates = np.asarray(result[0], dtype=np.int64)
        candidate_distances = np.asarray(result[1], dtype=np.float64)
        if candidates.ndim != 1:
            raise GraphValidationError("Annoy returned an invalid neighbor list")
        if candidate_distances.shape != candidates.shape:
            raise GraphValidationError("Annoy returned invalid neighbor distances")
        if candidates.size and (bool((candidates < 0).any()) or bool((candidates >= n).any())):
            raise GraphValidationError("Annoy returned an out-of-range neighbor id")
        if not np.isfinite(candidate_distances).all() or np.any(candidate_distances < 0.0):
            raise GraphValidationError("Annoy returned invalid neighbor distances")

        if include_self:
            self_positions = np.flatnonzero(candidates == row)
            if self_positions.size == 0:
                candidates = np.concatenate([np.asarray([row], dtype=np.int64), candidates])
                candidate_distances = np.concatenate(
                    [np.asarray([0.0], dtype=np.float64), candidate_distances]
                )
            elif int(self_positions[0]) != 0:
                position = int(self_positions[0])
                candidates = np.concatenate(
                    [candidates[position : position + 1], np.delete(candidates, position)]
                )
                candidate_distances = np.concatenate(
                    [
                        candidate_distances[position : position + 1],
                        np.delete(candidate_distances, position),
                    ]
                )
        else:
            keep = candidates != row
            candidates = candidates[keep]
            candidate_distances = candidate_distances[keep]

        if resolved.rerank:
            exact = np.linalg.norm(values[candidates] - values[row], axis=1)
            order = np.argsort(exact, kind="stable")[:output_width]
            selected = candidates[order]
            selected_distances = exact[order]
        else:
            selected = candidates[:output_width]
            selected_distances = candidate_distances[:output_width]
        count = int(selected.size)
        if count != output_width:
            raise GraphValidationError(
                f"Annoy returned only {count} usable neighbors; expected {output_width}"
            )
        if count:
            indices[row, :count] = selected
            distances[row, :count] = selected_distances

    return indices, distances


def knn_edges_annoy(
    X: np.ndarray,
    *,
    k: int,
    metric: Metric,
    include_self: bool = False,
    params: AnnoyParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build directed kNN edges for ``X`` using an Annoy candidate index."""

    indices, distances = knn_search_annoy(
        X,
        k=int(k),
        metric=metric,
        include_self=bool(include_self),
        params=params,
    )
    n = int(np.asarray(X).shape[0])
    source = np.repeat(np.arange(n, dtype=np.int64), indices.shape[1])
    destination = indices.reshape(-1)
    flat_distances = distances.reshape(-1)
    valid = destination >= 0
    edge_index = np.vstack([source[valid], destination[valid]]).astype(np.int64, copy=False)
    return edge_index, flat_distances[valid].astype(np.float64, copy=False)
