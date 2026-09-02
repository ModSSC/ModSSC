from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from modssc.graph import GraphBuilderSpec, GraphWeightsSpec, build_graph
from modssc.graph.construction.backends.annoy_backend import (
    AnnoyParams,
    knn_edges_annoy,
    knn_search_annoy,
)
from modssc.graph.errors import GraphValidationError


class _FakeAnnoyIndex:
    def __init__(self, dimension: int, metric: str, *, candidates: dict[int, list[int]]):
        self.dimension = dimension
        self.metric = metric
        self.candidates = candidates
        self.items: dict[int, np.ndarray] = {}
        self.seed: int | None = None
        self.n_trees: int | None = None
        self.queries: list[tuple[int, int, int, bool]] = []

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def add_item(self, item: int, vector: np.ndarray) -> None:
        self.items[item] = np.asarray(vector)

    def build(self, n_trees: int) -> None:
        self.n_trees = n_trees

    def get_nns_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int,
        include_distances: bool,
    ) -> tuple[list[int], list[float]]:
        self.queries.append((item, n, search_k, include_distances))
        selected = self.candidates[item][:n]
        return selected, [999.0] * len(selected)


class _RawResultAnnoyIndex(_FakeAnnoyIndex):
    def __init__(
        self,
        dimension: int,
        metric: str,
        *,
        result: tuple[object, object],
    ) -> None:
        super().__init__(dimension, metric, candidates={})
        self.result = result

    def get_nns_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int,
        include_distances: bool,
    ) -> tuple[object, object]:
        self.queries.append((item, n, search_k, include_distances))
        return self.result


def _install_fake_annoy(monkeypatch, candidates):
    built: list[_FakeAnnoyIndex] = []

    def factory(dimension: int, metric: str) -> _FakeAnnoyIndex:
        index = _FakeAnnoyIndex(dimension, metric, candidates=candidates)
        built.append(index)
        return index

    monkeypatch.setattr(
        "modssc.graph.construction.backends.annoy_backend.optional_import",
        lambda module, *, extra: SimpleNamespace(AnnoyIndex=factory),
    )
    return built


def _install_raw_result_annoy(monkeypatch, result: tuple[object, object]) -> None:
    monkeypatch.setattr(
        "modssc.graph.construction.backends.annoy_backend.optional_import",
        lambda module, *, extra: SimpleNamespace(
            AnnoyIndex=lambda dimension, metric: _RawResultAnnoyIndex(
                dimension,
                metric,
                result=result,
            )
        ),
    )


def test_annoy_supports_wide_query_seed_and_exact_reranking(monkeypatch) -> None:
    X = np.asarray([[0.0], [4.0], [1.0], [2.0]], dtype=np.float64)
    candidates = {row: [row, 1, 3, 2] for row in range(X.shape[0])}
    built = _install_fake_annoy(monkeypatch, candidates)

    indices, distances = knn_search_annoy(
        X,
        k=2,
        metric="euclidean",
        include_self=True,
        params=AnnoyParams(
            n_trees=10,
            query_k=4,
            search_k=-1,
            seed=17,
            rerank=True,
        ),
    )

    index = built[0]
    assert index.dimension == 1
    assert index.metric == "euclidean"
    assert index.seed == 17
    assert index.n_trees == 10
    assert index.queries == [(row, 4, -1, True) for row in range(4)]
    np.testing.assert_array_equal(indices[0], [0, 2])
    np.testing.assert_allclose(distances[0], [0.0, 1.0])
    assert all(item.dtype == np.float32 for item in index.items.values())


def test_annoy_protocol_preserves_returned_order_without_reranking(monkeypatch) -> None:
    X = np.asarray([[0.0], [4.0], [1.0], [2.0]], dtype=np.float64)
    candidates = {row: [row, 1, 3, 2] for row in range(X.shape[0])}
    _install_fake_annoy(monkeypatch, candidates)

    indices, distances = knn_search_annoy(
        X,
        k=2,
        metric="euclidean",
        include_self=True,
        params=AnnoyParams(query_k=4, rerank=False),
    )

    np.testing.assert_array_equal(indices[0], [0, 1])
    np.testing.assert_allclose(distances[0], [999.0, 999.0])


def test_annoy_guarantees_requested_self_policy(monkeypatch) -> None:
    X = np.asarray([[0.0], [1.0], [3.0]], dtype=np.float32)
    candidates_without_self = {0: [1, 2], 1: [0, 2], 2: [1, 0]}
    _install_fake_annoy(monkeypatch, candidates_without_self)

    included, included_distances = knn_search_annoy(
        X,
        k=2,
        metric="euclidean",
        include_self=True,
        params=AnnoyParams(query_k=2),
    )

    np.testing.assert_array_equal(included[:, 0], np.arange(3))
    np.testing.assert_allclose(included_distances[:, 0], 0.0)

    candidates_with_self = {0: [0, 1, 2], 1: [1, 0, 2], 2: [2, 1, 0]}
    _install_fake_annoy(monkeypatch, candidates_with_self)
    excluded, _ = knn_search_annoy(
        X,
        k=2,
        metric="euclidean",
        include_self=False,
        params=AnnoyParams(query_k=3, rerank=True),
    )
    assert not np.any(excluded == np.arange(3, dtype=np.int64)[:, None])


def test_annoy_edges_keep_final_k_after_candidate_search(monkeypatch) -> None:
    X = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
    candidates = {0: [0, 2, 1], 1: [1, 2, 0], 2: [2, 0, 1]}
    _install_fake_annoy(monkeypatch, candidates)

    edge_index, distances = knn_edges_annoy(
        X,
        k=2,
        metric="euclidean",
        include_self=True,
        params=AnnoyParams(query_k=3, rerank=True),
    )

    assert edge_index.shape == (2, 6)
    np.testing.assert_array_equal(edge_index[0], [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(edge_index[1], [0, 1, 1, 2, 2, 1])
    np.testing.assert_allclose(distances, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


def test_annoy_validates_metric_search_and_features(monkeypatch) -> None:
    X = np.asarray([[0.0], [1.0]], dtype=np.float32)
    _install_fake_annoy(monkeypatch, {0: [0, 1], 1: [1, 0]})

    with pytest.raises(GraphValidationError, match="requires metric='euclidean'"):
        knn_search_annoy(X, k=1, metric="cosine")  # type: ignore[arg-type]
    with pytest.raises(GraphValidationError, match="positive integer"):
        knn_search_annoy(X, k=0, metric="euclidean")
    with pytest.raises(GraphValidationError, match="2D array"):
        knn_search_annoy(np.asarray([0.0, 1.0]), k=1, metric="euclidean")
    with pytest.raises(GraphValidationError, match="feature column"):
        knn_search_annoy(np.empty((2, 0)), k=1, metric="euclidean")
    with pytest.raises(GraphValidationError, match="n_trees"):
        knn_search_annoy(
            X,
            k=1,
            metric="euclidean",
            params=AnnoyParams(n_trees=0),
        )
    with pytest.raises(GraphValidationError, match="search_k"):
        knn_search_annoy(
            X,
            k=1,
            metric="euclidean",
            params=AnnoyParams(search_k=0),
        )
    with pytest.raises(GraphValidationError, match="query_k"):
        knn_search_annoy(
            X,
            k=1,
            metric="euclidean",
            include_self=False,
            params=AnnoyParams(query_k=1),
        )
    with pytest.raises(GraphValidationError, match="finite"):
        knn_search_annoy(
            np.asarray([[np.nan], [1.0]]),
            k=1,
            metric="euclidean",
        )


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (([[0]], [0.0]), "invalid neighbor list"),
        (([0], [0.0, 1.0]), "invalid neighbor distances"),
        (([-1], [0.0]), "out-of-range"),
        (([2], [0.0]), "out-of-range"),
        (([0], [np.inf]), "invalid neighbor distances"),
        (([0], [-1.0]), "invalid neighbor distances"),
    ],
)
def test_annoy_rejects_invalid_index_results(monkeypatch, result, message) -> None:
    _install_raw_result_annoy(monkeypatch, result)

    with pytest.raises(GraphValidationError, match=message):
        knn_search_annoy(
            np.asarray([[0.0], [1.0]], dtype=np.float32),
            k=1,
            metric="euclidean",
            include_self=True,
            params=AnnoyParams(query_k=1),
        )


def test_annoy_handles_default_search_noncontiguous_input_and_nonleading_self(
    monkeypatch,
) -> None:
    contiguous = np.arange(8, dtype=np.float32).reshape(4, 2)
    X = contiguous[:, ::-1]
    assert not X.flags["C_CONTIGUOUS"]
    candidates = {row: [(row + 1) % 4, row, (row + 2) % 4, (row + 3) % 4] for row in range(4)}
    built = _install_fake_annoy(monkeypatch, candidates)

    indices, _ = knn_search_annoy(
        X,
        k=2,
        metric="euclidean",
        include_self=True,
        params=AnnoyParams(query_k=4, search_k=1),
    )

    np.testing.assert_array_equal(indices[:, 0], np.arange(4))
    assert all(item.flags["C_CONTIGUOUS"] for item in built[0].items.values())

    default_candidates = {
        row: [row] + [other for other in range(4) if other != row] for row in range(4)
    }
    _install_fake_annoy(monkeypatch, default_candidates)
    default_indices, _ = knn_search_annoy(
        contiguous,
        k=1,
        metric="euclidean",
        include_self=False,
    )
    assert default_indices.shape == (4, 1)


def test_annoy_rejects_short_results_and_supports_zero_width_output(monkeypatch) -> None:
    X = np.asarray([[0.0], [1.0]], dtype=np.float32)
    _install_fake_annoy(monkeypatch, {0: [0], 1: [1]})

    with pytest.raises(GraphValidationError, match="only 0 usable neighbors"):
        knn_search_annoy(
            X,
            k=1,
            metric="euclidean",
            include_self=False,
            params=AnnoyParams(query_k=2),
        )

    _install_fake_annoy(monkeypatch, {0: [0]})
    indices, distances = knn_search_annoy(
        X[:1],
        k=1,
        metric="euclidean",
        include_self=False,
    )
    assert indices.shape == (1, 0)
    assert distances.shape == (1, 0)


def test_annoy_empty_input_does_not_import_dependency(monkeypatch) -> None:
    monkeypatch.setattr(
        "modssc.graph.construction.backends.annoy_backend.optional_import",
        lambda *_args, **_kwargs: pytest.fail("empty input must not import Annoy"),
    )

    indices, distances = knn_search_annoy(
        np.empty((0, 2), dtype=np.float32),
        k=3,
        metric="euclidean",
        include_self=True,
    )

    assert indices.shape == (0, 0)
    assert distances.shape == (0, 0)


def test_annoy_graph_cache_reuses_one_build_and_binds_seed_and_search_spec(
    monkeypatch, tmp_path
) -> None:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    candidates = {
        row: [row] + [other for other in range(X.shape[0]) if other != row]
        for row in range(X.shape[0])
    }
    built = _install_fake_annoy(monkeypatch, candidates)
    common = {
        "scheme": "knn",
        "metric": "euclidean",
        "k": 2,
        "include_self_in_knn": True,
        "backend": "annoy",
        "annoy_n_trees": 10,
        "annoy_query_k": 3,
        "annoy_search_k": -1,
        "annoy_rerank": False,
        "symmetrize": "none",
        "self_loops": False,
        "normalize": "none",
        "weights": GraphWeightsSpec(kind="binary"),
    }
    spec = GraphBuilderSpec(**common)

    first = build_graph(X, spec=spec, seed=7, cache=True, cache_dir=tmp_path)
    reused = build_graph(X, spec=spec, seed=7, cache=True, cache_dir=tmp_path)

    assert reused.meta["fingerprint"] == first.meta["fingerprint"]
    assert len(built) == 1

    changed_seed = build_graph(X, spec=spec, seed=8, cache=True, cache_dir=tmp_path)
    changed_search = build_graph(
        X,
        spec=GraphBuilderSpec(**{**common, "annoy_query_k": 4}),
        seed=7,
        cache=True,
        cache_dir=tmp_path,
    )

    assert changed_seed.meta["fingerprint"] != first.meta["fingerprint"]
    assert changed_search.meta["fingerprint"] != first.meta["fingerprint"]
    assert len(built) == 3
