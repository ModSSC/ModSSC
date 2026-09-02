from __future__ import annotations

import numpy as np
import pytest

import modssc.graph.construction.api as graph_api
from modssc.graph import GraphBuilderSpec, GraphWeightsSpec, build_graph
from modssc.graph.construction.ops.diagonal import zero_diagonal_edges
from modssc.graph.construction.ops.symmetrize import symmetrize_edges
from modssc.graph.errors import GraphValidationError


def _dense(
    *,
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((n_nodes, n_nodes), dtype=edge_weight.dtype)
    matrix[edge_index[0], edge_index[1]] = edge_weight
    return matrix


def test_sum_symmetrization_is_exact_w_plus_w_transpose() -> None:
    edge_index = np.array(
        [[0, 1, 1, 2], [1, 0, 2, 2]],
        dtype=np.int64,
    )
    edge_weight = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    directed = _dense(n_nodes=3, edge_index=edge_index, edge_weight=edge_weight)

    result_index, result_weight = symmetrize_edges(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mode="sum",
    )

    assert result_weight is not None
    result = _dense(n_nodes=3, edge_index=result_index, edge_weight=result_weight)
    np.testing.assert_array_equal(result, directed + directed.T)
    assert result[0, 1] == 5.0  # an already symmetric pair is intentionally summed
    assert result[2, 2] == 10.0  # W[i, i] + W.T[i, i]
    assert result_weight.dtype == np.float64


def test_sum_symmetrization_materializes_implicit_unit_weights() -> None:
    edge_index = np.array([[0, 1, 1], [1, 0, 2]], dtype=np.int64)

    result_index, result_weight = symmetrize_edges(
        n_nodes=3,
        edge_index=edge_index,
        edge_weight=None,
        mode="sum",
    )

    assert result_weight is not None
    result = _dense(n_nodes=3, edge_index=result_index, edge_weight=result_weight)
    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_sum_symmetrization_empty_input_has_explicit_empty_weights() -> None:
    result_index, result_weight = symmetrize_edges(
        n_nodes=0,
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_weight=None,
        mode="sum",
    )

    assert result_index.shape == (2, 0)
    assert result_weight is not None
    assert result_weight.shape == (0,)
    assert result_weight.dtype == np.float32


def test_zero_diagonal_removes_only_explicit_self_edges() -> None:
    edge_index = np.array([[0, 0, 1, 2], [0, 1, 1, 0]], dtype=np.int64)
    edge_weight = np.array([7.0, 2.0, 8.0, 3.0], dtype=np.float32)

    result_index, result_weight = zero_diagonal_edges(
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

    np.testing.assert_array_equal(result_index, np.array([[0, 2], [1, 0]], dtype=np.int64))
    assert result_weight is not None
    np.testing.assert_array_equal(result_weight, np.array([2.0, 3.0], dtype=np.float32))


def test_zero_diagonal_preserves_unweighted_edges_and_fast_path() -> None:
    without_loop = np.array([[0, 1], [1, 0]], dtype=np.int64)
    same_index, same_weight = zero_diagonal_edges(
        edge_index=without_loop,
        edge_weight=None,
    )
    assert same_index is without_loop
    assert same_weight is None

    with_loop = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int64)
    result_index, result_weight = zero_diagonal_edges(
        edge_index=with_loop,
        edge_weight=None,
    )
    np.testing.assert_array_equal(result_index, np.array([[0], [1]], dtype=np.int64))
    assert result_weight is None


def test_zero_diagonal_rejects_inconsistent_sparse_arrays() -> None:
    with pytest.raises(ValueError, match=r"shape \(2, E\)"):
        zero_diagonal_edges(
            edge_index=np.zeros((3, 1), dtype=np.int64),
            edge_weight=None,
        )

    edge_index = np.array([[0, 1], [0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="one-dimensional"):
        zero_diagonal_edges(
            edge_index=edge_index,
            edge_weight=np.ones((2, 1), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="one value per edge"):
        zero_diagonal_edges(
            edge_index=edge_index,
            edge_weight=np.ones((1,), dtype=np.float32),
        )


def test_graph_spec_supports_sum_and_explicit_zero_diagonal() -> None:
    spec = GraphBuilderSpec(
        symmetrize="sum",
        self_loops=False,
        diagonal_policy="zero",
    )

    spec.validate()
    assert GraphBuilderSpec.from_dict(spec.to_dict()) == spec
    assert spec.to_dict()["diagonal_policy"] == "zero"
    assert "diagonal_policy" not in GraphBuilderSpec().to_dict()

    with pytest.raises(GraphValidationError, match="conflicts with diagonal_policy='zero'"):
        GraphBuilderSpec(self_loops=True, diagonal_policy="zero").validate()
    with pytest.raises(GraphValidationError, match="Unknown diagonal_policy"):
        GraphBuilderSpec(
            self_loops=False,
            diagonal_policy="invalid",  # type: ignore[arg-type]
        ).validate()


def test_build_graph_applies_sum_then_zero_diagonal(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_index = np.array([[0, 0, 1, 1], [0, 1, 0, 2]], dtype=np.int64)
    raw_distances = np.zeros(raw_index.shape[1], dtype=np.float32)
    monkeypatch.setattr(
        graph_api,
        "build_raw_edges",
        lambda *args, **kwargs: (raw_index, raw_distances),
    )
    spec = GraphBuilderSpec(
        scheme="knn",
        metric="euclidean",
        k=2,
        backend="numpy",
        weights=GraphWeightsSpec(kind="binary"),
        symmetrize="sum",
        self_loops=False,
        diagonal_policy="zero",
        normalize="none",
    )

    graph = build_graph(
        np.zeros((3, 1), dtype=np.float32),
        spec=spec,
        cache=False,
        resume=False,
    )

    assert graph.edge_weight is not None
    result = _dense(
        n_nodes=graph.n_nodes,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
    )
    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
