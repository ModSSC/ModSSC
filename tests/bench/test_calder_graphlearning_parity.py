from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse

from bench.campaign.protocols.calder.official import OFFICIAL_KNN_SHA256
from modssc.graph import GraphBuilderSpec, GraphWeightsSpec, build_graph
from modssc.transductive.methods.classic.laplace_learning import (
    LaplaceLearningSpec,
    laplace_learning_numpy,
)
from modssc.transductive.methods.pde.poisson_learning import (
    PoissonLearningSpec,
    poisson_learning_numpy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_KNN = REPO_ROOT / "bench/assets/calder2020/protocol_inputs/graph/mnist-vae-knn30.npz"


def _mini_official_knn(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbors = np.array(
        [
            [0, 1, 2],
            [1, 0, 2],
            [2, 1, 3],
            [3, 2, 4],
            [4, 3, 5],
            [5, 4, 3],
        ],
        dtype=np.int64,
    )
    distances = np.array(
        [
            [0.0, 0.4, 0.9],
            [0.0, 0.4, 0.7],
            [0.0, 0.7, 0.8],
            [0.0, 0.5, 0.8],
            [0.0, 0.5, 0.6],
            [0.0, 0.6, 0.9],
        ],
        dtype=np.float64,
    )
    rows = np.broadcast_to(
        np.arange(neighbors.shape[0], dtype=np.int64)[:, None],
        neighbors.shape,
    ).copy()
    np.savez(path, I=rows, J=neighbors, D=distances)
    return rows, neighbors, distances


def _official_weight_matrix(
    rows: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    squared = distances * distances
    epsilon = squared[:, -1] / 4.0
    directed_weights = np.exp(-(squared / epsilon[:, None]))
    matrix = np.zeros((rows.shape[0], rows.shape[0]), dtype=np.float64)
    np.add.at(matrix, (rows.reshape(-1), neighbors.reshape(-1)), directed_weights.reshape(-1))
    return (matrix + matrix.T) / 2.0


def test_full_mnist_graph_matches_the_frozen_calder_csr_exactly() -> None:
    with np.load(OFFICIAL_KNN, allow_pickle=False) as archive:
        rows = np.asarray(archive["I"][:, :10], dtype=np.int64)
        neighbors = np.asarray(archive["J"][:, :10], dtype=np.int64)
        distances = np.asarray(archive["D"][:, :10], dtype=np.float64)

    squared = distances * distances
    epsilon = squared[:, -1] / 4.0
    directed_weights = np.exp(-(squared / epsilon[:, None]))
    expected = sparse.coo_matrix(
        (
            directed_weights.reshape(-1),
            (rows.reshape(-1), neighbors.reshape(-1)),
        ),
        shape=(70_000, 70_000),
        dtype=np.float64,
    ).tocsr()
    expected = ((expected + expected.T) / 2.0).tocsr()
    expected.sort_indices()

    graph = build_graph(
        np.zeros((70_000, 1), dtype=np.float64),
        spec=GraphBuilderSpec(
            scheme="knn",
            metric="euclidean",
            k=10,
            symmetrize="mean",
            weights=GraphWeightsSpec(kind="knn_gaussian"),
            normalize="none",
            self_loops=True,
            include_self_in_knn=True,
            edge_weight_dtype="float64",
            backend="precomputed",
            precomputed_path=str(OFFICIAL_KNN),
            precomputed_sha256=OFFICIAL_KNN_SHA256,
        ),
        seed=1,
        cache=False,
    )
    actual = sparse.coo_matrix(
        (
            graph.edge_weight,
            (graph.edge_index[1], graph.edge_index[0]),
        ),
        shape=(70_000, 70_000),
        dtype=np.float64,
    ).tocsr()
    actual.sort_indices()

    assert expected.nnz == actual.nnz == 984_538
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data, expected.data)


def _dense_weight_matrix(
    n_nodes: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    np.add.at(matrix, (edge_index[1], edge_index[0]), edge_weight)
    return matrix


def _official_conjgrad(matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.zeros_like(rhs)
    residual = rhs - matrix @ x
    direction = residual
    squared = np.sum(residual**2, axis=0)
    error = 1.0
    iteration = 0
    while error > 1.0e-5 and iteration < 100_000:
        iteration += 1
        product = matrix @ direction
        alpha = squared / np.sum(direction * product, axis=0)
        x += alpha * direction
        residual -= alpha * product
        squared_new = np.sum(residual**2, axis=0)
        error = float(np.sqrt(np.sum(squared_new)))
        direction = residual + (squared_new / squared) * direction
        squared = squared_new
    return x, iteration


def _official_laplace(
    weight_matrix: np.ndarray,
    y: np.ndarray,
    labeled_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    laplacian = np.diag(weight_matrix.sum(axis=1)) - weight_matrix
    labeled = np.flatnonzero(labeled_mask)
    unlabeled = np.flatnonzero(~labeled_mask)
    onehot = np.eye(2, dtype=np.float64)[y[labeled]]
    matrix = laplacian[np.ix_(unlabeled, unlabeled)]
    rhs = -laplacian[np.ix_(unlabeled, labeled)] @ onehot
    scale = 1.0 / np.sqrt(np.diag(matrix) + 1.0e-10)
    solved, iterations = _official_conjgrad(
        scale[:, None] * matrix * scale[None, :],
        scale[:, None] * rhs,
    )
    scores = np.ones((y.size, 2), dtype=np.float64)
    scores[unlabeled] = scale[:, None] * solved
    scores[labeled] = onehot
    return scores, iterations


def _official_poisson(
    weight_matrix: np.ndarray,
    y: np.ndarray,
    labeled_mask: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    matrix = weight_matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    labeled = np.flatnonzero(labeled_mask)
    onehot = np.zeros((y.size, 2), dtype=np.float64)
    onehot[labeled] = np.eye(2, dtype=np.float64)[y[labeled]]
    class_fraction = onehot.sum(axis=0) / float(labeled.size)
    source = onehot.copy()
    source[labeled] -= class_fraction

    degree = matrix.sum(axis=1)
    inverse_degree = 1.0 / (degree + 1.0e-10)
    transition = inverse_degree[:, None] * matrix.T
    rhs = inverse_degree[:, None] * source
    scores = np.zeros_like(source)

    mixing = labeled_mask.astype(np.float64)
    mixing /= mixing.sum()
    stationary = degree / degree.sum()
    random_walk = matrix.T * inverse_degree[None, :]
    residual = float(np.max(np.abs(mixing - stationary)))
    iteration = 0
    while (iteration < 50 or residual > 1.0 / y.size) and iteration < 1000:
        scores = rhs + transition @ scores
        mixing = random_walk @ mixing
        residual = float(np.max(np.abs(mixing - stationary)))
        iteration += 1
    scores = scores @ np.diag(1.0 / class_fraction)
    return scores, iteration, residual


def test_calder_graph_laplace_and_poisson_match_archived_formulas(tmp_path) -> None:
    path = tmp_path / "mini-official-knn.npz"
    rows, neighbors, distances = _mini_official_knn(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = GraphBuilderSpec(
        scheme="knn",
        metric="euclidean",
        k=3,
        symmetrize="mean",
        weights=GraphWeightsSpec(kind="knn_gaussian"),
        normalize="none",
        self_loops=True,
        include_self_in_knn=True,
        edge_weight_dtype="float64",
        backend="precomputed",
        precomputed_path=str(path),
        precomputed_sha256=digest,
    )
    graph = build_graph(
        np.zeros((6, 1), dtype=np.float64),
        spec=spec,
        seed=1,
        cache=False,
    )
    expected_weight_matrix = _official_weight_matrix(rows, neighbors, distances)
    actual_weight_matrix = _dense_weight_matrix(
        graph.n_nodes,
        graph.edge_index,
        graph.edge_weight,
    )

    assert graph.edge_weight.dtype == np.float64
    np.testing.assert_allclose(actual_weight_matrix, expected_weight_matrix, rtol=0.0, atol=1e-15)

    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labeled_mask = np.array([True, False, False, False, False, True])
    expected_laplace, expected_laplace_iterations = _official_laplace(
        expected_weight_matrix,
        y,
        labeled_mask,
    )
    actual_laplace = laplace_learning_numpy(
        n_nodes=6,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=LaplaceLearningSpec(
            backend="numpy",
            solver="calder2020_conjugate_gradient",
            cg_tol=1.0e-5,
            cg_max_iter=100_000,
        ),
    )
    assert actual_laplace.F.dtype == np.float64
    assert actual_laplace.n_iter == expected_laplace_iterations
    np.testing.assert_allclose(actual_laplace.F, expected_laplace, rtol=0.0, atol=1e-14)

    expected_poisson, expected_poisson_iterations, expected_residual = _official_poisson(
        expected_weight_matrix,
        y,
        labeled_mask,
    )
    actual_poisson = poisson_learning_numpy(
        n_nodes=6,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        y=y,
        labeled_mask=labeled_mask,
        spec=PoissonLearningSpec(
            backend="numpy",
            solver="paper_iteration",
            center_sources=True,
            balance_scores=True,
            min_iter=50,
            max_iter=1000,
        ),
    )
    assert actual_poisson.F.dtype == np.float64
    assert actual_poisson.n_iter == expected_poisson_iterations
    assert abs(actual_poisson.residual - expected_residual) <= 1.0e-14
    np.testing.assert_allclose(actual_poisson.F, expected_poisson, rtol=0.0, atol=1e-13)
