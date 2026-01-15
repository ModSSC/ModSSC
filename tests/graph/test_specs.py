from __future__ import annotations

import numpy as np
import pytest

from modssc.graph.errors import GraphValidationError
from modssc.graph.specs import (
    GraphBuilderSpec,
    GraphFeaturizerSpec,
    GraphWeightsSpec,
)
from modssc.graph.validation import (
    validate_edge_index,
    validate_features,
    validate_view_matrix,
)


def test_weights_validation() -> None:
    GraphWeightsSpec(kind="binary").validate(metric="cosine")
    GraphWeightsSpec(kind="heat", sigma=0.5).validate(metric="euclidean")

    with pytest.raises(GraphValidationError):
        GraphWeightsSpec(kind="heat", sigma=0.0).validate(metric="cosine")

    with pytest.raises(GraphValidationError):
        GraphWeightsSpec(kind="cosine").validate(metric="euclidean")


def test_builder_spec_validation_knn_epsilon_anchor() -> None:
    GraphBuilderSpec(scheme="knn", k=5).validate()
    GraphBuilderSpec(scheme="epsilon", radius=0.3).validate()
    GraphBuilderSpec(scheme="anchor", k=5, n_anchors=10, anchors_k=3, candidate_limit=50).validate()

    with pytest.raises(GraphValidationError):
        GraphBuilderSpec(scheme="knn", k=0).validate()

    with pytest.raises(GraphValidationError):
        GraphBuilderSpec(scheme="epsilon", radius=-1.0).validate()

    with pytest.raises(GraphValidationError):
        GraphBuilderSpec(scheme="anchor", k=5, anchors_k=0).validate()

    with pytest.raises(GraphValidationError):
        GraphBuilderSpec(scheme="epsilon", radius=1.0, backend="faiss").validate()


def test_builder_roundtrip_dict() -> None:
    spec = GraphBuilderSpec(
        scheme="anchor",
        metric="cosine",
        k=12,
        n_anchors=20,
        anchors_k=4,
        candidate_limit=123,
        backend="numpy",
        chunk_size=64,
    )
    d = spec.to_dict()
    spec2 = GraphBuilderSpec.from_dict(d)
    assert spec2.to_dict() == d


def test_featurizer_spec_struct_validation_and_roundtrip() -> None:
    GraphFeaturizerSpec(views=("attr", "diffusion", "struct")).validate()

    with pytest.raises(GraphValidationError):
        GraphFeaturizerSpec(views=()).validate()

    with pytest.raises(GraphValidationError):
        GraphFeaturizerSpec(views=("struct",), struct_dim=0).validate()

    spec = GraphFeaturizerSpec(
        views=("struct",),
        struct_method="node2vec",
        struct_dim=16,
        walk_length=10,
        num_walks_per_node=3,
        window_size=2,
        p=0.5,
        q=2.0,
    )
    d = spec.to_dict()
    spec2 = GraphFeaturizerSpec.from_dict(d)
    assert spec2.to_dict() == d


def test_weights_spec_validation():
    with pytest.raises(GraphValidationError, match="Unknown weight kind"):
        GraphWeightsSpec(kind="invalid").validate(metric="cosine")

    with pytest.raises(GraphValidationError, match="sigma must be > 0"):
        GraphWeightsSpec(kind="heat", sigma=0.0).validate(metric="euclidean")

    with pytest.raises(GraphValidationError, match="cosine weights require metric='cosine'"):
        GraphWeightsSpec(kind="cosine").validate(metric="euclidean")


def test_builder_spec_validation_schemes():
    with pytest.raises(GraphValidationError, match="Unknown metric"):
        GraphBuilderSpec(metric="invalid").validate()

    with pytest.raises(GraphValidationError, match="k must be a positive integer"):
        GraphBuilderSpec(scheme="knn", k=0).validate()

    with pytest.raises(GraphValidationError, match="radius must be > 0"):
        GraphBuilderSpec(scheme="epsilon", radius=0.0).validate()

    with pytest.raises(GraphValidationError, match="k must be a positive integer"):
        GraphBuilderSpec(scheme="anchor", k=0).validate()

    with pytest.raises(GraphValidationError, match="anchors_k must be a positive integer"):
        GraphBuilderSpec(scheme="anchor", anchors_k=0).validate()

    with pytest.raises(GraphValidationError, match="n_anchors must be a positive integer"):
        GraphBuilderSpec(scheme="anchor", n_anchors=0).validate()

    with pytest.raises(GraphValidationError, match="candidate_limit must be > 0"):
        GraphBuilderSpec(scheme="anchor", candidate_limit=0).validate()

    with pytest.raises(GraphValidationError, match="Unknown anchors_method"):
        GraphBuilderSpec(scheme="anchor", anchors_method="invalid").validate()

    with pytest.raises(GraphValidationError, match="Unknown scheme"):
        GraphBuilderSpec(scheme="invalid").validate()


def test_builder_spec_validation_general():
    with pytest.raises(GraphValidationError, match="Unknown symmetrize mode"):
        GraphBuilderSpec(symmetrize="invalid").validate()

    with pytest.raises(GraphValidationError, match="Unknown normalize mode"):
        GraphBuilderSpec(normalize="invalid").validate()

    with pytest.raises(GraphValidationError, match="Unknown backend"):
        GraphBuilderSpec(backend="invalid").validate()

    with pytest.raises(GraphValidationError, match="chunk_size must be > 0"):
        GraphBuilderSpec(chunk_size=0).validate()

    with pytest.raises(GraphValidationError, match="faiss backend does not support epsilon"):
        GraphBuilderSpec(backend="faiss", scheme="epsilon", radius=0.5).validate()

    with pytest.raises(GraphValidationError, match="faiss_hnsw_m must be > 0"):
        GraphBuilderSpec(faiss_hnsw_m=0).validate()
    with pytest.raises(GraphValidationError, match="faiss_ef_search must be > 0"):
        GraphBuilderSpec(faiss_ef_search=0).validate()
    with pytest.raises(GraphValidationError, match="faiss_ef_construction must be > 0"):
        GraphBuilderSpec(faiss_ef_construction=0).validate()


def test_featurizer_spec_validation():
    with pytest.raises(GraphValidationError, match="diffusion_steps must be >= 0"):
        GraphFeaturizerSpec(diffusion_steps=-1).validate()

    with pytest.raises(GraphValidationError, match="diffusion_alpha must be in"):
        GraphFeaturizerSpec(diffusion_alpha=1.5).validate()

    with pytest.raises(GraphValidationError, match="views cannot be empty"):
        GraphFeaturizerSpec(views=()).validate()

    with pytest.raises(GraphValidationError, match="Unknown view"):
        GraphFeaturizerSpec(views=("invalid",)).validate()

    with pytest.raises(GraphValidationError, match="Unknown struct_method"):
        GraphFeaturizerSpec(views=("struct",), struct_method="invalid").validate()

    with pytest.raises(GraphValidationError, match="struct_dim must be > 0"):
        GraphFeaturizerSpec(views=("struct",), struct_dim=0).validate()
    with pytest.raises(GraphValidationError, match="walk_length must be > 1"):
        GraphFeaturizerSpec(views=("struct",), walk_length=1).validate()
    with pytest.raises(GraphValidationError, match="num_walks_per_node must be > 0"):
        GraphFeaturizerSpec(views=("struct",), num_walks_per_node=0).validate()
    with pytest.raises(GraphValidationError, match="window_size must be > 0"):
        GraphFeaturizerSpec(views=("struct",), window_size=0).validate()
    with pytest.raises(GraphValidationError, match="p must be > 0"):
        GraphFeaturizerSpec(views=("struct",), p=0.0).validate()
    with pytest.raises(GraphValidationError, match="q must be > 0"):
        GraphFeaturizerSpec(views=("struct",), q=0.0).validate()


def test_validate_features():
    with pytest.raises(GraphValidationError, match="must have a shape attribute"):
        validate_features("string")

    class BadShape:
        shape = (10,)

    with pytest.raises(GraphValidationError, match="must be a 2D array-like"):
        validate_features(BadShape())

    class NegShape:
        shape = (-1, 10)

    with pytest.raises(GraphValidationError, match="must have non-negative dimensions"):
        validate_features(NegShape())


def test_validate_edge_index():
    n = 10

    with pytest.raises(GraphValidationError, match="edge_index must have shape"):
        validate_edge_index(np.zeros((3, 10)), n_nodes=n)

    with pytest.raises(GraphValidationError, match="edge_index must be integer typed"):
        validate_edge_index(np.zeros((2, 10), dtype=float), n_nodes=n)

    with pytest.raises(GraphValidationError, match="edge_index has out-of-range node ids"):
        validate_edge_index(np.array([[0], [n]]), n_nodes=n)


def test_validate_view_matrix():
    n = 5

    with pytest.raises(GraphValidationError, match="must be 2D"):
        validate_view_matrix(np.zeros(n), n_nodes=n, name="v")

    with pytest.raises(GraphValidationError, match="must have 5 rows"):
        validate_view_matrix(np.zeros((n + 1, 2)), n_nodes=n, name="v")

    with pytest.raises(GraphValidationError, match="contains non-finite values"):
        arr = np.zeros((n, 2))
        arr[0, 0] = np.nan
        validate_view_matrix(arr, n_nodes=n, name="v")


def test_validation_happy_paths():
    class GoodShape:
        shape = (10, 5)

    validate_features(GoodShape())

    n = 10
    ei = np.zeros((2, 5), dtype=int)
    validate_edge_index(ei, n_nodes=n)

    ei_empty = np.zeros((2, 0), dtype=int)
    validate_edge_index(ei_empty, n_nodes=n)

    vm = np.zeros((n, 5))
    validate_view_matrix(vm, n_nodes=n, name="test")


def test_validate_wrappers():
    from modssc.graph.specs import GraphBuilderSpec, GraphFeaturizerSpec
    from modssc.graph.validation import validate_builder_spec, validate_featurizer_spec

    spec = GraphBuilderSpec()
    validate_builder_spec(spec)

    fspec = GraphFeaturizerSpec()
    validate_featurizer_spec(fspec)
