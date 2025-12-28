from __future__ import annotations

import pytest

from modssc.graph.errors import GraphValidationError
from modssc.graph.specs import (
    GraphBuilderSpec,
    GraphFeaturizerSpec,
    GraphWeightsSpec,
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
