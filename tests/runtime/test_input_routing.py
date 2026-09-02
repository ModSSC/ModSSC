from __future__ import annotations

import numpy as np
import pytest

from modssc.data_augmentation import UnlabeledAugmentationResult
from modssc.data_loader.types import LoadedDataset, Split
from modssc.graph.artifacts import GraphArtifact
from modssc.inductive import InductiveExecutionInput
from modssc.preprocess.store import ArtifactStore
from modssc.preprocess.types import PreprocessResult, ResolvedPlan
from modssc.runtime.execution import ExecutionContext, RunIdentity
from modssc.runtime.input_routing import (
    InputRoutingError,
    ScientificInputRequest,
    route_scientific_input,
)
from modssc.sampling import InductiveGraphSamplingPolicy, SamplingResult
from modssc.transductive import TransductiveExecutionInput


def _preprocess() -> PreprocessResult:
    return PreprocessResult(
        dataset=LoadedDataset(
            train=Split(
                X=np.arange(10, dtype=np.float32).reshape(5, 2),
                y=np.array([0, 1, 0, 1, 0], dtype=np.int64),
            ),
            meta={"dataset_fingerprint": "dataset", "modality": "graph"},
        ),
        plan=ResolvedPlan(steps=()),
        preprocess_fingerprint="preprocess",
        train_artifacts=ArtifactStore(),
    )


def _sampling() -> SamplingResult:
    return SamplingResult(
        schema_version=1,
        created_at="",
        dataset_fingerprint="dataset",
        split_fingerprint="split",
        plan={},
        masks={
            "train": np.array([True, True, True, False, False]),
            "val": np.array([False, False, False, True, False]),
            "test": np.array([False, False, False, False, True]),
            "labeled": np.array([True, False, False, False, False]),
            "unlabeled": np.array([False, True, True, False, False]),
        },
        stats={"labeled": 1},
    )


def _graph() -> GraphArtifact:
    return GraphArtifact(
        n_nodes=5,
        edge_index=np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
        meta={"fingerprint": "graph"},
    )


def test_configured_transductive_augmentation_is_an_explicit_native_error() -> None:
    with pytest.raises(InputRoutingError) as caught:
        route_scientific_input(
            ScientificInputRequest(
                regime="transductive",
                preprocess=_preprocess(),
                sampling=_sampling(),
                graph=_graph(),
                augmentation_configured=True,
            )
        )

    assert caught.value.kind == "augmentation_regime"
    assert caught.value.code == "E_INPUT_AUGMENTATION_REGIME"


def test_inductive_graph_sampling_and_graph_delivery_are_native_and_traced() -> None:
    augmentation = UnlabeledAugmentationResult(
        weak=np.ones((2, 2), dtype=np.float32),
        strong=np.ones((2, 2), dtype=np.float32) * 2,
        second_strong=None,
        online=None,
        sample_ids=np.array([1, 2], dtype=np.int64),
    )
    routed = route_scientific_input(
        ScientificInputRequest(
            regime="inductive",
            preprocess=_preprocess(),
            sampling=_sampling(),
            graph=_graph(),
            augmentation=augmentation,
            inductive_graph_policy=InductiveGraphSamplingPolicy.MASKS_TO_INDICES,
        )
    )

    assert isinstance(routed.execution_input, InductiveExecutionInput)
    assert routed.execution_input.graph is not None
    assert not routed.sampling.is_graph()
    assert [event["code"] for event in routed.events] == [
        "sampling.graph_masks_to_inductive_indices",
        "augmentation.inductive_delivered",
        "graph.inductive_delivered",
    ]
    assert routed.to_dict()["augmentation_delivered"] is True


def test_inductive_configured_augmentation_cannot_silently_disappear() -> None:
    with pytest.raises(InputRoutingError) as caught:
        route_scientific_input(
            ScientificInputRequest(
                regime="inductive",
                preprocess=_preprocess(),
                sampling=_sampling().as_inductive_indices(),
                augmentation_configured=True,
            )
        )

    assert caught.value.kind == "augmentation_missing"


def test_transductive_route_builds_the_exact_native_mask_input(tmp_path) -> None:
    context = ExecutionContext(
        identity=RunIdentity(config_sha256="0" * 64, seed=3),
        output_dir=tmp_path,
        resume_policy="auto",
    )
    routed = route_scientific_input(
        ScientificInputRequest(
            regime="transductive",
            preprocess=_preprocess(),
            sampling=_sampling(),
            graph=_graph(),
            execution_context=context,
        )
    )

    assert isinstance(routed.execution_input, TransductiveExecutionInput)
    assert routed.masks is not None
    np.testing.assert_array_equal(
        routed.execution_input.masks["labeled_mask"],
        [True, False, False, False, False],
    )
    assert routed.expected_labeled_count == 1
    assert routed.execution_input.execution_context is context
