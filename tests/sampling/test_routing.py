from __future__ import annotations

import numpy as np
import pytest

from modssc.sampling import (
    InductiveGraphSamplingPolicy,
    SamplingResult,
    SamplingValidationError,
    route_sampling_for_regime,
)


def _graph_sampling() -> SamplingResult:
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
    )


def test_inductive_graph_mask_routing_is_rejected_by_default() -> None:
    with pytest.raises(SamplingValidationError, match="masks_to_indices"):
        route_sampling_for_regime(_graph_sampling(), regime="inductive")


def test_inductive_graph_mask_conversion_is_explicit_and_traced() -> None:
    routed = route_sampling_for_regime(
        _graph_sampling(),
        regime="inductive",
        inductive_graph_policy=InductiveGraphSamplingPolicy.MASKS_TO_INDICES,
    )

    assert not routed.sampling.is_graph()
    np.testing.assert_array_equal(routed.sampling.indices["train_labeled"], [0])
    np.testing.assert_array_equal(routed.sampling.indices["train_unlabeled"], [1, 2])
    assert routed.events[0].code == "sampling.graph_masks_to_inductive_indices"
    assert routed.events[0].policy == "masks_to_indices"
    assert routed.to_dict()["representation"] == "indices"


def test_transductive_routing_preserves_graph_masks_without_conversion() -> None:
    source = _graph_sampling()
    routed = route_sampling_for_regime(source, regime="transductive")

    assert routed.sampling is source
    assert routed.events == ()
    assert routed.to_dict() == {"representation": "graph_masks", "events": []}


def test_sampling_routing_rejects_invalid_public_arguments() -> None:
    with pytest.raises(TypeError, match="SamplingResult"):
        route_sampling_for_regime(object(), regime="inductive")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="regime"):
        route_sampling_for_regime(_graph_sampling(), regime="hybrid")  # type: ignore[arg-type]
    with pytest.raises(SamplingValidationError, match="must be one of"):
        route_sampling_for_regime(
            _graph_sampling(),
            regime="inductive",
            inductive_graph_policy="silently_convert",
        )
