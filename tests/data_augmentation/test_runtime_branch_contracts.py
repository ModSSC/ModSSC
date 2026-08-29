from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from modssc.data_augmentation import materialize_views, prepare_unlabeled_augmentation
from modssc.data_augmentation.errors import DataAugmentationValidationError
from modssc.data_augmentation.runtime import _is_graph_like_sample, validate_augmentation_regime

_IDENTITY = {"steps": []}


def test_augmentation_regime_validation_covers_invalid_and_transductive_requests() -> None:
    with pytest.raises(DataAugmentationValidationError, match="regime must"):
        validate_augmentation_regime(regime="semi", configured=False)
    validate_augmentation_regime(regime="transductive", configured=False)
    with pytest.raises(DataAugmentationValidationError, match="transductive"):
        validate_augmentation_regime(regime="transductive", configured=True)


def test_graph_like_sample_accepts_attribute_protocol() -> None:
    assert _is_graph_like_sample(SimpleNamespace(x=object(), edge_index=object()))


@pytest.mark.parametrize("sample_ids", [None, np.array([42], dtype=np.int64)])
def test_graph_materialization_handles_default_and_numpy_ids(sample_ids) -> None:
    graph = {
        "x": np.ones((2, 2), dtype=np.float32),
        "edge_index": np.array([[0], [1]], dtype=np.int64),
    }
    weak, strong, second = materialize_views(
        graph,
        weak_plan=_IDENTITY,
        strong_plan=_IDENTITY,
        seed=3,
        modality="graph",
        sample_ids=sample_ids,
        strong_views=2,
    )

    assert weak is graph
    assert strong is graph
    assert second is graph


def test_prepare_unlabeled_online_builds_native_runtime() -> None:
    result = prepare_unlabeled_augmentation(
        np.arange(12, dtype=np.float32).reshape(6, 2),
        unlabeled_indices=np.array([4, 1], dtype=np.int64),
        weak_plan=_IDENTITY,
        strong_plan=_IDENTITY,
        seed=5,
        mode="online",
        modality="tabular",
    )

    assert result.online is not None
    np.testing.assert_array_equal(result.weak, [[8.0, 9.0], [2.0, 3.0]])
    assert result.strong is result.weak
    assert result.second_strong is None
