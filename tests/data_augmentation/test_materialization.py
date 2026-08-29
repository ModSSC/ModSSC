from __future__ import annotations

import numpy as np
import pytest

from modssc.data_augmentation import prepare_unlabeled_augmentation
from modssc.data_augmentation.errors import DataAugmentationValidationError

_IDENTITY = {"steps": []}


def test_unlabeled_materialization_selects_stable_absolute_ids() -> None:
    result = prepare_unlabeled_augmentation(
        np.arange(20, dtype=np.float32).reshape(10, 2),
        unlabeled_indices=np.array([7, 3], dtype=np.int64),
        weak_plan=_IDENTITY,
        strong_plan=_IDENTITY,
        seed=5,
        mode="fixed",
        modality="tabular",
        strong_views=2,
    )

    expected = np.array([[14.0, 15.0], [6.0, 7.0]], dtype=np.float32)
    np.testing.assert_array_equal(result.weak, expected)
    np.testing.assert_array_equal(result.strong, expected)
    np.testing.assert_array_equal(result.second_strong, expected)
    np.testing.assert_array_equal(result.sample_ids, [7, 3])
    assert result.online is None


def test_unlabeled_materialization_preserves_structured_graph_fields() -> None:
    payload = {
        "x": np.arange(12, dtype=np.float32).reshape(6, 2),
        "edge_index": np.array([[1, 4, 4], [4, 1, 5]], dtype=np.int64),
        "edge_weight": np.array([0.2, 0.3, 0.4], dtype=np.float32),
        "description": "metadata",
    }
    result = prepare_unlabeled_augmentation(
        payload,
        unlabeled_indices=np.array([4, 1], dtype=np.int64),
        weak_plan=_IDENTITY,
        strong_plan=_IDENTITY,
        seed=5,
        mode="fixed",
        modality="tabular",
    )

    assert result.weak["description"] == "metadata"
    np.testing.assert_array_equal(result.weak["x"], payload["x"][[4, 1]])
    np.testing.assert_array_equal(result.weak["edge_index"], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(
        result.weak["edge_weight"],
        np.array([0.2, 0.3], dtype=np.float32),
    )


def test_online_materialization_rejects_two_strong_views_natively() -> None:
    with pytest.raises(DataAugmentationValidationError, match="exactly one strong view"):
        prepare_unlabeled_augmentation(
            np.zeros((3, 2), dtype=np.float32),
            unlabeled_indices=np.array([0, 1], dtype=np.int64),
            weak_plan=_IDENTITY,
            strong_plan=_IDENTITY,
            seed=5,
            mode="online",
            modality="tabular",
            strong_views=2,
        )
