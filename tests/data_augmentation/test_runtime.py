from __future__ import annotations

import numpy as np
import pytest

from modssc.data_augmentation import (
    AugmentationPlan,
    OnlineAugmentation,
    StepConfig,
)
from modssc.data_augmentation.api import build_strategy


def _runtime() -> OnlineAugmentation:
    weak = AugmentationPlan(
        steps=(StepConfig(op_id="core.identity", params={}),),
        modality="tabular",
    )
    strong = AugmentationPlan(
        steps=(StepConfig(op_id="tabular.gaussian_noise", params={"std": 0.5}),),
        modality="tabular",
    )
    return OnlineAugmentation(
        strategy=build_strategy(weak=weak, strong=strong),
        seed=17,
        modality="tabular",
    )


def test_online_augmentation_is_replayable_per_step_and_sample() -> None:
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    runtime = _runtime()
    indices = np.array([4, 1, 5], dtype=np.int64)
    sample_ids = np.array([104, 101, 105], dtype=np.int64)

    weak_a, strong_a = runtime.pair_batch(X, indices=indices, sample_ids=sample_ids, step=8)
    weak_b, strong_b = runtime.pair_batch(X, indices=indices, sample_ids=sample_ids, step=8)
    _, strong_next = runtime.pair_batch(X, indices=indices, sample_ids=sample_ids, step=9)

    np.testing.assert_array_equal(weak_a, X[indices])
    np.testing.assert_array_equal(weak_a, weak_b)
    np.testing.assert_array_equal(strong_a, strong_b)
    assert not np.array_equal(strong_a, strong_next)


def test_online_augmentation_weak_batch_uses_absolute_sample_ids() -> None:
    X = np.zeros((3, 5), dtype=np.float32)
    runtime = _runtime()
    first = runtime.weak_batch(
        X,
        indices=np.array([0, 1]),
        sample_ids=np.array([20, 21]),
        step=0,
    )
    np.testing.assert_array_equal(first, np.zeros((2, 5), dtype=np.float32))


def test_online_augmentation_rejects_misaligned_ids_and_negative_step() -> None:
    runtime = _runtime()
    X = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="same length"):
        runtime.pair_batch(
            X,
            indices=np.array([0, 1]),
            sample_ids=np.array([10]),
            step=0,
        )
    with pytest.raises(ValueError, match="step must be"):
        runtime.weak_batch(
            X,
            indices=np.array([0]),
            sample_ids=np.array([10]),
            step=-1,
        )
    with pytest.raises(ValueError, match="same length"):
        runtime.weak_batch(
            X,
            indices=np.array([0, 1]),
            sample_ids=np.array([10]),
            step=0,
        )


def test_online_augmentation_supports_torch_indices_and_empty_batches() -> None:
    torch = pytest.importorskip("torch")
    runtime = _runtime()
    X = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    weak = runtime.weak_batch(
        X,
        indices=torch.tensor([2, 0]),
        sample_ids=torch.tensor([12, 10]),
        step=1,
    )
    empty_weak, empty_strong = runtime.pair_batch(
        X,
        indices=torch.empty(0, dtype=torch.int64),
        sample_ids=torch.empty(0, dtype=torch.int64),
        step=1,
    )

    torch.testing.assert_close(weak, X[torch.tensor([2, 0])])
    assert empty_weak.shape == empty_strong.shape == (0, 4)
