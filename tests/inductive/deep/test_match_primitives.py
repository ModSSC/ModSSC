from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from modssc.inductive.deep.match_primitives import (
    FixedSSLBatchSampler,
    deinterleave_batch,
    interleave_batch,
)
from modssc.inductive.errors import InductiveValidationError


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_google_interleave_oracle_and_inverse(backend: str) -> None:
    # Google libml/utils.py reshapes [-1, groups], transposes, then flattens.
    value = np.arange(30, dtype=np.int64).reshape(30, 1)
    x = value if backend == "numpy" else torch.from_numpy(value)
    output = interleave_batch(x, 15)
    expected = np.array(
        [
            0,
            15,
            1,
            16,
            2,
            17,
            3,
            18,
            4,
            19,
            5,
            20,
            6,
            21,
            7,
            22,
            8,
            23,
            9,
            24,
            10,
            25,
            11,
            26,
            12,
            27,
            13,
            28,
            14,
            29,
        ],
        dtype=np.int64,
    ).reshape(30, 1)
    actual = output if isinstance(output, np.ndarray) else output.numpy()
    np.testing.assert_array_equal(actual, expected)
    restored = deinterleave_batch(output, 15)
    restored_array = restored if isinstance(restored, np.ndarray) else restored.numpy()
    np.testing.assert_array_equal(restored_array, value)


@pytest.mark.parametrize(
    ("value", "groups", "message"),
    [
        (np.array(1), 1, "batch dimension"),
        (np.zeros((3, 2)), 0, "positive integer"),
        (np.zeros((3, 2)), True, "positive integer"),
        (np.zeros((5, 2)), 2, "divisible"),
    ],
)
def test_interleave_validation(value: np.ndarray, groups: int, message: str) -> None:
    with pytest.raises(InductiveValidationError, match=message):
        interleave_batch(value, groups)
    with pytest.raises(InductiveValidationError, match=message):
        deinterleave_batch(value, groups)


def test_replacement_sampler_has_fixed_batches_independent_streams_and_resume() -> None:
    sampler = FixedSSLBatchSampler(7, 11, seed=19)
    assert iter(sampler) is sampler
    first = sampler.next_batch()
    assert first.labeled.shape == (64,)
    assert first.unlabeled.shape == (448,)
    assert first.labeled.dtype == first.unlabeled.dtype == np.int64
    assert int(first.labeled.min()) >= 0 and int(first.labeled.max()) < 7
    assert int(first.unlabeled.min()) >= 0 and int(first.unlabeled.max()) < 11

    state = copy.deepcopy(sampler.state_dict())
    expected = sampler.next_batch()
    resumed = FixedSSLBatchSampler(7, 11, seed=19)
    resumed.load_state_dict(state)
    actual = next(resumed)
    np.testing.assert_array_equal(actual.labeled, expected.labeled)
    np.testing.assert_array_equal(actual.unlabeled, expected.unlabeled)


def test_torchssl_replacement_stream_is_exact_random_sampler_oracle() -> None:
    """Match the pinned TorchSSL ``RandomSampler`` index algorithm exactly."""

    sampler = FixedSSLBatchSampler(
        7,
        11,
        labeled_batch_size=64,
        unlabeled_batch_size=448,
        seed=19,
        mode="replacement",
    )
    initial = copy.deepcopy(sampler.state_dict())

    def reference_indices(stream: str, *, size: int, count: int) -> np.ndarray:
        generator = torch.Generator(device="cpu")
        generator.set_state(initial[stream]["rng_state"])
        random_sampler = torch.utils.data.RandomSampler(
            range(size),
            replacement=True,
            num_samples=count,
            generator=generator,
        )
        return np.asarray(list(random_sampler), dtype=np.int64)

    expected_labeled = reference_indices("labeled", size=7, count=3 * 64)
    expected_unlabeled = reference_indices("unlabeled", size=11, count=3 * 448)
    actual = [sampler.next_batch() for _ in range(3)]
    np.testing.assert_array_equal(
        np.concatenate([batch.labeled for batch in actual]),
        expected_labeled,
    )
    np.testing.assert_array_equal(
        np.concatenate([batch.unlabeled for batch in actual]),
        expected_unlabeled,
    )
    assert initial["labeled"]["rng_backend"] == "torch_cpu"
    assert initial["unlabeled"]["rng_backend"] == "torch_cpu"


def test_google_shuffle_repeat_matches_repeat_buffer_transition_oracle() -> None:
    """Exercise the TF1 ``repeat().shuffle(buffer)`` state transition."""

    sampler = FixedSSLBatchSampler(
        3,
        5,
        labeled_batch_size=5,
        unlabeled_batch_size=9,
        seed=23,
        mode="shuffle_repeat",
        shuffle_buffer=8,
    )
    initial = copy.deepcopy(sampler.state_dict())

    def reference_indices(stream: str, *, size: int, count: int) -> np.ndarray:
        state = initial[stream]
        generator = np.random.Generator(np.random.PCG64())
        generator.bit_generator.state = copy.deepcopy(state["rng_state"])
        buffer = np.asarray(state["buffer"], dtype=np.int64).copy()
        cursor = int(state["cursor"])
        result = np.empty(count, dtype=np.int64)
        for position in range(count):
            slot = int(generator.integers(0, buffer.size, dtype=np.int64))
            result[position] = buffer[slot]
            buffer[slot] = cursor
            cursor = (cursor + 1) % size
        return result

    expected_labeled = reference_indices("labeled", size=3, count=4 * 5)
    expected_unlabeled = reference_indices("unlabeled", size=5, count=4 * 9)
    actual = [sampler.next_batch() for _ in range(4)]
    np.testing.assert_array_equal(
        np.concatenate([batch.labeled for batch in actual]),
        expected_labeled,
    )
    np.testing.assert_array_equal(
        np.concatenate([batch.unlabeled for batch in actual]),
        expected_unlabeled,
    )
    assert initial["labeled"]["rng_backend"] == "numpy_pcg64"
    assert initial["unlabeled"]["rng_backend"] == "numpy_pcg64"


def test_shuffle_repeat_state_restores_live_buffer_cursor_and_generator() -> None:
    sampler = FixedSSLBatchSampler(
        3,
        5,
        labeled_batch_size=4,
        unlabeled_batch_size=7,
        seed=23,
        mode="shuffle_repeat",
        shuffle_buffer=8,
    )
    next(sampler)
    state = copy.deepcopy(sampler.state_dict())
    assert state["labeled"]["buffer"].shape == (8,)
    assert state["labeled"]["cursor"] in range(3)

    expected = [next(sampler) for _ in range(3)]
    resumed = FixedSSLBatchSampler(
        3,
        5,
        labeled_batch_size=4,
        unlabeled_batch_size=7,
        seed=23,
        mode="shuffle_repeat",
        shuffle_buffer=8,
    )
    resumed.load_state_dict(state)
    actual = [next(resumed) for _ in range(3)]
    for expected_batch, actual_batch in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(actual_batch.labeled, expected_batch.labeled)
        np.testing.assert_array_equal(actual_batch.unlabeled, expected_batch.unlabeled)


def test_sampler_modes_select_reference_semantics() -> None:
    shuffled = FixedSSLBatchSampler(
        250,
        50_000,
        mode="shuffle_repeat",
    )
    replacement = FixedSSLBatchSampler(
        250,
        50_000,
        mode="replacement",
    )
    assert shuffled.mode == "shuffle_repeat"
    assert replacement.mode == "replacement"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"labeled_size": 0, "unlabeled_size": 1}, "labeled_size"),
        ({"labeled_size": 1, "unlabeled_size": -1}, "unlabeled_size"),
        ({"labeled_size": 1, "unlabeled_size": 1, "labeled_batch_size": 0}, "labeled_batch"),
        ({"labeled_size": 1, "unlabeled_size": 1, "unlabeled_batch_size": 0}, "unlabeled_batch"),
        ({"labeled_size": 1, "unlabeled_size": 1, "shuffle_buffer": 0}, "shuffle_buffer"),
        ({"labeled_size": 1, "unlabeled_size": 1, "mode": "bad"}, "mode must"),
        ({"labeled_size": 1, "unlabeled_size": 1, "seed": True}, "seed must"),
    ],
)
def test_sampler_constructor_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(InductiveValidationError, match=message):
        FixedSSLBatchSampler(**kwargs)


def test_sampler_restore_validation_is_atomic() -> None:
    sampler = FixedSSLBatchSampler(
        3,
        5,
        labeled_batch_size=2,
        unlabeled_batch_size=3,
        seed=4,
        mode="shuffle_repeat",
        shuffle_buffer=4,
    )
    good = copy.deepcopy(sampler.state_dict())
    expected = next(sampler)

    for mutate in (
        lambda state: state.update(version=999),
        lambda state: state["config"].update(seed=99),
        lambda state: state.update(batches_yielded=-1),
        lambda state: state.update(labeled=[]),
        lambda state: state["labeled"].update(size=99),
        lambda state: state["labeled"].update(cursor=99),
        lambda state: state["labeled"].update(draws=-1),
        lambda state: state["labeled"].update(buffer=np.array([0])),
        lambda state: state["labeled"].update(buffer=np.array([0, 1, 2, 9])),
        lambda state: state["labeled"].update(rng_state={"bad": True}),
    ):
        bad = copy.deepcopy(good)
        mutate(bad)
        fresh = FixedSSLBatchSampler(
            3,
            5,
            labeled_batch_size=2,
            unlabeled_batch_size=3,
            seed=4,
            mode="shuffle_repeat",
            shuffle_buffer=4,
        )
        with pytest.raises(InductiveValidationError):
            fresh.load_state_dict(bad)

    resumed = FixedSSLBatchSampler(
        3,
        5,
        labeled_batch_size=2,
        unlabeled_batch_size=3,
        seed=4,
        mode="shuffle_repeat",
        shuffle_buffer=4,
    )
    resumed.load_state_dict(good)
    actual = next(resumed)
    np.testing.assert_array_equal(actual.labeled, expected.labeled)
    np.testing.assert_array_equal(actual.unlabeled, expected.unlabeled)


def test_replacement_restore_rejects_buffer() -> None:
    sampler = FixedSSLBatchSampler(2, 2, labeled_batch_size=1, unlabeled_batch_size=1)
    state = copy.deepcopy(sampler.state_dict())
    state["labeled"]["buffer"] = np.array([0], dtype=np.int64)
    with pytest.raises(InductiveValidationError, match="must not contain"):
        sampler.load_state_dict(state)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state["labeled"].update(rng_backend="numpy_pcg64"),
        lambda state: state["labeled"].update(rng_state=np.array([1, 2, 3], dtype=np.uint8)),
        lambda state: state["labeled"].update(rng_state=torch.ones(3, dtype=torch.uint8)),
        lambda state: state["labeled"].update(rng_state=torch.ones(3, dtype=torch.int64)),
        lambda state: state["labeled"].update(rng_state=torch.ones((2, 3), dtype=torch.uint8)),
    ],
)
def test_replacement_restore_rejects_incompatible_torch_rng_state(
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    sampler = FixedSSLBatchSampler(2, 2, labeled_batch_size=1, unlabeled_batch_size=1)
    state = copy.deepcopy(sampler.state_dict())
    mutation(state)
    with pytest.raises(InductiveValidationError, match="RNG"):
        sampler.load_state_dict(state)
