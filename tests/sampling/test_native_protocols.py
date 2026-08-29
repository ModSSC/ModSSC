from __future__ import annotations

from collections import deque

import numpy as np

from modssc.sampling.api import sample
from modssc.sampling.plan import SamplingPlan
from modssc.sampling.splitters import make_holdout_split
from tests.sampling._stubs import LoadedDataset, Split


def _dataset(labels: np.ndarray, *, with_official_test: bool = True) -> LoadedDataset:
    train = Split(X=np.zeros((labels.size, 1), dtype=np.float32), y=labels)
    test = None
    if with_official_test:
        test = Split(X=np.zeros((7, 1), dtype=np.float32), y=np.arange(7) % 3)
    return LoadedDataset(train=train, test=test, meta={"dataset_fingerprint": "native-protocol"})


def _balanced_stream_reference(labels: np.ndarray) -> np.ndarray:
    classes = int(np.max(labels)) + 1
    class_data = [deque() for _ in range(classes)]
    positions = np.zeros(classes, dtype=np.int64)
    target = np.asarray(
        [np.count_nonzero(labels == label) for label in range(classes)],
        dtype=np.float64,
    )
    target /= target.max()
    ordered: list[int] = []
    for index, label in enumerate(labels):
        class_data[int(label)].append(index)
        while True:
            selected = int(np.argmax(target - positions / max(int(positions.max()), 1)))
            if not class_data[selected]:
                break
            ordered.append(class_data[selected].popleft())
            positions[selected] += 1
    for remaining in class_data:
        ordered.extend(remaining)
    return np.asarray(ordered, dtype=np.int64)


def _legacy_per_class_reference(labels: np.ndarray, *, seed: int, per_class: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    selected: list[np.ndarray] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label).astype(np.int64)
        rng.shuffle(indices)
        selected.append(indices[:per_class])
    return np.sort(np.concatenate(selected))


def test_holdout_supports_exact_ordered_sizes() -> None:
    parts = make_holdout_split(
        n_samples=10,
        y=np.arange(10),
        test_fraction=0.9,
        val_fraction=0.9,
        test_size=2,
        val_size=1,
        stratify=False,
        shuffle=False,
        holdout_from="start",
        rng=np.random.default_rng(99),
    )

    np.testing.assert_array_equal(parts["test"], [0, 1])
    np.testing.assert_array_equal(parts["val"], [2])
    np.testing.assert_array_equal(parts["train"], np.arange(3, 10))


def test_native_torchssl_sampling_matches_legacy_random_state_protocol() -> None:
    labels = np.repeat(np.arange(4), 25)
    plan = SamplingPlan.from_dict(
        {
            "component_seeds": {"labeling": "run"},
            "split": {
                "kind": "holdout",
                "test_fraction": 0.0,
                "val_fraction": 0.0,
                "stratify": False,
                "shuffle": False,
            },
            "labeling": {
                "mode": "per_class",
                "value": 3,
                "strategy": "balanced",
                "min_per_class": 3,
                "selection_order": "permutation",
                "rng_backend": "legacy_random_state",
                "unlabeled_pool": "includes_labeled",
            },
        }
    )

    result, _ = sample(_dataset(labels), plan=plan, seed=2, save=False)

    np.testing.assert_array_equal(
        result.labeled_idx,
        _legacy_per_class_reference(labels, seed=2, per_class=3),
    )
    np.testing.assert_array_equal(result.train_idx, np.arange(labels.size))
    np.testing.assert_array_equal(result.unlabeled_idx, result.train_idx)
    assert result.test_idx.size == 7


def test_native_balanced_stream_replaces_google_split_archive() -> None:
    labels = np.repeat(np.arange(4), 40)
    expected_order = _balanced_stream_reference(labels)
    expected_labeled = _legacy_per_class_reference(labels, seed=1, per_class=3)
    assert expected_order[0] not in expected_labeled
    plan = SamplingPlan.from_dict(
        {
            "component_seeds": {"labeling": "run"},
            "partition": {
                "ordering": "class_balanced_stream",
                "shuffle": False,
            },
            "split": {
                "kind": "holdout",
                "test_fraction": 0.0,
                "val_fraction": 0.0,
                "val_size": 1,
                "stratify": False,
                "shuffle": False,
                "holdout_from": "start",
            },
            "labeling": {
                "mode": "per_class",
                "value": 3,
                "strategy": "balanced",
                "min_per_class": 3,
                "selection_order": "permutation",
                "rng_backend": "legacy_random_state",
                "selection_scope": "partition",
                "unlabeled_pool": "includes_labeled",
            },
        }
    )

    result, _ = sample(_dataset(labels), plan=plan, seed=1, save=False)

    np.testing.assert_array_equal(result.val_idx, expected_order[:1])
    np.testing.assert_array_equal(result.train_idx, expected_order[1:])
    np.testing.assert_array_equal(result.labeled_idx, expected_labeled)
    np.testing.assert_array_equal(result.unlabeled_idx, result.train_idx)
    assert result.schema_version == 2


def test_native_sampling_options_round_trip_without_changing_defaults() -> None:
    plan = SamplingPlan.from_dict(
        {
            "component_seeds": {"labeling": "run"},
            "partition": {"ordering": "class_balanced_stream", "shuffle": False},
            "split": {"val_size": 1, "shuffle": False, "holdout_from": "end"},
            "labeling": {
                "rng_backend": "legacy_random_state",
                "selection_scope": "partition",
                "unlabeled_pool": "includes_labeled",
            },
        }
    )

    assert SamplingPlan.from_dict(plan.as_dict()) == plan
    assert plan.component_seeds.resolve(11)["labeling"] == 11
    assert "partition" not in SamplingPlan().as_dict()
    assert SamplingPlan().fingerprint_schema_version() == 1
