from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from modssc.sampling.api import sample
from modssc.sampling.errors import SamplingError
from modssc.sampling.fingerprint import stable_hash
from modssc.sampling.plan import (
    FixedIndicesArtifactSpec,
    HoldoutSplitSpec,
    KFoldSplitSpec,
    LabelingSpec,
    PartitionSpec,
    SamplingComponentSeeds,
    SamplingPlan,
)
from modssc.sampling.storage import load_split
from tests.sampling._stubs import make_toy_dataset


def test_sample_holdout_without_official_test(tmp_path) -> None:
    ds = make_toy_dataset(n=100, with_test=False)
    plan = SamplingPlan(
        split=HoldoutSplitSpec(test_fraction=0.2, val_fraction=0.1, stratify=True),
        labeling=LabelingSpec(mode="fraction", value=0.1, per_class=True),
    )
    res, path = sample(
        ds,
        plan=plan,
        seed=0,
        dataset_fingerprint=ds.meta["dataset_fingerprint"],
        save=True,
        cache_root=tmp_path,
    )
    assert path is not None
    assert res.indices["test"].size > 0
    assert res.refs["test"] == "train"

    loaded = load_split(path)
    assert loaded.split_fingerprint == res.split_fingerprint


def test_sample_kfold_without_official_test() -> None:
    ds = make_toy_dataset(n=50, with_test=False)
    plan = SamplingPlan(
        split=KFoldSplitSpec(k=5, fold=2, stratify=True, shuffle=True, val_fraction=0.2),
        labeling=LabelingSpec(mode="count", value=10, strategy="balanced"),
    )
    res, _ = sample(
        ds, plan=plan, seed=1, dataset_fingerprint=ds.meta["dataset_fingerprint"], save=False
    )
    assert res.indices["test"].size > 0
    assert res.indices["val"].size > 0


def test_sample_respects_official_test() -> None:
    ds = make_toy_dataset(n=60, with_test=True)
    plan = SamplingPlan(
        split=HoldoutSplitSpec(test_fraction=0.9, val_fraction=0.2, stratify=False),
        labeling=LabelingSpec(mode="fraction", value=0.2),
    )
    res, _ = sample(
        ds, plan=plan, seed=0, dataset_fingerprint=ds.meta["dataset_fingerprint"], save=False
    )
    assert res.refs["test"] == "test"
    assert res.indices["test"].size == len(ds.test.y)


def test_missing_dataset_fingerprint_raises() -> None:
    ds = make_toy_dataset()
    ds = ds.__class__(train=ds.train, test=ds.test, meta={})
    plan = SamplingPlan(labeling=LabelingSpec())
    with pytest.raises(SamplingError):
        sample(ds, plan=plan, seed=0, save=False)


def test_component_split_seed_keeps_test_fixed_while_labeling_varies_and_replays(
    tmp_path,
) -> None:
    ds = make_toy_dataset(n=200, with_test=False)
    plan = SamplingPlan(
        component_seeds=SamplingComponentSeeds(split=2005),
        split=HoldoutSplitSpec(test_fraction=0.25, val_fraction=0.0, stratify=True),
        labeling=LabelingSpec(mode="count", value=20, strategy="proportional"),
    )

    first, path = sample(
        ds,
        plan=plan,
        seed=1,
        dataset_fingerprint=ds.meta["dataset_fingerprint"],
        save=True,
        cache_root=tmp_path,
    )
    second, _ = sample(
        ds,
        plan=plan,
        seed=2,
        dataset_fingerprint=ds.meta["dataset_fingerprint"],
        save=False,
    )
    other_split, _ = sample(
        ds,
        plan=SamplingPlan(
            component_seeds=SamplingComponentSeeds(split=2006),
            split=plan.split,
            labeling=plan.labeling,
        ),
        seed=1,
        dataset_fingerprint=ds.meta["dataset_fingerprint"],
        save=False,
    )

    assert path is not None
    np.testing.assert_array_equal(first.test_idx, second.test_idx)
    np.testing.assert_array_equal(first.train_idx, second.train_idx)
    assert not np.array_equal(first.labeled_idx, second.labeled_idx)
    assert first.split_fingerprint != second.split_fingerprint
    assert first.split_fingerprint != other_split.split_fingerprint
    assert not np.array_equal(first.test_idx, other_split.test_idx)

    replayed = load_split(path)
    assert replayed.split_fingerprint == first.split_fingerprint
    np.testing.assert_array_equal(replayed.test_idx, first.test_idx)
    np.testing.assert_array_equal(replayed.labeled_idx, first.labeled_idx)


def test_sample_propagates_master_seed_to_fixed_indices_artifact(tmp_path) -> None:
    ds = make_toy_dataset(n=20, with_test=False)
    path = tmp_path / "permutations.npz"
    source_sha256 = "f" * 64
    metadata = json.dumps(
        {
            "format": "ragged_int64_v1",
            "row_count": 2,
            "schema_version": 1,
            "source_key": "perm",
            "source_sha256": source_sha256,
        },
        sort_keys=True,
    ).encode()
    np.savez(
        path,
        metadata_json=np.frombuffer(metadata, dtype=np.uint8),
        offsets=np.array([0, 2, 4], dtype=np.int64),
        values=np.array([0, 1, 2, 3], dtype=np.int64),
    )
    plan = SamplingPlan(
        split=HoldoutSplitSpec(
            test_fraction=0.0,
            val_fraction=0.0,
            stratify=False,
        ),
        labeling=LabelingSpec(
            fixed_indices_artifact=FixedIndicesArtifactSpec(
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_sha256=source_sha256,
                expected_size=2,
            )
        ),
    )

    result, _ = sample(
        ds,
        plan=plan,
        seed=1,
        dataset_fingerprint=ds.meta["dataset_fingerprint"],
        save=False,
    )

    np.testing.assert_array_equal(result.labeled_idx, [2, 3])


def test_default_component_seed_behavior_preserves_legacy_split_fingerprint() -> None:
    ds = make_toy_dataset(n=40, with_test=False)
    plan = SamplingPlan(
        split=HoldoutSplitSpec(test_fraction=0.25, val_fraction=0.0, stratify=True),
        labeling=LabelingSpec(mode="count", value=5),
    )

    result, _ = sample(
        ds,
        plan=plan,
        seed=7,
        dataset_fingerprint=ds.meta["dataset_fingerprint"],
        save=False,
    )

    assert result.schema_version == 1
    assert result.split_fingerprint == (
        "7dbe078e36a3aa221be5c73d00ba9f24a0b1b177220a31117bc707755d0eae6c"
    )
    assert result.split_fingerprint == stable_hash(
        {
            "schema_version": 1,
            "dataset_fingerprint": ds.meta["dataset_fingerprint"],
            "plan": plan.as_dict(),
            "seed": 7,
        }
    )


def test_partition_subsampling_without_shuffle_preserves_source_prefix() -> None:
    ds = make_toy_dataset(n=20, with_test=False)
    plan = SamplingPlan(
        partition=PartitionSpec(max_samples=8, shuffle=False),
        split=HoldoutSplitSpec(
            test_fraction=0.25,
            val_fraction=0.0,
            stratify=False,
        ),
        labeling=LabelingSpec(
            mode="count",
            value=1,
            strategy="random",
            min_per_class=0,
        ),
    )

    first, _ = sample(
        ds,
        plan=plan,
        seed=1,
        dataset_fingerprint=ds.meta["dataset_fingerprint"],
        save=False,
    )
    second, _ = sample(
        ds,
        plan=plan,
        seed=2,
        dataset_fingerprint=ds.meta["dataset_fingerprint"],
        save=False,
    )

    expected = np.arange(8, dtype=np.int64)
    for result in (first, second):
        selected = np.sort(np.concatenate([result.train_idx, result.val_idx, result.test_idx]))
        np.testing.assert_array_equal(selected, expected)


def test_partition_subsampling_is_exact_reproducible_and_replayable(tmp_path) -> None:
    ds = make_toy_dataset(n=48_842, with_test=False)
    canonical_fingerprint = ds.meta["dataset_fingerprint"]
    plan = SamplingPlan(
        partition=PartitionSpec(max_samples=3442, shuffle=True),
        split=HoldoutSplitSpec(
            test_fraction=0.49128413712957584,
            val_fraction=0.0,
            stratify=False,
        ),
        labeling=LabelingSpec(
            mode="count",
            value=60,
            strategy="random",
            min_per_class=0,
        ),
    )

    result, path = sample(
        ds,
        plan=plan,
        seed=1,
        dataset_fingerprint=canonical_fingerprint,
        save=True,
        cache_root=tmp_path,
    )
    repeated, _ = sample(
        ds,
        plan=plan,
        seed=1,
        dataset_fingerprint=canonical_fingerprint,
        save=False,
    )
    other_seed, _ = sample(
        ds,
        plan=plan,
        seed=2,
        dataset_fingerprint=canonical_fingerprint,
        save=False,
    )

    assert path is not None
    assert (
        result.labeled_idx.size,
        result.unlabeled_idx.size,
        result.test_idx.size,
    ) == (60, 1691, 1691)
    assert result.stats["policy"]["partition_source_n"] == 48_842
    assert result.stats["policy"]["partition_selected_n"] == 3442
    selected = np.sort(np.concatenate([result.train_idx, result.val_idx, result.test_idx]))
    assert selected.size == 3442
    assert np.unique(selected).size == 3442
    assert selected.max() < 48_842
    excluded = np.setdiff1d(np.arange(48_842, dtype=np.int64), selected)
    assert excluded.size == 45_400
    assert np.intersect1d(excluded, result.train_idx).size == 0
    assert np.intersect1d(excluded, result.labeled_idx).size == 0
    assert np.intersect1d(excluded, result.unlabeled_idx).size == 0
    assert np.intersect1d(excluded, result.test_idx).size == 0
    assert len(ds.train.y) == 48_842
    assert ds.meta["dataset_fingerprint"] == canonical_fingerprint
    assert result.dataset_fingerprint == canonical_fingerprint
    assert repeated.dataset_fingerprint == canonical_fingerprint
    assert other_seed.dataset_fingerprint == canonical_fingerprint
    assert repeated.split_fingerprint == result.split_fingerprint
    np.testing.assert_array_equal(repeated.train_idx, result.train_idx)
    np.testing.assert_array_equal(repeated.test_idx, result.test_idx)
    assert other_seed.split_fingerprint != result.split_fingerprint
    other_selected = np.sort(
        np.concatenate([other_seed.train_idx, other_seed.val_idx, other_seed.test_idx])
    )
    assert not np.array_equal(other_selected, selected)

    replay = load_split(path)
    assert replay.dataset_fingerprint == canonical_fingerprint
    assert replay.split_fingerprint == result.split_fingerprint
    np.testing.assert_array_equal(replay.train_idx, result.train_idx)
    np.testing.assert_array_equal(replay.labeled_idx, result.labeled_idx)
    np.testing.assert_array_equal(replay.test_idx, result.test_idx)
