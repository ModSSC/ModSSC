from __future__ import annotations

import pytest

from modssc.sampling.api import sample
from modssc.sampling.errors import SamplingError
from modssc.sampling.plan import HoldoutSplitSpec, KFoldSplitSpec, LabelingSpec, SamplingPlan
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
