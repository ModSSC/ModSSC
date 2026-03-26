"""Additional test coverage for sampling/api.py complex branches."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from modssc.sampling.api import load_split, sample
from modssc.sampling.plan import (
    ImbalanceSpec,
    KFoldSplitSpec,
    LabelingSpec,
    SamplingPlan,
    SamplingPolicy,
)
from tests.sampling._stubs import make_graph_dataset, make_toy_dataset


def test_load_split_wrapper(tmp_path):
    """Test load_split wrapper function."""

    with patch("modssc.sampling.api._load_split") as mock_load:
        p = tmp_path / "dummy"
        load_split(p)
        mock_load.assert_called_once_with(p)


def test_sample_inductive_official_kfold():
    """Test _sample_inductive with respect_official_test and KFold split."""
    ds = make_toy_dataset(n=100, with_test=True)
    plan = SamplingPlan(
        split=KFoldSplitSpec(k=5, fold=0, stratify=True, shuffle=True),
        labeling=LabelingSpec(mode="fraction", value=0.1),
        policy=SamplingPolicy(respect_official_test=True),
    )
    res, _ = sample(ds, plan=plan, seed=0, dataset_fingerprint="fp", save=False)

    assert res.indices["train"].size > 0
    assert res.indices["val"].size > 0
    assert res.indices["test"].size == len(ds.test.y)
    assert res.refs["test"] == "test"


def test_sample_inductive_override_official_error():
    """Test _sample_inductive raises NotImplementedError for override_official."""
    ds = make_toy_dataset(n=100, with_test=True)
    plan = SamplingPlan(
        split=KFoldSplitSpec(k=5, fold=0),
        labeling=LabelingSpec(),
        policy=SamplingPolicy(respect_official_test=True, allow_override_official=True),
    )
    with pytest.raises(NotImplementedError, match="override_official is not implemented"):
        sample(ds, plan=plan, seed=0, dataset_fingerprint="fp", save=False)


def test_sample_inductive_imbalance_labeled():
    """Test _sample_inductive with imbalance applied to labeled set."""
    ds = make_toy_dataset(n=100, with_test=False)

    imb_spec = ImbalanceSpec(apply_to="labeled", kind="subsample_max_per_class", max_per_class=5)
    plan = SamplingPlan(
        split=KFoldSplitSpec(k=5, fold=0),
        labeling=LabelingSpec(mode="fraction", value=0.5),
        imbalance=imb_spec,
    )
    res, _ = sample(ds, plan=plan, seed=0, dataset_fingerprint="fp", save=False)

    assert res.indices["train_labeled"].size > 0


def test_sample_graph_kfold():
    """Test _sample_graph with KFold split."""
    ds = make_graph_dataset(n_nodes=100, with_official_masks=False)
    plan = SamplingPlan(
        split=KFoldSplitSpec(k=5, fold=0, stratify=True, shuffle=True, val_fraction=0.1),
        labeling=LabelingSpec(mode="fraction", value=0.1),
    )
    res, _ = sample(ds, plan=plan, seed=0, dataset_fingerprint="fp", save=False)

    assert res.masks["train"].sum() > 0
    assert res.masks["val"].sum() > 0
    assert res.masks["test"].sum() > 0


def test_sample_graph_imbalance_labeled():
    """Test _sample_graph with imbalance applied to labeled set."""
    ds = make_graph_dataset(n_nodes=100, with_official_masks=False)
    imb_spec = ImbalanceSpec(apply_to="labeled", kind="subsample_max_per_class", max_per_class=5)
    plan = SamplingPlan(
        split=KFoldSplitSpec(k=5, fold=0),
        labeling=LabelingSpec(mode="fraction", value=0.5),
        imbalance=imb_spec,
    )
    res, _ = sample(ds, plan=plan, seed=0, dataset_fingerprint="fp", save=False)


def test_sample_inductive_imbalance_train():
    """Test _sample_inductive with imbalance applied to train set."""
    ds = make_toy_dataset(n=100, with_test=False)
    imb_spec = ImbalanceSpec(apply_to="train", kind="subsample_max_per_class", max_per_class=5)
    plan = SamplingPlan(
        split=KFoldSplitSpec(k=5, fold=0),
        labeling=LabelingSpec(mode="fraction", value=0.5),
        imbalance=imb_spec,
    )
    res, _ = sample(ds, plan=plan, seed=0, dataset_fingerprint="fp", save=False)

    assert res.indices["train"].size <= 15


def test_save_and_load_split(tmp_path):
    """Test save_split and load_split."""
    ds = make_toy_dataset(n=20, with_test=False)
    plan = SamplingPlan()
    res, _ = sample(ds, plan=plan, seed=0, dataset_fingerprint="fp", save=False)

    out_dir = tmp_path / "split"
    from modssc.sampling.api import load_split, save_split

    save_split(res, out_dir)
    assert (out_dir / "split.json").exists()
    assert (out_dir / "arrays.npz").exists()

    loaded = load_split(out_dir)
    assert loaded.split_fingerprint == res.split_fingerprint
    np.testing.assert_array_equal(loaded.indices["train"], res.indices["train"])


def test_sample_save(tmp_path):
    """Test sample with save=True."""
    ds = make_toy_dataset(n=20, with_test=False)
    plan = SamplingPlan()

    res, path = sample(
        ds, plan=plan, seed=0, dataset_fingerprint="fp", save=True, cache_root=tmp_path
    )

    assert path is not None
    assert path.exists()
    assert (path / "split.json").exists()


def test_sample_graph_official_masks():
    """Test _sample_graph with official masks."""
    ds = make_graph_dataset(n_nodes=100, with_official_masks=True)
    plan = SamplingPlan(policy=SamplingPolicy(use_official_graph_masks=True))
    res, _ = sample(ds, plan=plan, seed=0, dataset_fingerprint="fp", save=False)

    assert res.masks["train"].sum() == 20
    assert res.masks["val"].sum() == 10
    assert res.masks["test"].sum() == 20
