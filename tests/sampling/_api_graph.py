from __future__ import annotations

from modssc.sampling.api import sample
from modssc.sampling.plan import HoldoutSplitSpec, LabelingSpec, SamplingPlan, SamplingPolicy
from tests.sampling._stubs import make_graph_dataset


def test_graph_uses_official_masks_by_default() -> None:
    ds = make_graph_dataset(with_official_masks=True)
    plan = SamplingPlan(
        split=HoldoutSplitSpec(test_fraction=0.2, val_fraction=0.1, stratify=True),
        labeling=LabelingSpec(mode="per_class", value=2),
    )
    res, _ = sample(
        ds, plan=plan, seed=0, dataset_fingerprint=ds.meta["dataset_fingerprint"], save=False
    )
    assert res.is_graph()
    assert res.masks["train"].sum() == 20
    assert res.masks["labeled"].sum() > 0
    assert (res.masks["labeled"] & ~res.masks["train"]).sum() == 0


def test_graph_generate_masks_when_disabled() -> None:
    ds = make_graph_dataset(with_official_masks=True)
    plan = SamplingPlan(
        split=HoldoutSplitSpec(test_fraction=0.2, val_fraction=0.1, stratify=True),
        labeling=LabelingSpec(mode="fraction", value=0.1),
        policy=SamplingPolicy(use_official_graph_masks=False),
    )
    res, _ = sample(
        ds, plan=plan, seed=0, dataset_fingerprint=ds.meta["dataset_fingerprint"], save=False
    )
    assert res.masks["train"].sum() + res.masks["val"].sum() + res.masks["test"].sum() == len(
        ds.train.y
    )
