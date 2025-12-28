"""Sampling example (semi-supervised split) on the built-in 'toy' dataset.

Goal
- Show how to create a reproducible split (train/val + labeled/unlabeled).
- Demonstrates the SamplingPlan API.

Expected
- Runs after: pip install modssc
"""

from __future__ import annotations

from modssc.data_loader import load_dataset
from modssc.sampling import HoldoutSplitSpec, LabelingSpec, SamplingPlan, sample


def main() -> None:
    ds = load_dataset("toy", download=True)
    ds_fp = str(ds.meta.get("dataset_fingerprint"))

    plan = SamplingPlan(
        split=HoldoutSplitSpec(test_fraction=0.0, val_fraction=0.2, stratify=True),
        labeling=LabelingSpec(mode="fraction", value=0.2, per_class=True, min_per_class=1),
    )

    res, _ = sample(ds, plan=plan, seed=0, dataset_fingerprint=ds_fp, save=False)

    print("is_graph:", res.is_graph())
    print("train size:", int(res.train_idx.size))
    print("val size:", int(res.val_idx.size))
    print("test size:", int(res.test_idx.size), "(empty here because official test is used)")
    print("labeled size:", int(res.labeled_idx.size))
    print("unlabeled size:", int(res.unlabeled_idx.size))
    print("refs:", dict(res.refs))
    print("stats keys:", sorted(res.stats.keys()))


if __name__ == "__main__":
    main()
