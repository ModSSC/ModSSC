from __future__ import annotations

import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset
from modssc.sampling.api import sample
from modssc.sampling.plan import (
    HoldoutSplitSpec,
    ImbalanceSpec,
    KFoldSplitSpec,
    LabelingSpec,
    SamplingPlan,
    SamplingPolicy,
)
from modssc.sampling.result import SamplingResult

_LOGGER = logging.getLogger(__name__)


def _plan_from_dict(obj: Mapping[str, Any]) -> SamplingPlan:
    split_obj = dict(obj.get("split", {"kind": "holdout"}))
    if split_obj.get("kind") == "kfold":
        split = KFoldSplitSpec(
            k=int(split_obj.get("k", 5)),
            fold=int(split_obj.get("fold", 0)),
            stratify=bool(split_obj.get("stratify", True)),
            shuffle=bool(split_obj.get("shuffle", True)),
            val_fraction=float(split_obj.get("val_fraction", 0.0)),
        )
    else:
        split = HoldoutSplitSpec(
            test_fraction=float(split_obj.get("test_fraction", 0.2)),
            val_fraction=float(split_obj.get("val_fraction", 0.1)),
            stratify=bool(split_obj.get("stratify", True)),
            shuffle=bool(split_obj.get("shuffle", True)),
        )

    lab_obj = dict(obj.get("labeling", {"mode": "fraction", "value": 0.1}))
    labeling = LabelingSpec(
        mode=str(lab_obj.get("mode", "fraction")),
        value=lab_obj.get("value", 0.1),
        per_class=bool(lab_obj.get("per_class", False)),
        min_per_class=int(lab_obj.get("min_per_class", 1)),
        strategy=str(lab_obj.get("strategy", "proportional")),
        fixed_indices=lab_obj.get("fixed_indices"),
    )

    imb_obj = dict(obj.get("imbalance", {"kind": "none"}))
    imbalance = ImbalanceSpec(
        kind=str(imb_obj.get("kind", "none")),
        apply_to=str(imb_obj.get("apply_to", "train")),
        max_per_class=imb_obj.get("max_per_class"),
        alpha=imb_obj.get("alpha"),
        min_per_class=int(imb_obj.get("min_per_class", 1)),
    )

    pol_obj = dict(obj.get("policy", {}))
    policy = SamplingPolicy(
        respect_official_test=bool(pol_obj.get("respect_official_test", True)),
        use_official_graph_masks=bool(pol_obj.get("use_official_graph_masks", True)),
        allow_override_official=bool(pol_obj.get("allow_override_official", False)),
    )

    return SamplingPlan(split=split, labeling=labeling, imbalance=imbalance, policy=policy)


def run(
    dataset: LoadedDataset,
    *,
    plan_dict: Mapping[str, Any],
    seed: int,
    dataset_id: str | None = None,
) -> SamplingResult:
    start = perf_counter()
    plan = _plan_from_dict(plan_dict)
    policy = {
        "respect_official_test": bool(plan.policy.respect_official_test),
        "use_official_graph_masks": bool(plan.policy.use_official_graph_masks),
        "allow_override_official": bool(plan.policy.allow_override_official),
    }
    _LOGGER.info("Sampling start: seed=%s dataset_id=%s", int(seed), dataset_id)
    _LOGGER.debug(
        "Sampling plan: split=%s labeling=%s imbalance=%s policy=%s",
        plan.split.as_dict(),
        plan.labeling.as_dict(),
        plan.imbalance.as_dict(),
        policy,
    )
    ds_fp = dataset.meta.get("dataset_fingerprint") if isinstance(dataset.meta, dict) else None
    if ds_fp is None:
        raise ValueError("dataset.meta['dataset_fingerprint'] is required for sampling")

    _LOGGER.debug("Sampling dataset_fingerprint=%s", ds_fp)
    result, _ = sample(
        dataset,
        plan=plan,
        seed=int(seed),
        dataset_fingerprint=str(ds_fp),
        dataset_id=dataset_id,
        save=False,
    )

    y_train = np.asarray(dataset.train.y)
    n_train = int(y_train.shape[0])
    n_test = None
    if dataset.test is not None:
        y_test = np.asarray(dataset.test.y)
        n_test = int(y_test.shape[0])

    n_nodes = None
    if (
        getattr(dataset.train, "edges", None) is not None
        or getattr(dataset.train, "masks", None) is not None
    ):
        n_nodes = int(np.asarray(dataset.train.y).shape[0])

    result.validate(n_train=n_train, n_test=n_test, n_nodes=n_nodes)
    _LOGGER.info(
        "Sampling result: train=%s val=%s test=%s labeled=%s unlabeled=%s graph=%s",
        int(result.train_idx.shape[0]),
        int(result.val_idx.shape[0]),
        int(result.test_idx.shape[0]),
        int(result.labeled_idx.shape[0]),
        int(result.unlabeled_idx.shape[0]),
        result.is_graph(),
    )
    _LOGGER.debug("Sampling stats: %s", dict(result.stats))
    _LOGGER.info("Sampling stage done: duration_s=%.3f", perf_counter() - start)
    return result
