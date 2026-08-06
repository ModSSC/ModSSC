from __future__ import annotations

import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset, Split
from modssc.sampling.api import sample
from modssc.sampling.plan import SamplingPlan
from modssc.sampling.result import SamplingResult

from ..utils.hashing import hash_any

_LOGGER = logging.getLogger(__name__)


def _concat_rows(first: Any, second: Any, *, field: str) -> Any:
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        if set(first) != set(second):
            raise ValueError(f"cannot merge official splits: {field} keys differ")
        return {key: _concat_rows(first[key], second[key], field=f"{field}.{key}") for key in first}
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        return np.concatenate([first, second], axis=0)
    try:
        import torch

        if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
            return torch.cat([first, second], dim=0)
    except ImportError:  # pragma: no cover - torch is optional
        pass
    raise ValueError(
        f"cannot merge official splits: unsupported {field} containers "
        f"{type(first).__name__}/{type(second).__name__}"
    )


def prepare_dataset(dataset: LoadedDataset, *, plan_dict: Mapping[str, Any]) -> LoadedDataset:
    """Apply protocol-level dataset transformations before sampling.

    ``merge_official_splits`` exists for published transductive protocols such
    as Calder et al.'s MNIST experiment, whose graph and label permutations use
    all 70,000 official train+test examples as a single vertex pool.
    """

    plan = _plan_from_dict(plan_dict)
    if not plan.policy.merge_official_splits:
        return dataset
    if dataset.test is None:
        raise ValueError("merge_official_splits requires an official test split")
    if any(
        value is not None
        for value in (
            dataset.train.edges,
            dataset.train.masks,
            dataset.test.edges,
            dataset.test.masks,
        )
    ):
        raise ValueError("merge_official_splits does not support provider graph edges or masks")

    n_train = int(np.asarray(dataset.train.y).shape[0])
    n_test = int(np.asarray(dataset.test.y).shape[0])
    source_fingerprint = str(dataset.meta.get("dataset_fingerprint", ""))
    if not source_fingerprint:
        raise ValueError("dataset.meta['dataset_fingerprint'] is required before merging splits")
    transform = {
        "name": "merge_official_splits",
        "version": 1,
        "source_fingerprint": source_fingerprint,
        "n_train": n_train,
        "n_test": n_test,
        "order": ["train", "test"],
    }
    meta = dict(dataset.meta)
    meta.update(
        {
            "dataset_fingerprint_source": source_fingerprint,
            "dataset_fingerprint": hash_any(transform),
            "official_splits_merged": True,
            "official_split_merge": transform,
        }
    )
    return LoadedDataset(
        train=Split(
            X=_concat_rows(dataset.train.X, dataset.test.X, field="X"),
            y=_concat_rows(dataset.train.y, dataset.test.y, field="y"),
        ),
        test=None,
        meta=meta,
    )


def _plan_from_dict(obj: Mapping[str, Any]) -> SamplingPlan:
    if not isinstance(obj, Mapping):
        raise ValueError("sampling.plan must be a mapping")
    return SamplingPlan.from_dict(dict(obj))


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
        "merge_official_splits": bool(plan.policy.merge_official_splits),
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
