"""Dataset transformations owned by the sampling protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset, Split
from modssc.sampling.fingerprint import stable_hash
from modssc.sampling.plan import SamplingPlan


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


def prepare_dataset(dataset: LoadedDataset, *, plan: SamplingPlan) -> LoadedDataset:
    """Apply dataset transformations declared by a native sampling plan.

    ``merge_official_splits`` assembles the provider's train and test splits,
    in that order, into the single vertex pool used by transductive protocols.
    The derived fingerprint identifies the transformed pool while retaining the
    provider fingerprint in metadata for provenance.
    """

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
            "dataset_fingerprint": stable_hash(transform),
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


__all__ = ["prepare_dataset"]
