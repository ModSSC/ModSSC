"""Sampling and splitting for semi-supervised experiments.

This module takes a canonical dataset from `modssc.data_loader` and produces
reproducible experimental splits (holdout, k-fold) plus labeled/unlabeled
partitions.

It does NOT download datasets. Use `modssc.data_loader` for that.
"""

from modssc.sampling.api import (
    default_split_cache_dir,
    load_split,
    sample,
    save_split,
    split_dir_for,
)
from modssc.sampling.errors import (
    MissingDatasetFingerprintError,
    SamplingError,
    SamplingValidationError,
)
from modssc.sampling.plan import (
    HoldoutSplitSpec,
    ImbalanceSpec,
    KFoldSplitSpec,
    LabelingSpec,
    SamplingPlan,
    SamplingPolicy,
)
from modssc.sampling.result import SamplingResult

__all__ = [
    "SamplingError",
    "MissingDatasetFingerprintError",
    "SamplingValidationError",
    "HoldoutSplitSpec",
    "KFoldSplitSpec",
    "LabelingSpec",
    "ImbalanceSpec",
    "SamplingPolicy",
    "SamplingPlan",
    "SamplingResult",
    "sample",
    "save_split",
    "load_split",
    "default_split_cache_dir",
    "split_dir_for",
]
