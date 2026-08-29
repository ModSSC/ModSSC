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
from modssc.sampling.dataset import prepare_dataset
from modssc.sampling.errors import (
    MissingDatasetFingerprintError,
    SamplingError,
    SamplingValidationError,
)
from modssc.sampling.plan import (
    FixedIndicesArtifactSpec,
    HoldoutSplitSpec,
    ImbalanceSpec,
    KFoldSplitSpec,
    LabelingSpec,
    OrderedPartitionArtifactSpec,
    PartitionSpec,
    SamplingComponentSeeds,
    SamplingPlan,
    SamplingPolicy,
)
from modssc.sampling.result import SamplingResult
from modssc.sampling.routing import (
    InductiveGraphSamplingPolicy,
    SamplingRoutingEvent,
    SamplingRoutingResult,
    route_sampling_for_regime,
)

__all__ = [
    "SamplingError",
    "MissingDatasetFingerprintError",
    "SamplingValidationError",
    "HoldoutSplitSpec",
    "KFoldSplitSpec",
    "FixedIndicesArtifactSpec",
    "OrderedPartitionArtifactSpec",
    "PartitionSpec",
    "SamplingComponentSeeds",
    "LabelingSpec",
    "ImbalanceSpec",
    "SamplingPolicy",
    "SamplingPlan",
    "SamplingResult",
    "InductiveGraphSamplingPolicy",
    "SamplingRoutingEvent",
    "SamplingRoutingResult",
    "prepare_dataset",
    "sample",
    "save_split",
    "load_split",
    "default_split_cache_dir",
    "split_dir_for",
    "route_sampling_for_regime",
]
