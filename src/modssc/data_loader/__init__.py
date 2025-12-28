"""Dataset download, caching and loading (canonical datasets only).

This module is responsible for:
- resolving dataset identifiers (catalog keys or provider URIs)
- downloading raw data into a local cache
- materializing a canonical dataset (official splits only when provided)
- storing processed data + manifests with stable fingerprints

It does NOT implement experimental splits (holdout, kfold, label fraction).
Those belong to a dedicated sampling/splitting component.
"""

from modssc.data_loader.api import (
    available_datasets,
    available_providers,
    cache_dir,
    dataset_info,
    download_all_datasets,
    download_dataset,
    load_dataset,
)
from modssc.data_loader.errors import (
    DataLoaderError,
    DatasetNotCachedError,
    InvalidDatasetURIError,
    OptionalDependencyError,
    ProviderNotFoundError,
    UnknownDatasetError,
)
from modssc.data_loader.formats import OutputFormat, get_output_format
from modssc.data_loader.numpy_adapter import dataset_to_numpy, split_to_numpy, to_numpy
from modssc.data_loader.types import (
    DatasetIdentity,
    DatasetRequest,
    DatasetSpec,
    DownloadReport,
    LoadedDataset,
    Split,
)

__all__ = [
    "DataLoaderError",
    "DatasetNotCachedError",
    "InvalidDatasetURIError",
    "OptionalDependencyError",
    "ProviderNotFoundError",
    "UnknownDatasetError",
    "DatasetIdentity",
    "DatasetRequest",
    "DatasetSpec",
    "DownloadReport",
    "LoadedDataset",
    "Split",
    "OutputFormat",
    "available_datasets",
    "available_providers",
    "cache_dir",
    "dataset_info",
    "download_all_datasets",
    "download_dataset",
    "load_dataset",
    "get_output_format",
    "to_numpy",
    "split_to_numpy",
    "dataset_to_numpy",
]
