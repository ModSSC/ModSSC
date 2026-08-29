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
    CacheEntryExpectation,
    CachePromotionItem,
    CachePromotionReport,
    available_datasets,
    available_providers,
    cache_dir,
    dataset_fingerprint,
    dataset_info,
    download_all_datasets,
    download_dataset,
    load_dataset,
    promote_cache_entries,
    provider_names,
    resolve_dataset_identity,
    verify_dataset_content,
)
from modssc.data_loader.errors import (
    CachePromotionError,
    DataLoaderError,
    DatasetNotCachedError,
    DatasetSelectionError,
    InvalidDatasetURIError,
    OptionalDependencyError,
    ProviderNotFoundError,
    UnknownDatasetError,
)
from modssc.data_loader.formats import OutputFormat, get_output_format
from modssc.data_loader.numpy_adapter import dataset_to_numpy, split_to_numpy, to_numpy
from modssc.data_loader.selection import select_rows
from modssc.data_loader.types import (
    DatasetIdentity,
    DatasetRequest,
    DatasetSpec,
    DownloadReport,
    LoadedDataset,
    Split,
)

__all__ = [
    "CachePromotionError",
    "CacheEntryExpectation",
    "CachePromotionItem",
    "CachePromotionReport",
    "DataLoaderError",
    "DatasetNotCachedError",
    "DatasetSelectionError",
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
    "dataset_fingerprint",
    "dataset_info",
    "download_all_datasets",
    "download_dataset",
    "load_dataset",
    "provider_names",
    "promote_cache_entries",
    "resolve_dataset_identity",
    "verify_dataset_content",
    "get_output_format",
    "to_numpy",
    "split_to_numpy",
    "dataset_to_numpy",
    "select_rows",
]
