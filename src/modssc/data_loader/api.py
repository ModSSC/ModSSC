from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader import cache
from modssc.data_loader.catalog import DATASET_CATALOG
from modssc.data_loader.errors import (
    DatasetNotCachedError,
    OptionalDependencyError,
    UnknownDatasetError,
)
from modssc.data_loader.manifest import build_manifest, write_manifest
from modssc.data_loader.numpy_adapter import dataset_to_numpy
from modssc.data_loader.providers import create_provider, get_provider_names
from modssc.data_loader.storage import FileStorage
from modssc.data_loader.types import (
    DatasetIdentity,
    DatasetRequest,
    DatasetSpec,
    DownloadReport,
    LoadedDataset,
)
from modssc.data_loader.uri import is_uri, parse_uri

SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)


def _split_stats(split: Any) -> tuple[int | None, int | None]:
    y = getattr(split, "y", None)
    if y is None:
        return None, None
    try:
        y_arr = np.asarray(y)
    except Exception:
        return None, None
    if y_arr.ndim == 0:
        return None, None
    n_samples = int(y_arr.shape[0])
    n_classes = None
    if n_samples <= 1_000_000:
        try:
            if np.issubdtype(y_arr.dtype, np.integer):
                y_int = y_arr.astype(np.int64, copy=False)
                if y_int.size == 0:
                    n_classes = 0
                else:
                    min_label = int(y_int.min())
                    max_label = int(y_int.max())
                    if min_label >= 0 and max_label <= 1_000_000:
                        counts = np.bincount(y_int, minlength=max_label + 1)
                        n_classes = int(np.count_nonzero(counts))
                    else:
                        n_classes = int(np.unique(y_arr).size)
            else:
                n_classes = int(np.unique(y_arr).size)
        except Exception:
            n_classes = None
    return n_samples, n_classes


def cache_dir() -> Path:
    return cache.default_cache_dir()


def available_datasets() -> list[str]:
    return sorted(DATASET_CATALOG.keys())


def dataset_info(dataset_id: str) -> DatasetSpec:
    if dataset_id in DATASET_CATALOG:
        return DATASET_CATALOG[dataset_id]
    # provider uri
    if is_uri(dataset_id):
        parsed = parse_uri(dataset_id)
        # minimal spec without importing heavy deps
        provider = parsed.provider
        required_extra = _provider_extra(provider)
        return DatasetSpec(
            key=dataset_id,
            provider=provider,
            uri=dataset_id,
            modality="unknown",
            task="unknown",
            description="Provider dataset URI (not in curated catalog).",
            required_extra=required_extra,
            source_kwargs={},
        )
    raise UnknownDatasetError(dataset_id)


def load_dataset(
    dataset_id: str,
    *,
    cache_dir: Path | None = None,
    download: bool = True,
    force: bool = False,
    options: Mapping[str, Any] | None = None,
    as_numpy: bool = False,
    allow_object: bool = True,
) -> LoadedDataset:
    """Load a dataset from processed cache, optionally downloading if missing."""
    start = perf_counter()
    layout = _layout(cache_dir)
    req = DatasetRequest(id=dataset_id, options=options or {})
    identity = _resolve_identity(req)

    fp = identity.fingerprint(schema_version=SCHEMA_VERSION)
    logger.info(
        "Dataset load: id=%s provider=%s version=%s fingerprint=%s download=%s force=%s cache_dir=%s",
        dataset_id,
        identity.provider,
        identity.version,
        fp,
        bool(download),
        bool(force),
        str(layout.root),
    )
    logger.debug("Dataset resolved_kwargs: %s", dict(identity.resolved_kwargs))

    if not force and cache.is_cached(layout, fp):
        ds = _load_processed(layout, fp)
        n_train, n_classes = _split_stats(ds.train)
        n_test, _ = _split_stats(ds.test)
        logger.info(
            "Dataset cached: id=%s train=%s test=%s n_classes=%s duration_s=%.3f",
            dataset_id,
            n_train,
            n_test,
            n_classes,
            perf_counter() - start,
        )
        return dataset_to_numpy(ds, allow_object=allow_object) if as_numpy else ds

    if not download:
        raise DatasetNotCachedError(dataset_id)

    ds = _download_and_store(layout, identity, force=force)
    n_train, n_classes = _split_stats(ds.train)
    n_test, _ = _split_stats(ds.test)
    logger.info(
        "Dataset ready: id=%s train=%s test=%s n_classes=%s duration_s=%.3f",
        dataset_id,
        n_train,
        n_test,
        n_classes,
        perf_counter() - start,
    )
    return dataset_to_numpy(ds, allow_object=allow_object) if as_numpy else ds


def download_dataset(
    dataset_id: str,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    options: Mapping[str, Any] | None = None,
    as_numpy: bool = False,
    allow_object: bool = True,
) -> LoadedDataset:
    start = perf_counter()
    layout = _layout(cache_dir)
    req = DatasetRequest(id=dataset_id, options=options or {})
    identity = _resolve_identity(req)
    fp = identity.fingerprint(schema_version=SCHEMA_VERSION)

    if not force and cache.is_cached(layout, fp):
        ds = _load_processed(layout, fp)
        logger.info(
            "Dataset cached: id=%s provider=%s fingerprint=%s duration_s=%.3f",
            dataset_id,
            identity.provider,
            fp,
            perf_counter() - start,
        )
        return dataset_to_numpy(ds, allow_object=allow_object) if as_numpy else ds

    ds = _download_and_store(layout, identity, force=force)
    logger.info(
        "Dataset downloaded: id=%s provider=%s fingerprint=%s duration_s=%.3f",
        dataset_id,
        identity.provider,
        fp,
        perf_counter() - start,
    )
    return dataset_to_numpy(ds, allow_object=allow_object) if as_numpy else ds


def download_all_datasets(
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    modalities: Sequence[str] | None = None,
    ignore_missing_extras: bool = True,
    skip_cached: bool = False,
) -> DownloadReport:
    start = perf_counter()
    layout = _layout(cache_dir)
    keys = _select_catalog_keys(include=include, exclude=exclude, modalities=modalities)
    logger.info(
        "Download all datasets: count=%s force=%s skip_cached=%s cache_dir=%s",
        len(keys),
        bool(force),
        bool(skip_cached),
        str(layout.root),
    )

    downloaded: list[str] = []
    skipped_cached: list[str] = []
    skipped_missing: list[str] = []
    missing_extras: dict[str, list[str]] = {}
    failed: dict[str, str] = {}

    for key in keys:
        try:
            identity = _resolve_identity(DatasetRequest(id=key, options={}))
            fp = identity.fingerprint(schema_version=SCHEMA_VERSION)

            if skip_cached and cache.is_cached(layout, fp):
                skipped_cached.append(key)
                continue

            download_dataset(key, cache_dir=layout.root, force=force)
            downloaded.append(key)

        except OptionalDependencyError as e:
            if ignore_missing_extras:
                skipped_missing.append(key)
                missing_extras.setdefault(e.extra, []).append(key)
            else:
                raise
        except Exception as e:
            failed[key] = f"{type(e).__name__}: {e}"

    report = DownloadReport(
        downloaded=downloaded,
        skipped_already_cached=skipped_cached,
        skipped_missing_extras=skipped_missing,
        missing_extras=missing_extras,
        failed=failed,
    )
    logger.info(
        "Download all datasets done: downloaded=%s skipped_cached=%s skipped_missing=%s failed=%s duration_s=%.3f",
        len(downloaded),
        len(skipped_cached),
        len(skipped_missing),
        len(failed),
        perf_counter() - start,
    )
    return report


# ----------------------------
# Internal helpers
# ----------------------------


def _layout(cache_dir_override: Path | None) -> cache.CacheLayout:
    root = (cache_dir_override or cache.default_cache_dir()).expanduser().resolve()
    layout = cache.CacheLayout(root=root)
    cache.ensure_layout(layout)
    return layout


def _select_catalog_keys(
    *,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    modalities: Sequence[str] | None,
) -> list[str]:
    keys = list(DATASET_CATALOG.keys()) if include is None else list(include)
    selected = set(keys)
    if exclude:
        selected.difference_update(exclude)
    if modalities:
        wanted = set(modalities)
        selected = {k for k in selected if DATASET_CATALOG[k].modality in wanted}
    return sorted(selected)


def _provider_extra(provider: str) -> str | None:
    # keep this minimal to avoid importing provider modules on dataset_info
    mapping = {
        "toy": None,
        "openml": "openml",
        "hf": "hf",
        "tfds": "tfds",
        "torchvision": "vision",
        "torchaudio": "audio",
        "pyg": "graph",
    }
    return mapping.get(provider)


def _resolve_identity(req: DatasetRequest) -> DatasetIdentity:
    # catalog key
    if req.id in DATASET_CATALOG:
        spec = DATASET_CATALOG[req.id]
        # catalog options override the spec
        options = {**dict(spec.source_kwargs), **dict(req.options)}
        parsed = parse_uri(spec.uri)
        provider = create_provider(parsed.provider)
        identity = provider.resolve(parsed, options=options)
        # enforce curated modality/task when known
        identity = replace(identity, modality=spec.modality, task=spec.task)
        logger.debug(
            "Resolved dataset id=%s provider=%s canonical_uri=%s",
            req.id,
            identity.provider,
            identity.canonical_uri,
        )
        return identity

    # uri mode
    if not is_uri(req.id):
        raise UnknownDatasetError(req.id)

    parsed = parse_uri(req.id)
    provider = create_provider(parsed.provider)
    identity = provider.resolve(parsed, options=dict(req.options))
    logger.debug(
        "Resolved dataset uri=%s provider=%s canonical_uri=%s",
        req.id,
        identity.provider,
        identity.canonical_uri,
    )
    return identity


def _download_and_store(
    layout: cache.CacheLayout, identity: DatasetIdentity, *, force: bool
) -> LoadedDataset:
    fp = identity.fingerprint(schema_version=SCHEMA_VERSION)

    if not force and cache.is_cached(layout, fp):
        return _load_processed(layout, fp)

    with cache.cache_lock(layout, fp):
        if not force and cache.is_cached(layout, fp):
            return _load_processed(layout, fp)

        provider = create_provider(identity.provider)
        raw_dir = layout.raw_dir(identity.provider, identity.dataset_id, identity.version)
        raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Loading provider dataset: provider=%s dataset_id=%s version=%s raw_dir=%s",
            identity.provider,
            identity.dataset_id,
            identity.version,
            str(raw_dir),
        )
        ds = provider.load_canonical(identity, raw_dir=raw_dir)

        # Inject fingerprint into meta so it persists and is available to consumers (e.g. sampling)
        if ds.meta is None:
            ds = replace(ds, meta={})
        ds.meta["dataset_fingerprint"] = fp

        # store processed
        processed_dir = layout.processed_dir(fp)
        if processed_dir.exists():
            # clean partial state
            import shutil

            shutil.rmtree(processed_dir, ignore_errors=True)

        storage = FileStorage()
        storage.save(processed_dir, ds)

        manifest = build_manifest(
            schema_version=SCHEMA_VERSION, fingerprint=fp, identity=identity, dataset=ds
        )
        write_manifest(layout.manifest_path(fp), manifest)
        cache.index_upsert(layout, fingerprint=fp, manifest=manifest)

        logger.info(
            "Dataset stored: provider=%s dataset_id=%s fingerprint=%s",
            identity.provider,
            identity.dataset_id,
            fp,
        )
        return ds


def _load_processed(layout: cache.CacheLayout, fingerprint: str) -> LoadedDataset:
    storage = FileStorage()
    ds = storage.load(layout.processed_dir(fingerprint))
    # Ensure fingerprint is in meta
    if ds.meta is None:
        ds = replace(ds, meta={})
    ds.meta["dataset_fingerprint"] = fingerprint
    logger.debug("Loaded cached dataset: fingerprint=%s", fingerprint)
    return ds


def available_providers() -> list[str]:
    """Public helper: list provider names."""
    return get_provider_names()


def provider_names() -> list[str]:
    """Backward-compatible alias for available_providers."""
    return available_providers()
