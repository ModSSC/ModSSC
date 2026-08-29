from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader import dataset_info, load_dataset, verify_dataset_content
from modssc.data_loader.errors import DataLoaderError
from modssc.data_loader.types import LoadedDataset

from ..schema import BenchConfigError, DatasetConfig

_LOGGER = logging.getLogger(__name__)


def _verified_content_evidence(
    cfg: DatasetConfig,
    *,
    cache_dir: Path | None,
) -> dict[str, str]:
    try:
        evidence = verify_dataset_content(
            cfg.id,
            cache_dir=cache_dir,
            options=dict(cfg.options),
            rehash=True,
        )
    except (DataLoaderError, OSError) as exc:
        raise BenchConfigError(
            f"dataset content verification failed for {cfg.id!r}: {exc}",
            code="E_BENCH_DATASET_INTEGRITY",
        ) from exc
    required = {
        "cache_fingerprint",
        "content_sha256",
        "content_manifest_sha256",
        "cache_state_sha256",
    }
    if not isinstance(evidence, Mapping) or not required.issubset(evidence):
        raise BenchConfigError(
            f"dataset content verification returned incomplete evidence for {cfg.id!r}",
            code="E_BENCH_DATASET_INTEGRITY",
        )
    return {key: str(evidence[key]) for key in sorted(required)}


def _attach_verified_content(
    dataset: LoadedDataset,
    evidence: Mapping[str, str],
) -> LoadedDataset:
    meta = dict(dataset.meta or {})
    meta.update(
        {
            "dataset_fingerprint": evidence["cache_fingerprint"],
            "dataset_cache_fingerprint": evidence["cache_fingerprint"],
            "dataset_content_sha256": evidence["content_sha256"],
            "dataset_content_manifest_sha256": evidence["content_manifest_sha256"],
            "dataset_content_state_sha256": evidence["cache_state_sha256"],
            "dataset_content_rehashed": True,
        }
    )
    return replace(dataset, meta=meta)


def _split_size(split: Any) -> int | None:
    if split is None:
        return None
    y = getattr(split, "y", None)
    if y is None:
        return None
    try:
        return int(np.asarray(y).shape[0])
    except Exception:
        try:
            return int(len(y))
        except Exception:
            return None


def load(cfg: DatasetConfig) -> tuple[LoadedDataset, dict[str, Any]]:
    start = perf_counter()
    cache_dir = Path(cfg.cache_dir).expanduser().resolve() if cfg.cache_dir else None
    _LOGGER.info(
        "Dataset start: id=%s download=%s cache_dir=%s",
        cfg.id,
        bool(cfg.download),
        str(cache_dir) if cache_dir else None,
    )
    _LOGGER.debug("Dataset options: %s", sorted(cfg.options.keys()))
    ds = load_dataset(
        cfg.id,
        cache_dir=cache_dir,
        download=bool(cfg.download),
        options=dict(cfg.options),
    )
    if cfg.integrity is not None:
        _LOGGER.info("Dataset integrity rehash start: id=%s", cfg.id)
        evidence = _verified_content_evidence(cfg, cache_dir=cache_dir)
        ds = _attach_verified_content(ds, evidence)
        _LOGGER.info("Dataset integrity rehash done: id=%s", cfg.id)
    info = dataset_info(cfg.id).as_dict()
    n_train = _split_size(ds.train)
    n_test = _split_size(ds.test)
    has_graph = ds.has_graph
    fingerprint = ds.meta.get("dataset_fingerprint") if isinstance(ds.meta, dict) else None
    _LOGGER.info(
        "Dataset loaded: train=%s test=%s graph=%s fingerprint=%s provider=%s",
        n_train,
        n_test,
        bool(has_graph),
        fingerprint,
        info.get("provider"),
    )
    _LOGGER.info("Dataset stage done: duration_s=%.3f", perf_counter() - start)
    return ds, info


def revalidate_integrity(dataset: LoadedDataset, cfg: DatasetConfig) -> dict[str, str] | None:
    """Rehash a declared dataset again before publishing a run result."""

    if cfg.integrity is None:
        return None
    cache_dir = Path(cfg.cache_dir).expanduser().resolve() if cfg.cache_dir else None
    _LOGGER.info("Dataset integrity final rehash start: id=%s", cfg.id)
    evidence = _verified_content_evidence(cfg, cache_dir=cache_dir)
    verify_integrity(_attach_verified_content(dataset, evidence), cfg)
    _LOGGER.info("Dataset integrity final rehash done: id=%s", cfg.id)
    return evidence


def verify_integrity(dataset: LoadedDataset, cfg: DatasetConfig) -> None:
    """Fail closed when a YAML-declared dataset identity does not match."""

    expected = cfg.integrity
    if expected is None:
        return
    metadata = dataset.meta if isinstance(dataset.meta, Mapping) else {}
    fields = {
        "fingerprint": "dataset_fingerprint",
        "content_sha256": "dataset_content_sha256",
        "content_manifest_sha256": "dataset_content_manifest_sha256",
    }
    for config_field, metadata_field in fields.items():
        expected_value = getattr(expected, config_field)
        if expected_value is None:
            continue
        actual_value = metadata.get(metadata_field)
        if actual_value != expected_value:
            raise BenchConfigError(
                f"dataset.integrity.{config_field} differs for dataset {cfg.id!r}: "
                f"computed {actual_value!r}, expected {expected_value!r}",
                code="E_BENCH_DATASET_INTEGRITY",
            )
