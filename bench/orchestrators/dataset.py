from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader import dataset_info, load_dataset
from modssc.data_loader.types import LoadedDataset

from ..schema import DatasetConfig

_LOGGER = logging.getLogger(__name__)


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
    info = dataset_info(cfg.id).as_dict()
    n_train = _split_size(ds.train)
    n_test = _split_size(ds.test)
    has_graph = getattr(ds.train, "edges", None) is not None or getattr(ds.train, "masks", None)
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
