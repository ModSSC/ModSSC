from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_loader.optional import optional_import
from modssc.data_loader.providers.base import BaseProvider
from modssc.data_loader.types import DatasetIdentity, LoadedDataset, Split
from modssc.data_loader.uri import ParsedURI


def _split_ref(ref: str) -> tuple[str, str | None]:
    ref = ref.strip()
    if "/" in ref:
        name, cfg = ref.split("/", 1)
        name = name.strip()
        cfg = cfg.strip() or None
        return name, cfg
    return ref, None


def _normalize_filter(values: Any) -> list[Any] | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple, set, np.ndarray)):
        return list(values)
    return [values]


def _apply_class_filter(
    X: np.ndarray, y: np.ndarray, *, class_filter: list[Any] | None
) -> tuple[np.ndarray, np.ndarray]:
    if class_filter is None:
        return X, y
    mask = np.isin(y, np.asarray(class_filter))
    return X[mask], y[mask]


def _limit_samples(
    X: np.ndarray, y: np.ndarray, *, max_samples: int | None, seed: int | None
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None:
        return X, y
    n = int(y.shape[0])
    max_n = int(max_samples)
    if max_n <= 0 or n == 0:
        return X[:0], y[:0]
    take = min(n, max_n)
    idx = np.arange(n, dtype=np.int64)
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(idx)
    return X[idx[:take]], y[idx[:take]]


def _apply_limits(
    split: Split | None,
    *,
    class_filter: list[Any] | None,
    max_samples: int | None,
    seed: int | None,
) -> Split | None:
    if split is None:
        return None
    X = np.asarray(split.X)
    y = np.asarray(split.y)
    X, y = _apply_class_filter(X, y, class_filter=class_filter)
    X, y = _limit_samples(X, y, max_samples=max_samples, seed=seed)
    return Split(X=X, y=y)


class HuggingFaceDatasetsProvider(BaseProvider):
    name = "hf"
    required_extra = "hf"

    def resolve(self, parsed: ParsedURI, *, options: Mapping[str, Any]) -> DatasetIdentity:
        name, cfg = _split_ref(parsed.reference)
        text_column = options.get("text_column", "text")
        label_column = options.get("label_column", "label")

        resolved_kwargs = {
            "name": name,
            "config": cfg if cfg is not None else options.get("config", None),
            "text_column": text_column,
            "label_column": label_column,
            "prefer_test_split": bool(options.get("prefer_test_split", True)),
            "class_filter": _normalize_filter(options.get("class_filter")),
            "max_train_samples": options.get("max_train_samples"),
            "max_test_samples": options.get("max_test_samples"),
            "seed": options.get("seed"),
        }

        return DatasetIdentity(
            provider=self.name,
            canonical_uri=f"hf:{name}"
            + (f"/{resolved_kwargs['config']}" if resolved_kwargs["config"] else ""),
            dataset_id=name,
            version=None,
            modality=str(options.get("modality", "text")),
            task=str(options.get("task", "classification")),
            required_extra=self.required_extra,
            resolved_kwargs=resolved_kwargs,
        )

    def load_canonical(self, identity: DatasetIdentity, *, raw_dir: Path) -> LoadedDataset:
        datasets_mod = optional_import(
            "datasets",
            extra=self.required_extra or "hf",
            purpose="Hugging Face datasets loading",
        )
        load_dataset = datasets_mod.load_dataset

        cfg = dict(identity.resolved_kwargs)
        name = str(cfg["name"])
        config = cfg.get("config")
        text_column = str(cfg.get("text_column", "text"))
        label_column = str(cfg.get("label_column", "label"))

        raw_dir.mkdir(parents=True, exist_ok=True)

        if config is None:
            ds = load_dataset(name, cache_dir=str(raw_dir))
        else:
            ds = load_dataset(name, config, cache_dir=str(raw_dir))

        train = _extract_split(ds, "train", text_column, label_column)
        test = _extract_official_test(
            ds, text_column, label_column, prefer_test=bool(cfg.get("prefer_test_split", True))
        )

        class_filter = _normalize_filter(cfg.get("class_filter"))
        seed = cfg.get("seed")
        train = _apply_limits(
            train,
            class_filter=class_filter,
            max_samples=cfg.get("max_train_samples"),
            seed=seed,
        )
        test = _apply_limits(
            test,
            class_filter=class_filter,
            max_samples=cfg.get("max_test_samples"),
            seed=None if seed is None else int(seed) + 1,
        )

        meta = {
            "provider": "hf",
            "name": name,
            "config": config,
            "text_column": text_column,
            "label_column": label_column,
        }
        return LoadedDataset(train=train, test=test, meta=meta)


def _extract_official_test(
    ds: Any, text_col: str, label_col: str, *, prefer_test: bool
) -> Split | None:
    if not _is_dictlike(ds):
        return None
    candidates = ["test", "validation"] if prefer_test else ["validation", "test"]
    for split_name in candidates:
        if split_name in ds:
            return _extract_split(ds, split_name, text_col, label_col)
    return None


def _extract_split(ds: Any, split_name: str, text_col: str, label_col: str) -> Split:
    split_obj = ds[split_name] if _is_dictlike(ds) else ds
    X = np.asarray(split_obj[text_col], dtype=object)
    y = np.asarray(split_obj[label_col])
    return Split(X=X, y=y)


def _is_dictlike(obj: Any) -> bool:
    return isinstance(obj, Mapping) or (
        hasattr(obj, "__contains__") and hasattr(obj, "__getitem__")
    )
