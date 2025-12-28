from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_loader.optional import optional_import
from modssc.data_loader.providers.base import BaseProvider
from modssc.data_loader.types import DatasetIdentity, LoadedDataset, Split
from modssc.data_loader.uri import ParsedURI


class TorchvisionProvider(BaseProvider):
    name = "torchvision"
    required_extra = "vision"

    def resolve(self, parsed: ParsedURI, *, options: Mapping[str, Any]) -> DatasetIdentity:
        dataset_class = parsed.reference.strip()
        resolved_kwargs = {
            "dataset_class": dataset_class,
            "class_filter": _normalize_filter(options.get("class_filter")),
            "max_train_samples": options.get("max_train_samples"),
            "max_test_samples": options.get("max_test_samples"),
            "seed": options.get("seed"),
        }
        return DatasetIdentity(
            provider=self.name,
            canonical_uri=f"torchvision:{dataset_class}",
            dataset_id=dataset_class,
            version=None,
            modality="vision",
            task=str(options.get("task", "classification")),
            required_extra=self.required_extra,
            resolved_kwargs=resolved_kwargs,
        )

    def load_canonical(self, identity: DatasetIdentity, *, raw_dir: Path) -> LoadedDataset:
        tv_datasets = optional_import(
            "torchvision.datasets",
            extra=self.required_extra or "vision",
            purpose="torchvision dataset loading",
        )
        ds_cls = getattr(tv_datasets, str(identity.resolved_kwargs["dataset_class"]))

        root = raw_dir / "source"
        root.mkdir(parents=True, exist_ok=True)

        train_ds, test_ds = _make_train_test(ds_cls, root)
        X_train, y_train = _extract_xy(train_ds)
        X_test, y_test = _extract_xy(test_ds)

        class_filter = _normalize_filter(identity.resolved_kwargs.get("class_filter"))
        seed = identity.resolved_kwargs.get("seed")
        X_train, y_train = _apply_limits(
            X_train,
            y_train,
            class_filter=class_filter,
            max_samples=identity.resolved_kwargs.get("max_train_samples"),
            seed=seed,
        )
        X_test, y_test = _apply_limits(
            X_test,
            y_test,
            class_filter=class_filter,
            max_samples=identity.resolved_kwargs.get("max_test_samples"),
            seed=None if seed is None else int(seed) + 1,
        )

        meta = {"provider": "torchvision", "dataset_class": identity.dataset_id}
        return LoadedDataset(
            train=Split(X=X_train, y=y_train), test=Split(X=X_test, y=y_test), meta=meta
        )


def _make_train_test(ds_cls: Any, root: Path) -> tuple[Any, Any]:
    try:
        train_ds = ds_cls(root=str(root), train=True, download=True)
        test_ds = ds_cls(root=str(root), train=False, download=True)
        return train_ds, test_ds
    except TypeError:
        # some datasets use split="train"/"test"
        train_ds = ds_cls(root=str(root), split="train", download=True)
        test_ds = ds_cls(root=str(root), split="test", download=True)
        return train_ds, test_ds


def _extract_xy(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(dataset, "data") and hasattr(dataset, "targets"):
        X = np.asarray(dataset.data)
        y = np.asarray(dataset.targets)
        return X, y

    X_list: list[Any] = []
    y_list: list[Any] = []
    for sample in dataset:
        X_list.append(sample[0])
        y_list.append(sample[1])
    return np.asarray(X_list, dtype=object), np.asarray(y_list)


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
    X: np.ndarray,
    y: np.ndarray,
    *,
    class_filter: list[Any] | None,
    max_samples: int | None,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = _apply_class_filter(X, y, class_filter=class_filter)
    X, y = _limit_samples(X, y, max_samples=max_samples, seed=seed)
    return X, y
