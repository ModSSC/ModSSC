from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_loader.optional import optional_import
from modssc.data_loader.providers.base import BaseProvider
from modssc.data_loader.types import DatasetIdentity, LoadedDataset, Split
from modssc.data_loader.uri import ParsedURI


class TorchaudioProvider(BaseProvider):
    """Provider for datasets shipped in torchaudio.

    Design choice:
    For audio workloads, storing raw file paths is usually preferable to storing
    decoded waveforms in the processed cache (smaller, faster, framework-agnostic).

    This provider therefore tries to build X as an object array of file paths when
    possible (using torchaudio's internal walker). If that is not available, it
    falls back to iterating and storing waveforms as numpy arrays.
    """

    name = "torchaudio"
    required_extra = "audio"

    def resolve(self, parsed: ParsedURI, *, options: Mapping[str, Any]) -> DatasetIdentity:
        dataset_class = str(options.get("dataset_class") or parsed.reference).strip()
        if not dataset_class:
            dataset_class = parsed.reference

        canonical_uri = f"torchaudio:{dataset_class}"
        resolved_kwargs = {
            "dataset_class": dataset_class,
            "class_filter": _normalize_filter(options.get("class_filter")),
            "max_train_samples": options.get("max_train_samples"),
            "max_test_samples": options.get("max_test_samples"),
            "seed": options.get("seed"),
        }
        return DatasetIdentity(
            provider=self.name,
            canonical_uri=canonical_uri,
            dataset_id=dataset_class,
            version=None,
            modality="audio",
            task=str(options.get("task", "classification")),
            required_extra=self.required_extra,
            resolved_kwargs=resolved_kwargs,
        )

    def load_canonical(self, identity: DatasetIdentity, *, raw_dir: Path) -> LoadedDataset:
        ta_datasets = optional_import(
            "torchaudio.datasets",
            extra=self.required_extra or "audio",
            purpose="torchaudio dataset loading",
        )
        ds_cls = getattr(ta_datasets, str(identity.resolved_kwargs["dataset_class"]))

        root = raw_dir / "source"
        root.mkdir(parents=True, exist_ok=True)

        train_ds, test_ds = _official_or_single(ds_cls, root)

        X_train, y_train = _extract_xy(train_ds, dataset_class=str(identity.dataset_id))
        class_filter = _normalize_filter(identity.resolved_kwargs.get("class_filter"))
        seed = identity.resolved_kwargs.get("seed")
        X_train, y_train = _apply_limits(
            X_train,
            y_train,
            class_filter=class_filter,
            max_samples=identity.resolved_kwargs.get("max_train_samples"),
            seed=seed,
        )
        train = Split(X=X_train, y=y_train)

        test = None
        if test_ds is not None:
            X_test, y_test = _extract_xy(test_ds, dataset_class=str(identity.dataset_id))
            X_test, y_test = _apply_limits(
                X_test,
                y_test,
                class_filter=class_filter,
                max_samples=identity.resolved_kwargs.get("max_test_samples"),
                seed=None if seed is None else int(seed) + 1,
            )
            test = Split(X=X_test, y=y_test)

        meta = {
            "provider": "torchaudio",
            "dataset_class": identity.dataset_id,
            "representation": "paths_or_waveforms",
        }
        return LoadedDataset(train=train, test=test, meta=meta)


def _official_or_single(ds_cls: Any, root: Path) -> tuple[Any, Any | None]:
    """Return (train, test) when the dataset supports official subsets, else (ds, None)."""
    try:
        train_ds = ds_cls(root=str(root), subset="training", download=True)
        test_ds = ds_cls(root=str(root), subset="testing", download=True)
        return train_ds, test_ds
    except TypeError:
        ds = ds_cls(root=str(root), download=True)
        return ds, None


def _paths_from_walker(dataset: Any) -> list[Path] | None:
    base = getattr(dataset, "_path", None)
    walker = getattr(dataset, "_walker", None)
    if base is None or walker is None:
        return None
    base_path = Path(str(base))
    paths: list[Path] = []
    for entry in list(walker):
        p = Path(str(entry))
        if not p.is_absolute():
            p = base_path / p

        # Fix for datasets (e.g. YESNO) where walker entries lack extension
        if not p.exists():
            if p.with_suffix(".wav").exists():
                p = p.with_suffix(".wav")
            elif p.with_suffix(".flac").exists():
                p = p.with_suffix(".flac")

        paths.append(p)
    return paths


def _labels_from_paths(paths: Sequence[Path], *, dataset_class: str) -> list[Any]:
    cls = dataset_class.strip().upper()
    if cls == "SPEECHCOMMANDS":
        # label is the directory name, ex: .../yes/xxx.wav
        return [p.parent.name for p in paths]
    if cls == "YESNO":
        # label is encoded in filename, ex: 1_0_1_0_0_1_0_1.wav
        # Use simple binary classification based on the first item.
        return ["yes" if p.stem.startswith("1") else "no" for p in paths]
    # reasonable fallback
    return [p.parent.name for p in paths]


def _as_object_vector(items: Sequence[Any]) -> np.ndarray:
    arr = np.empty((len(items),), dtype=object)
    for i, v in enumerate(items):
        arr[i] = v
    return arr


def _extract_xy(dataset: Any, *, dataset_class: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) from a torchaudio dataset.

    Prefer returning file paths when the dataset exposes a walker, otherwise
    fall back to decoded waveforms.
    """
    paths = _paths_from_walker(dataset)
    if paths is not None and len(paths) > 0:
        X = _as_object_vector([str(p) for p in paths])
        y = _as_object_vector(_labels_from_paths(paths, dataset_class=dataset_class))
        return X, y

    # Fallback: iterate samples and store waveforms as numpy arrays (object vector).
    from modssc.data_loader.numpy_adapter import to_numpy

    X_list: list[Any] = []
    y_list: list[Any] = []
    for sample in dataset:
        if len(sample) < 3:
            raise ValueError(
                "Expected torchaudio sample with at least 3 elements (waveform, sr, label, ...)."
            )
        X_list.append(to_numpy(sample[0], allow_object=True))
        y_list.append(sample[2])
    return _as_object_vector(X_list), _as_object_vector(y_list)


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
