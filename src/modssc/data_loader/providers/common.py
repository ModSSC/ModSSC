from __future__ import annotations

from typing import Any

import numpy as np

from modssc.data_loader.types import Split


def normalize_filter(values: Any) -> list[Any] | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple, set, np.ndarray)):
        return list(values)
    return [values]


def apply_class_filter(
    X: np.ndarray, y: np.ndarray, *, class_filter: list[Any] | None
) -> tuple[np.ndarray, np.ndarray]:
    if class_filter is None:
        return X, y
    mask = np.isin(y, np.asarray(class_filter))
    return X[mask], y[mask]


def limit_samples(
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


def apply_limits(
    X: np.ndarray,
    y: np.ndarray,
    *,
    class_filter: list[Any] | None,
    max_samples: int | None,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = apply_class_filter(X, y, class_filter=class_filter)
    X, y = limit_samples(X, y, max_samples=max_samples, seed=seed)
    return X, y


def apply_limits_to_split(
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
    X, y = apply_limits(
        X,
        y,
        class_filter=class_filter,
        max_samples=max_samples,
        seed=seed,
    )
    return Split(X=X, y=y)
