from __future__ import annotations

import numpy as np


def _class_counts(y_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_sub = np.asarray(y_sub)
    if y_sub.size == 0:
        return np.asarray([], dtype=y_sub.dtype), np.asarray([], dtype=np.int64)
    if y_sub.dtype.kind in {"i", "u"}:
        y_int = y_sub.astype(np.int64, copy=False)
        min_label = int(y_int.min())
        max_label = int(y_int.max())
        if min_label >= 0 and max_label <= 1_000_000:
            counts = np.bincount(y_int, minlength=max_label + 1)
            classes = np.nonzero(counts)[0].astype(y_sub.dtype, copy=False)
            return classes, counts[classes]
    return np.unique(y_sub, return_counts=True)


def random_split(
    indices: np.ndarray, *, n_holdout: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if n_holdout <= 0:
        return indices.copy(), np.asarray([], dtype=np.int64)
    if n_holdout >= indices.size:
        return np.asarray([], dtype=np.int64), indices.copy()
    perm = rng.permutation(indices)
    holdout = np.sort(perm[:n_holdout])
    keep = np.sort(perm[n_holdout:])
    return keep, holdout


def stratified_holdout(
    indices: np.ndarray,
    y: np.ndarray,
    *,
    n_holdout: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    # If y cannot be indexed, fallback to random.
    if n_holdout <= 0:
        return np.sort(indices), np.asarray([], dtype=np.int64)
    if n_holdout >= indices.size:
        return np.asarray([], dtype=np.int64), np.sort(indices)

    y_sub = y[indices]
    classes, counts = _class_counts(y_sub)
    total = int(indices.size)

    expected = (counts.astype(float) * float(n_holdout)) / float(total)
    base = np.floor(expected).astype(int)
    remainder = expected - base

    # distribute remaining counts
    remaining = int(n_holdout - base.sum())
    if remaining > 0:
        order = np.argsort(-remainder)
        for i in order[:remaining]:
            base[i] += 1

    holdout_parts: list[np.ndarray] = []
    keep_parts: list[np.ndarray] = []
    for cls, n_cls_hold in zip(classes, base, strict=True):
        cls_idx = indices[y_sub == cls]
        cls_idx = rng.permutation(cls_idx)
        n_cls_hold = int(min(max(n_cls_hold, 0), cls_idx.size))
        holdout_parts.append(cls_idx[:n_cls_hold])
        keep_parts.append(cls_idx[n_cls_hold:])

    holdout = (
        np.sort(np.concatenate(holdout_parts)) if holdout_parts else np.asarray([], dtype=np.int64)
    )
    keep = np.sort(np.concatenate(keep_parts)) if keep_parts else np.asarray([], dtype=np.int64)

    return keep, holdout


def make_holdout_split(
    *,
    n_samples: int,
    y: np.ndarray,
    test_fraction: float,
    val_fraction: float,
    stratify: bool,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")

    all_idx = np.arange(n_samples, dtype=np.int64)

    n_test = int(round(float(test_fraction) * float(n_samples)))
    n_test = max(0, min(n_samples, n_test))

    if stratify:
        train_val, test = stratified_holdout(all_idx, y, n_holdout=n_test, rng=rng)
    else:
        train_val, test = random_split(all_idx, n_holdout=n_test, rng=rng)

    n_train_val = int(train_val.size)
    n_val = int(round(float(val_fraction) * float(n_train_val)))
    n_val = max(0, min(n_train_val, n_val))

    if stratify:
        train, val = stratified_holdout(train_val, y, n_holdout=n_val, rng=rng)
    else:
        train, val = random_split(train_val, n_holdout=n_val, rng=rng)

    return {"train": train, "val": val, "test": test}


def make_kfold_split(
    *,
    n_samples: int,
    y: np.ndarray,
    k: int,
    fold: int,
    stratify: bool,
    shuffle: bool,
    val_fraction: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    if k <= 1:
        raise ValueError("k must be >= 2")
    if fold < 0 or fold >= k:
        raise ValueError("fold must satisfy 0 <= fold < k")

    all_idx = np.arange(n_samples, dtype=np.int64)

    folds = (
        stratified_kfold(all_idx, y, k=k, rng=rng, shuffle=shuffle)
        if stratify
        else plain_kfold(all_idx, k=k, rng=rng, shuffle=shuffle)
    )
    test = np.sort(folds[fold])
    train_pool = np.sort(np.concatenate([f for i, f in enumerate(folds) if i != fold]))

    if val_fraction > 0.0:
        n_val = int(round(float(val_fraction) * float(train_pool.size)))
        n_val = max(0, min(train_pool.size, n_val))
        if stratify:
            train, val = stratified_holdout(train_pool, y, n_holdout=n_val, rng=rng)
        else:
            train, val = random_split(train_pool, n_holdout=n_val, rng=rng)
    else:
        train = train_pool
        val = np.asarray([], dtype=np.int64)

    return {"train": train, "val": val, "test": test}


def plain_kfold(
    indices: np.ndarray, *, k: int, rng: np.random.Generator, shuffle: bool
) -> list[np.ndarray]:
    idx = indices.copy()
    if shuffle:
        idx = rng.permutation(idx)
    parts = np.array_split(idx, k)
    return [np.asarray(p, dtype=np.int64) for p in parts]


def stratified_kfold(
    indices: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> list[np.ndarray]:
    y_sub = y[indices]
    classes = np.unique(y_sub)
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(k)]

    for cls in classes:
        cls_idx = indices[y_sub == cls]
        cls_idx = cls_idx.copy()
        if shuffle:
            cls_idx = rng.permutation(cls_idx)
        chunks = np.array_split(cls_idx, k)
        for i, c in enumerate(chunks):
            fold_parts[i].append(np.asarray(c, dtype=np.int64))

    folds: list[np.ndarray] = []
    for i in range(k):
        if fold_parts[i]:
            fold = np.concatenate(fold_parts[i])
            if shuffle:
                fold = rng.permutation(fold)
            folds.append(np.asarray(fold, dtype=np.int64))
        else:
            folds.append(np.asarray([], dtype=np.int64))
    return folds
