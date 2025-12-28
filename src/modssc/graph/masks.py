from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import GraphValidationError


@dataclass(frozen=True)
class Masks:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    unlabeled: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "unlabeled": self.unlabeled,
        }


def masks_from_indices(
    *,
    n: int,
    train_idx: Any,
    val_idx: Any,
    test_idx: Any,
    labeled_idx: Any | None = None,
) -> Masks:
    n = int(n)
    if n <= 0:
        raise GraphValidationError("n must be positive")

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    for name, idx in ("train", train_idx), ("val", val_idx), ("test", test_idx):
        if idx.ndim != 1:
            raise GraphValidationError(f"{name}_idx must be 1D")
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise GraphValidationError(f"{name}_idx contains indices outside [0, n)")

    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)

    train[train_idx] = True
    val[val_idx] = True
    test[test_idx] = True

    if labeled_idx is None:
        unlabeled = ~train
    else:
        labeled_idx = np.asarray(labeled_idx, dtype=np.int64)
        if labeled_idx.ndim != 1:
            raise GraphValidationError("labeled_idx must be 1D")
        if labeled_idx.size and (labeled_idx.min() < 0 or labeled_idx.max() >= n):
            raise GraphValidationError("labeled_idx contains indices outside [0, n)")
        labeled = np.zeros(n, dtype=bool)
        labeled[labeled_idx] = True
        unlabeled = train & (~labeled)

    return Masks(train=train, val=val, test=test, unlabeled=unlabeled)
