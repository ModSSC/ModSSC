from __future__ import annotations

import numpy as np


def class_counts(y_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
