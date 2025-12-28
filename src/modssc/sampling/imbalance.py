from __future__ import annotations

import numpy as np

from modssc.sampling.plan import ImbalanceSpec


def apply_imbalance(
    *,
    idx: np.ndarray,
    y: np.ndarray,
    spec: ImbalanceSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    if spec.kind == "none" or idx.size == 0:
        return np.sort(idx)

    y_sub = y[idx]
    classes, counts = np.unique(y_sub, return_counts=True)

    if spec.kind == "subsample_max_per_class":
        if spec.max_per_class is None:
            raise ValueError("max_per_class is required for subsample_max_per_class")
        cap = int(spec.max_per_class)
        if cap < 1:
            raise ValueError("max_per_class must be >= 1")
        parts: list[np.ndarray] = []
        for cls in classes:
            cls_idx = idx[y_sub == cls]
            if cls_idx.size <= cap:
                parts.append(cls_idx)
            else:
                parts.append(rng.choice(cls_idx, size=cap, replace=False))
        return np.sort(np.concatenate(parts)) if parts else np.asarray([], dtype=np.int64)

    if spec.kind == "long_tail":
        if spec.alpha is None:
            raise ValueError("alpha is required for long_tail")
        alpha = float(spec.alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        min_per_class = int(spec.min_per_class)

        order = np.argsort(-counts)  # rank by frequency
        max_count = int(counts[order[0]]) if order.size else 0
        desired = np.zeros_like(counts)
        for rank, i in enumerate(order):
            desired_i = int(round(max_count * (alpha**rank)))
            desired_i = max(desired_i, min_per_class)
            desired[i] = min(desired_i, int(counts[i]))

        parts = []
        for cls, n_sel in zip(classes, desired, strict=True):
            cls_idx = idx[y_sub == cls]
            if int(n_sel) >= cls_idx.size:
                parts.append(cls_idx)
            else:
                parts.append(rng.choice(cls_idx, size=int(n_sel), replace=False))
        return np.sort(np.concatenate(parts)) if parts else np.asarray([], dtype=np.int64)

    raise ValueError(f"Unknown imbalance kind: {spec.kind!r}")
