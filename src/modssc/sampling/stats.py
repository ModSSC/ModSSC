from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from modssc.sampling.common import class_counts as _class_counts


def class_distribution(y: np.ndarray, idx: np.ndarray) -> dict[str, Any]:
    if idx.size == 0:
        return {"n": 0, "classes": {}}
    y_sub = y[idx]
    classes, counts = _class_counts(y_sub)
    return {
        "n": int(idx.size),
        "classes": {str(c): int(n) for c, n in zip(classes, counts, strict=True)},
    }


def build_inductive_stats(
    *,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_ref: str,
    y_test: np.ndarray | None,
    test_idx: np.ndarray,
    labeled_idx: np.ndarray,
    unlabeled_idx: np.ndarray,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    stats: dict[str, Any] = {"policy": dict(policy)}
    stats["train"] = class_distribution(y_train, train_idx)
    stats["val"] = class_distribution(y_train, val_idx)
    stats["train_labeled"] = class_distribution(y_train, labeled_idx)
    stats["train_unlabeled"] = {"n": int(unlabeled_idx.size)}
    if test_ref == "train":
        stats["test"] = class_distribution(y_train, test_idx)
    else:
        stats["test"] = class_distribution(y_test if y_test is not None else y_train, test_idx)
    return stats


def build_graph_stats(
    *, masks: Mapping[str, np.ndarray], y: np.ndarray, labeled_idx: np.ndarray
) -> dict[str, Any]:
    def mask_count(m: np.ndarray) -> int:
        return int(m.sum())

    stats: dict[str, Any] = {
        "nodes": int(y.shape[0]),
        "train": mask_count(masks["train"]),
        "val": mask_count(masks["val"]),
        "test": mask_count(masks["test"]),
        "labeled": mask_count(masks["labeled"]),
        "unlabeled": mask_count(masks["unlabeled"]),
        "labeled_class_dist": class_distribution(y, labeled_idx),
    }
    return stats
