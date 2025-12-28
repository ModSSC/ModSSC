from __future__ import annotations

import logging

import numpy as np

from modssc.sampling.errors import SamplingValidationError
from modssc.sampling.plan import LabelingSpec

logger = logging.getLogger(__name__)


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


def select_labeled(
    *,
    train_idx: np.ndarray,
    y: np.ndarray,
    spec: LabelingSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    if train_idx.size == 0:
        return np.asarray([], dtype=np.int64)

    if spec.fixed_indices is not None:
        fixed = np.asarray([int(i) for i in spec.fixed_indices], dtype=np.int64)
        # validate membership
        if np.setdiff1d(fixed, train_idx).size:
            raise SamplingValidationError("fixed_indices must be a subset of train indices")
        if np.unique(fixed).size != fixed.size:
            logger.debug("fixed_indices contains duplicates")
            raise SamplingValidationError("fixed_indices contains duplicates")
        return np.sort(fixed)

    y_train = y[train_idx]
    classes, counts = _class_counts(y_train)
    n_classes = int(classes.size)

    if spec.mode == "fraction":
        frac = float(spec.value)
        if not (0.0 < frac <= 1.0):
            raise ValueError("label fraction must be in (0, 1]")
        target = int(round(frac * float(train_idx.size)))
    elif spec.mode == "count":
        target = int(spec.value)
    elif spec.mode == "per_class":
        target = int(spec.value) * n_classes
    else:
        raise ValueError(f"Unknown labeling mode: {spec.mode!r}")

    target = max(0, min(int(train_idx.size), target))
    min_per_class = int(spec.min_per_class)

    # allocation per class
    per_class = np.zeros(n_classes, dtype=int)
    if spec.mode == "per_class":
        per_class[:] = int(spec.value)
    elif spec.strategy == "balanced" and target > 0:
        base = target // n_classes
        rem = target % n_classes
        per_class[:] = base
        # distribute remainder to largest classes
        order = np.argsort(-counts)
        for i in order[:rem]:
            per_class[i] += 1
    else:
        # proportional
        expected = (counts.astype(float) * float(target)) / float(train_idx.size)
        per_class = np.floor(expected).astype(int)
        rem = int(target - per_class.sum())
        if rem > 0:
            order = np.argsort(-(expected - per_class))
            for i in order[:rem]:
                per_class[i] += 1

    # enforce min_per_class when possible
    for i in range(n_classes):
        if counts[i] >= min_per_class:
            per_class[i] = max(per_class[i], min_per_class)

    # cap by available
    per_class = np.minimum(per_class, counts)

    # adjust total to target (best effort)
    total = int(per_class.sum())
    if total > target:
        # remove from largest allocations
        order = np.argsort(-per_class)
        i = 0
        while total > target and i < order.size:
            j = int(order[i])
            if per_class[j] > 0 and (counts[j] < min_per_class or per_class[j] > min_per_class):
                per_class[j] -= 1
                total -= 1
            else:
                i += 1
    elif total < target:
        # add where possible
        order = np.argsort(-(counts - per_class))
        i = 0
        while total < target and i < order.size:
            j = int(order[i])
            if per_class[j] < counts[j]:
                per_class[j] += 1
                total += 1
            else:
                i += 1

    # sample indices per class
    labeled_parts: list[np.ndarray] = []
    for cls, n_sel in zip(classes, per_class, strict=True):
        cls_idx = train_idx[y_train == cls]
        if n_sel <= 0:
            continue
        chosen = rng.choice(cls_idx, size=int(n_sel), replace=False)
        labeled_parts.append(np.asarray(chosen, dtype=np.int64))

    labeled = (
        np.sort(np.concatenate(labeled_parts)) if labeled_parts else np.asarray([], dtype=np.int64)
    )

    # final guard
    if labeled.size > train_idx.size:
        raise SamplingValidationError("labeled size cannot exceed train size")
    if np.unique(labeled).size != labeled.size:
        raise SamplingValidationError("labeled contains duplicates")

    return labeled
