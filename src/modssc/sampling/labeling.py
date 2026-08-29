from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from modssc.sampling.common import class_counts as _class_counts
from modssc.sampling.errors import SamplingValidationError
from modssc.sampling.plan import LabelingSpec

logger = logging.getLogger(__name__)


def _validate_labeled_indices(*, labeled: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    if labeled.size > train_idx.size:
        raise SamplingValidationError("labeled size cannot exceed train size")
    if np.unique(labeled).size != labeled.size:
        raise SamplingValidationError("labeled contains duplicates")
    if np.setdiff1d(labeled, train_idx).size:
        raise SamplingValidationError("labeled indices must be a subset of train indices")
    return labeled


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_from_artifact(
    *,
    spec: LabelingSpec,
    run_seed: int | None,
    train_idx: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    artifact = spec.fixed_indices_artifact
    if artifact is None:  # pragma: no cover - caller guards this
        raise AssertionError("fixed_indices_artifact is required")
    if run_seed is None or isinstance(run_seed, bool) or int(run_seed) < 0:
        raise SamplingValidationError("fixed_indices_artifact requires a non-negative run_seed")
    source = Path(artifact.path).expanduser().resolve()
    if not source.is_file():
        raise SamplingValidationError(f"fixed-indices artifact is missing: {source}")
    actual_sha256 = _sha256_file(source)
    if actual_sha256 != artifact.sha256:
        raise SamplingValidationError(
            "fixed-indices artifact SHA-256 differs: "
            f"computed {actual_sha256}, expected {artifact.sha256}"
        )
    try:
        with np.load(source, allow_pickle=False) as archive:
            metadata = _read_fixed_indices_metadata(archive)
            if metadata.get("source_sha256") != artifact.source_sha256:
                raise SamplingValidationError(
                    "fixed-indices artifact source SHA-256 differs from the sampling plan"
                )
            if metadata.get("source_key") != artifact.key:
                raise SamplingValidationError(
                    "fixed-indices artifact source key differs from the sampling plan"
                )
            values = _read_int64_vector(archive, "values")
            offsets = _read_int64_vector(archive, "offsets")
            if offsets.size < 1 or offsets[0] != 0 or offsets[-1] != values.size:
                raise SamplingValidationError("fixed-indices artifact offsets are invalid")
            if np.any(np.diff(offsets) < 0):
                raise SamplingValidationError("fixed-indices artifact offsets are not monotonic")
            expected_rows = metadata.get("row_count")
            if (
                isinstance(expected_rows, bool)
                or not isinstance(expected_rows, int)
                or expected_rows != int(offsets.size - 1)
            ):
                raise SamplingValidationError("fixed-indices artifact row count is invalid")
            index = int(run_seed) * int(artifact.index_stride) + int(artifact.index_offset)
            if index >= int(offsets.size - 1):
                raise SamplingValidationError(
                    f"fixed-indices artifact has no row for run_seed={run_seed}"
                )
            start, stop = int(offsets[index]), int(offsets[index + 1])
            labeled = values[start:stop].copy()
    except SamplingValidationError:
        raise
    except Exception as exc:
        raise SamplingValidationError(f"cannot load fixed-indices artifact: {source}") from exc

    if artifact.expected_size is not None and labeled.size != int(artifact.expected_size):
        raise SamplingValidationError(
            "fixed-indices artifact row has the wrong size: "
            f"{labeled.size} != {artifact.expected_size}"
        )
    labeled = _validate_labeled_indices(labeled=labeled, train_idx=train_idx)
    if artifact.expected_per_class is not None:
        classes = np.unique(y[train_idx])
        counts = np.asarray(
            [np.count_nonzero(y[labeled] == label) for label in classes],
            dtype=np.int64,
        )
        if not np.all(counts == int(artifact.expected_per_class)):
            raise SamplingValidationError(
                "fixed-indices artifact row does not have the expected per-class count"
            )
    return np.sort(labeled)


def _read_int64_vector(archive: object, key: str) -> np.ndarray:
    files = getattr(archive, "files", ())
    if key not in files:
        raise SamplingValidationError(f"fixed-indices artifact has no {key!r} array")
    raw = np.asarray(archive[key])  # type: ignore[index]
    if raw.ndim != 1 or raw.dtype.kind not in ("i", "u"):
        raise SamplingValidationError(
            f"fixed-indices artifact {key!r} must be a one-dimensional integer array"
        )
    return np.asarray(raw, dtype=np.int64)


def _read_fixed_indices_metadata(archive: object) -> Mapping[str, object]:
    files = getattr(archive, "files", ())
    if "metadata_json" not in files:
        raise SamplingValidationError("fixed-indices artifact has no 'metadata_json' array")
    encoded = np.asarray(archive["metadata_json"])  # type: ignore[index]
    if encoded.ndim != 1 or encoded.dtype != np.uint8:
        raise SamplingValidationError("fixed-indices artifact metadata_json must be a uint8 vector")
    try:
        metadata = json.loads(encoded.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SamplingValidationError("fixed-indices artifact metadata is invalid") from exc
    if not isinstance(metadata, Mapping):
        raise SamplingValidationError("fixed-indices artifact metadata must be a mapping")
    if metadata.get("schema_version") != 1:
        raise SamplingValidationError("fixed-indices artifact metadata schema_version must be 1")
    if metadata.get("format") != "ragged_int64_v1":
        raise SamplingValidationError("fixed-indices artifact format must be ragged_int64_v1")
    return metadata


def select_labeled(
    *,
    train_idx: np.ndarray,
    y: np.ndarray,
    spec: LabelingSpec,
    rng: np.random.Generator | np.random.RandomState,
    run_seed: int | None = None,
) -> np.ndarray:
    if train_idx.size == 0:
        if spec.class_counts is not None:
            raise SamplingValidationError(
                "class_counts cannot be satisfied by an empty train partition"
            )
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
    if spec.fixed_indices_artifact is not None:
        return _select_from_artifact(
            spec=spec,
            run_seed=run_seed,
            train_idx=train_idx,
            y=y,
        )

    if spec.class_counts is not None:
        if spec.mode != "count" or isinstance(spec.value, bool):
            raise SamplingValidationError(
                "class_counts requires labeling mode='count' and an integer value"
            )
        if any(
            isinstance(count, bool) or not isinstance(count, (int, np.integer)) or int(count) < 0
            for count in spec.class_counts.values()
        ):
            raise SamplingValidationError("class_counts values must be non-negative integers")
        requested_total = sum(int(count) for count in spec.class_counts.values())
        if requested_total <= 0 or int(spec.value) != requested_total:
            raise SamplingValidationError(
                "class_counts must request at least one sample and sum to labeling.value"
            )
        if spec.selection_order not in {"choice", "permutation"}:
            raise SamplingValidationError(
                f"Unknown labeling selection_order: {spec.selection_order!r}"
            )
        y_train = y[train_idx]
        classes = np.unique(y_train)
        available_by_label = {str(label): label for label in classes.tolist()}
        normalized_labels = [str(label) for label in spec.class_counts]
        if len(set(normalized_labels)) != len(normalized_labels):
            raise SamplingValidationError(
                "class_counts contains duplicate labels after normalization"
            )
        requested_by_label = {str(label): int(count) for label, count in spec.class_counts.items()}
        if set(requested_by_label) != set(available_by_label):
            raise SamplingValidationError(
                "class_counts labels must exactly match the classes in the train partition"
            )
        labeled_parts: list[np.ndarray] = []
        for label_key in sorted(requested_by_label):
            label = available_by_label[label_key]
            requested = requested_by_label[label_key]
            class_indices = train_idx[y_train == label]
            if requested > int(class_indices.size):
                raise SamplingValidationError(
                    f"class_counts requests {requested} samples for class {label_key!r}, "
                    f"but only {int(class_indices.size)} are available"
                )
            if requested == 0:
                continue
            if spec.selection_order == "permutation":
                chosen = rng.permutation(class_indices)[:requested]
            else:
                chosen = rng.choice(class_indices, size=requested, replace=False)
            labeled_parts.append(np.asarray(chosen, dtype=np.int64))
        if not labeled_parts:  # pragma: no cover - positive total is validated above
            raise RuntimeError("class_counts positive-total invariant failed")
        labeled = np.sort(np.concatenate(labeled_parts))
        return _validate_labeled_indices(labeled=labeled, train_idx=train_idx)

    y_train = y[train_idx]
    classes, counts = _class_counts(y_train)
    n_classes = int(classes.size)

    if spec.strategy not in ("proportional", "balanced", "random"):
        raise ValueError(f"Unknown labeling strategy: {spec.strategy!r}")

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
    requested_target = int(target)
    min_per_class = int(spec.min_per_class)

    if spec.strategy == "random":
        if spec.selection_order == "permutation":
            selected = rng.permutation(train_idx)[:target]
        else:
            selected = rng.choice(train_idx, size=target, replace=False)
        labeled = np.sort(np.asarray(selected, dtype=np.int64))
        return _validate_labeled_indices(labeled=labeled, train_idx=train_idx)

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
        if spec.selection_order == "permutation":
            chosen = rng.permutation(cls_idx)[: int(n_sel)]
        else:
            chosen = rng.choice(cls_idx, size=int(n_sel), replace=False)
        labeled_parts.append(np.asarray(chosen, dtype=np.int64))

    labeled = (
        np.sort(np.concatenate(labeled_parts)) if labeled_parts else np.asarray([], dtype=np.int64)
    )

    if int(labeled.size) != requested_target:
        logger.warning(
            "Labeling target adjusted: requested=%s effective=%s mode=%s strategy=%s "
            "min_per_class=%s n_classes=%s train=%s per_class=%s",
            requested_target,
            int(labeled.size),
            spec.mode,
            spec.strategy,
            min_per_class,
            n_classes,
            int(train_idx.size),
            {str(cls): int(n_sel) for cls, n_sel in zip(classes, per_class, strict=True)},
        )

    return _validate_labeled_indices(labeled=labeled, train_idx=train_idx)
