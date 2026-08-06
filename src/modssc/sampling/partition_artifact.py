from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from modssc.sampling.errors import SamplingValidationError
from modssc.sampling.plan import OrderedPartitionArtifactSpec

_ARRAY_NAMES = (
    "train",
    "val",
    "test",
    "train_labeled",
    "train_unlabeled",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_metadata(archive: Any, *, source: Path) -> Mapping[str, Any]:
    if "metadata_json" not in archive.files:
        raise SamplingValidationError("ordered partition artifact has no 'metadata_json' array")
    try:
        encoded = np.asarray(archive["metadata_json"])
        if encoded.ndim != 1 or encoded.dtype != np.uint8:
            raise SamplingValidationError(
                "ordered partition artifact metadata_json must be a uint8 vector"
            )
        decoded = json.loads(encoded.tobytes().decode("utf-8"))
    except SamplingValidationError:
        raise
    except Exception as exc:
        raise SamplingValidationError(
            f"ordered partition artifact metadata is invalid: {source}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise SamplingValidationError("ordered partition artifact metadata must be a mapping")
    if decoded.get("schema_version") != 1:
        raise SamplingValidationError(
            "ordered partition artifact metadata schema_version must be 1"
        )
    return decoded


def _read_index_vector(archive: Any, key: str) -> np.ndarray:
    raw = np.asarray(archive[key])
    if raw.ndim != 1 or raw.dtype.kind not in ("i", "u"):
        raise SamplingValidationError(
            f"ordered partition artifact {key!r} must be a one-dimensional integer array"
        )
    if raw.dtype.kind == "u" and raw.size and int(raw.max()) > np.iinfo(np.int64).max:
        raise SamplingValidationError(
            f"ordered partition artifact {key!r} contains indices outside int64 range"
        )
    return np.asarray(raw, dtype=np.int64).copy()


def _validate_metadata(
    metadata: Mapping[str, Any],
    *,
    spec: OrderedPartitionArtifactSpec,
    run_seed: int,
    n_train: int,
    n_test: int | None,
    dataset_fingerprint: str | None,
) -> None:
    if metadata.get("unlabeled_pool") != spec.unlabeled_pool:
        raise SamplingValidationError(
            "ordered partition artifact unlabeled_pool differs from the sampling plan"
        )
    if metadata.get("test_ref") != spec.test_ref:
        raise SamplingValidationError(
            "ordered partition artifact test_ref differs from the sampling plan"
        )
    if int(metadata.get("train_source_size", -1)) != int(n_train):
        raise SamplingValidationError(
            "ordered partition artifact train_source_size differs from the dataset"
        )
    expected_fingerprint = metadata.get("dataset_fingerprint")
    if expected_fingerprint is not None and str(expected_fingerprint) != str(dataset_fingerprint):
        raise SamplingValidationError(
            "ordered partition artifact dataset_fingerprint differs from the dataset"
        )
    if spec.test_ref == "test":
        if n_test is None:
            raise SamplingValidationError(
                "ordered partition artifact requires an official test split"
            )
        if int(metadata.get("test_source_size", -1)) != int(n_test):
            raise SamplingValidationError(
                "ordered partition artifact test_source_size differs from the dataset"
            )
    seeds = metadata.get("seeds")
    if (
        isinstance(seeds, (str, bytes))
        or not isinstance(seeds, list)
        or int(run_seed) not in [int(seed) for seed in seeds]
    ):
        raise SamplingValidationError(
            f"ordered partition artifact metadata has no run_seed={run_seed}"
        )


def _validate_expected_sizes(
    indices: Mapping[str, np.ndarray],
    spec: OrderedPartitionArtifactSpec,
) -> None:
    expected = {
        "train": spec.expected_train_size,
        "val": spec.expected_val_size,
        "test": spec.expected_test_size,
        "train_labeled": spec.expected_labeled_size,
        "train_unlabeled": spec.expected_unlabeled_size,
    }
    for name, size in expected.items():
        if size is not None and int(indices[name].size) != int(size):
            raise SamplingValidationError(
                f"ordered partition artifact {name} has the wrong size: "
                f"{indices[name].size} != {size}"
            )


def load_ordered_partition(
    *,
    spec: OrderedPartitionArtifactSpec,
    run_seed: int,
    y_train: np.ndarray,
    n_test: int | None,
    dataset_fingerprint: str | None = None,
) -> dict[str, np.ndarray]:
    """Load one authenticated partition without changing any array order."""

    if isinstance(run_seed, bool) or int(run_seed) < 0:
        raise SamplingValidationError("ordered partition artifact requires a non-negative run_seed")
    source = Path(spec.path).expanduser().resolve()
    if not source.is_file():
        raise SamplingValidationError(f"ordered partition artifact is missing: {source}")
    actual_sha256 = _sha256_file(source)
    if actual_sha256 != spec.sha256:
        raise SamplingValidationError(
            "ordered partition artifact SHA-256 differs: "
            f"computed {actual_sha256}, expected {spec.sha256}"
        )

    prefix = f"seed_{int(run_seed)}__"
    try:
        with np.load(source, allow_pickle=False) as archive:
            metadata = _read_metadata(archive, source=source)
            _validate_metadata(
                metadata,
                spec=spec,
                run_seed=int(run_seed),
                n_train=int(y_train.shape[0]),
                n_test=n_test,
                dataset_fingerprint=dataset_fingerprint,
            )
            missing = [name for name in _ARRAY_NAMES if f"{prefix}{name}" not in archive.files]
            if missing:
                joined = ", ".join(missing)
                raise SamplingValidationError(
                    f"ordered partition artifact is missing arrays for run_seed={run_seed}: "
                    f"{joined}"
                )
            indices = {
                name: _read_index_vector(archive, f"{prefix}{name}") for name in _ARRAY_NAMES
            }
    except SamplingValidationError:
        raise
    except Exception as exc:
        raise SamplingValidationError(f"cannot load ordered partition artifact: {source}") from exc

    _validate_expected_sizes(indices, spec)
    labeled = indices["train_labeled"]
    if labeled.size and (labeled.min() < 0 or labeled.max() >= int(y_train.shape[0])):
        raise SamplingValidationError(
            "ordered partition artifact train_labeled has out-of-range indices"
        )
    if spec.expected_per_class is not None:
        classes = np.unique(y_train)
        counts = np.asarray(
            [np.count_nonzero(y_train[labeled] == label) for label in classes],
            dtype=np.int64,
        )
        if not np.all(counts == int(spec.expected_per_class)):
            raise SamplingValidationError(
                "ordered partition artifact does not have the expected per-class labeled count"
            )
    return indices
