from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from modssc.sampling.api import sample
from modssc.sampling.errors import SamplingValidationError
from modssc.sampling.partition_artifact import load_ordered_partition
from modssc.sampling.plan import (
    ImbalanceSpec,
    OrderedPartitionArtifactSpec,
    PartitionSpec,
    SamplingPlan,
)
from tests.sampling._stubs import make_toy_dataset


def _valid_metadata() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seeds": [7],
        "train_source_size": 4,
        "test_source_size": 20,
        "unlabeled_pool": "includes_labeled",
        "test_ref": "test",
    }


def _valid_arrays() -> dict[str, np.ndarray]:
    return {
        "seed_7__train": np.array([3, 0, 2, 1]),
        "seed_7__val": np.array([], dtype=np.int64),
        "seed_7__test": np.array([1, 0]),
        "seed_7__train_labeled": np.array([0, 2]),
        "seed_7__train_unlabeled": np.array([3, 0, 2, 1]),
    }


def _write_artifact(
    path: Path,
    *,
    metadata: Any = None,
    arrays: dict[str, np.ndarray] | None = None,
    include_metadata: bool = True,
) -> None:
    payload = dict(_valid_arrays() if arrays is None else arrays)
    if include_metadata:
        value = _valid_metadata() if metadata is None else metadata
        encoded = json.dumps(value, sort_keys=True).encode("utf-8")
        payload["metadata_json"] = np.frombuffer(encoded, dtype=np.uint8)
    np.savez_compressed(path, **payload)


def _spec(path: Path, **changes: Any) -> OrderedPartitionArtifactSpec:
    values: dict[str, Any] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "unlabeled_pool": "includes_labeled",
        "test_ref": "test",
        "expected_train_size": 4,
        "expected_val_size": 0,
        "expected_test_size": 2,
        "expected_labeled_size": 2,
        "expected_unlabeled_size": 4,
        "expected_per_class": 1,
    }
    values.update(changes)
    return OrderedPartitionArtifactSpec(**values)


def test_ordered_partition_loader_preserves_order(tmp_path) -> None:
    path = tmp_path / "partition.npz"
    _write_artifact(path)

    indices = load_ordered_partition(
        spec=_spec(path),
        run_seed=7,
        y_train=np.array([0, 0, 1, 1]),
        n_test=20,
    )

    assert indices["train"].tolist() == [3, 0, 2, 1]
    assert indices["train_labeled"].tolist() == [0, 2]
    assert indices["train_unlabeled"].tolist() == [3, 0, 2, 1]
    assert indices["test"].tolist() == [1, 0]


def test_ordered_partition_loader_supports_train_referenced_test(tmp_path) -> None:
    path = tmp_path / "train-ref.npz"
    metadata = {**_valid_metadata(), "test_ref": "train"}
    _write_artifact(path, metadata=metadata)

    indices = load_ordered_partition(
        spec=_spec(path, test_ref="train"),
        run_seed=7,
        y_train=np.array([0, 0, 1, 1]),
        n_test=None,
    )

    assert indices["test"].tolist() == [1, 0]


def test_ordered_partition_loader_wraps_invalid_archive(tmp_path) -> None:
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"not-an-npz")
    with pytest.raises(SamplingValidationError, match="cannot load"):
        load_ordered_partition(
            spec=OrderedPartitionArtifactSpec(
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
            run_seed=0,
            y_train=np.array([0]),
            n_test=None,
        )


@pytest.mark.parametrize("run_seed", [-1, True])
def test_ordered_partition_loader_rejects_invalid_seed(tmp_path, run_seed) -> None:
    path = tmp_path / "partition.npz"
    _write_artifact(path)
    with pytest.raises(SamplingValidationError, match="non-negative run_seed"):
        load_ordered_partition(
            spec=_spec(path),
            run_seed=run_seed,
            y_train=np.array([0, 0, 1, 1]),
            n_test=20,
        )


def test_ordered_partition_loader_authenticates_file(tmp_path) -> None:
    missing = tmp_path / "missing.npz"
    with pytest.raises(SamplingValidationError, match="is missing"):
        load_ordered_partition(
            spec=OrderedPartitionArtifactSpec(path=str(missing), sha256="a" * 64),
            run_seed=0,
            y_train=np.array([0]),
            n_test=None,
        )

    path = tmp_path / "partition.npz"
    _write_artifact(path)
    with pytest.raises(SamplingValidationError, match="SHA-256 differs"):
        load_ordered_partition(
            spec=_spec(path, sha256="a" * 64),
            run_seed=7,
            y_train=np.array([0, 0, 1, 1]),
            n_test=20,
        )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ([], "must be a mapping"),
        ({"schema_version": 2}, "schema_version must be 1"),
        ({**_valid_metadata(), "unlabeled_pool": "complement"}, "unlabeled_pool differs"),
        ({**_valid_metadata(), "test_ref": "train"}, "test_ref differs"),
        ({**_valid_metadata(), "train_source_size": 5}, "train_source_size differs"),
        (
            {**_valid_metadata(), "dataset_fingerprint": "other"},
            "dataset_fingerprint differs",
        ),
        ({**_valid_metadata(), "test_source_size": 3}, "test_source_size differs"),
        ({**_valid_metadata(), "seeds": "7"}, "has no run_seed"),
        ({**_valid_metadata(), "seeds": [8]}, "has no run_seed"),
    ],
)
def test_ordered_partition_loader_rejects_invalid_metadata(
    tmp_path,
    metadata,
    message,
) -> None:
    path = tmp_path / "partition.npz"
    _write_artifact(path, metadata=metadata)
    with pytest.raises(SamplingValidationError, match=message):
        load_ordered_partition(
            spec=_spec(path),
            run_seed=7,
            y_train=np.array([0, 0, 1, 1]),
            n_test=20,
        )


def test_ordered_partition_loader_requires_official_test(tmp_path) -> None:
    path = tmp_path / "partition.npz"
    _write_artifact(path)
    with pytest.raises(SamplingValidationError, match="requires an official test"):
        load_ordered_partition(
            spec=_spec(path),
            run_seed=7,
            y_train=np.array([0, 0, 1, 1]),
            n_test=None,
        )


def test_ordered_partition_loader_rejects_missing_or_invalid_metadata(tmp_path) -> None:
    missing = tmp_path / "missing-meta.npz"
    _write_artifact(missing, include_metadata=False)
    with pytest.raises(SamplingValidationError, match="has no 'metadata_json'"):
        load_ordered_partition(
            spec=_spec(missing),
            run_seed=7,
            y_train=np.array([0, 0, 1, 1]),
            n_test=20,
        )

    invalid = tmp_path / "invalid-meta.npz"
    payload = _valid_arrays()
    payload["metadata_json"] = np.frombuffer(b"{", dtype=np.uint8)
    np.savez_compressed(invalid, **payload)
    with pytest.raises(SamplingValidationError, match="metadata is invalid"):
        load_ordered_partition(
            spec=_spec(invalid),
            run_seed=7,
            y_train=np.array([0, 0, 1, 1]),
            n_test=20,
        )


def test_ordered_partition_loader_rejects_missing_seed_array(tmp_path) -> None:
    path = tmp_path / "partition.npz"
    arrays = _valid_arrays()
    del arrays["seed_7__val"]
    _write_artifact(path, arrays=arrays)
    with pytest.raises(SamplingValidationError, match="missing arrays.*val"):
        load_ordered_partition(
            spec=_spec(path),
            run_seed=7,
            y_train=np.array([0, 0, 1, 1]),
            n_test=20,
        )


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (np.asarray([0.0, 2.0]), "one-dimensional integer array"),
        (np.asarray([[0, 2]]), "one-dimensional integer array"),
        (
            np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64),
            "outside int64 range",
        ),
    ],
)
def test_ordered_partition_loader_validates_index_array_before_conversion(
    tmp_path,
    invalid,
    message,
) -> None:
    path = tmp_path / "invalid-vector.npz"
    arrays = _valid_arrays()
    arrays["seed_7__train_labeled"] = invalid
    _write_artifact(path, arrays=arrays)

    with pytest.raises(SamplingValidationError, match=message):
        load_ordered_partition(
            spec=_spec(path),
            run_seed=7,
            y_train=np.array([0, 0, 1, 1]),
            n_test=20,
        )


@pytest.mark.parametrize(
    "metadata_json",
    [
        np.asarray([1.0, 2.0]),
        np.asarray([[1, 2]], dtype=np.uint8),
    ],
)
def test_ordered_partition_loader_validates_metadata_bytes(tmp_path, metadata_json) -> None:
    path = tmp_path / "invalid-metadata-vector.npz"
    payload = _valid_arrays()
    payload["metadata_json"] = metadata_json
    np.savez_compressed(path, **payload)

    with pytest.raises(SamplingValidationError, match="metadata_json must be a uint8 vector"):
        load_ordered_partition(
            spec=_spec(path),
            run_seed=7,
            y_train=np.array([0, 0, 1, 1]),
            n_test=20,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_train_size", 3, "train has the wrong size"),
        ("expected_val_size", 1, "val has the wrong size"),
        ("expected_test_size", 1, "test has the wrong size"),
        ("expected_labeled_size", 1, "train_labeled has the wrong size"),
        ("expected_unlabeled_size", 3, "train_unlabeled has the wrong size"),
    ],
)
def test_ordered_partition_loader_checks_expected_sizes(
    tmp_path,
    field,
    value,
    message,
) -> None:
    path = tmp_path / "partition.npz"
    _write_artifact(path)
    with pytest.raises(SamplingValidationError, match=message):
        load_ordered_partition(
            spec=_spec(path, **{field: value}),
            run_seed=7,
            y_train=np.array([0, 0, 1, 1]),
            n_test=20,
        )


def test_ordered_partition_loader_checks_labeled_indices_and_balance(tmp_path) -> None:
    invalid = tmp_path / "invalid-index.npz"
    arrays = _valid_arrays()
    arrays["seed_7__train_labeled"] = np.array([0, 5])
    _write_artifact(invalid, arrays=arrays)
    with pytest.raises(SamplingValidationError, match="out-of-range"):
        load_ordered_partition(
            spec=_spec(invalid),
            run_seed=7,
            y_train=np.array([0, 0, 1, 1]),
            n_test=20,
        )

    unbalanced = tmp_path / "unbalanced.npz"
    arrays = _valid_arrays()
    arrays["seed_7__train_labeled"] = np.array([0, 1])
    _write_artifact(unbalanced, arrays=arrays)
    with pytest.raises(SamplingValidationError, match="expected per-class"):
        load_ordered_partition(
            spec=_spec(unbalanced),
            run_seed=7,
            y_train=np.array([0, 0, 1, 1]),
            n_test=20,
        )


def test_sample_replays_ordered_inclusive_partition(tmp_path) -> None:
    dataset = make_toy_dataset(n=4, n_classes=2, with_test=True)
    path = tmp_path / "partition.npz"
    _write_artifact(path)
    plan = SamplingPlan(
        partition=PartitionSpec(ordered_indices_artifact=_spec(path, expected_per_class=None)),
    )

    result, _ = sample(
        dataset,
        plan=plan,
        seed=7,
        dataset_fingerprint=dataset.meta["dataset_fingerprint"],
        save=False,
    )

    assert result.train_idx.tolist() == [3, 0, 2, 1]
    assert result.unlabeled_idx.tolist() == [3, 0, 2, 1]
    assert result.stats["policy"]["ordered_indices"] is True
    assert result.stats["policy"]["unlabeled_pool"] == "includes_labeled"


def test_sample_rejects_imbalance_with_ordered_partition(tmp_path) -> None:
    dataset = make_toy_dataset(n=4, n_classes=2, with_test=True)
    path = tmp_path / "partition.npz"
    _write_artifact(path)
    plan = SamplingPlan(
        partition=PartitionSpec(ordered_indices_artifact=_spec(path, expected_per_class=None)),
        imbalance=ImbalanceSpec(kind="subsample_max_per_class", max_per_class=1),
    )

    with pytest.raises(ValueError, match="imbalance must be 'none'"):
        sample(
            dataset,
            plan=plan,
            seed=7,
            dataset_fingerprint=dataset.meta["dataset_fingerprint"],
            save=False,
        )
