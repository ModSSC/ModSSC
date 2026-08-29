from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from modssc.sampling.errors import SamplingValidationError
from modssc.sampling.labeling import _class_counts, _select_from_artifact, select_labeled
from modssc.sampling.plan import FixedIndicesArtifactSpec, LabelingSpec

_SOURCE_SHA256 = "f" * 64


def _artifact_spec(path, **overrides) -> LabelingSpec:
    values = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": _SOURCE_SHA256,
        "key": "perm",
        "index_stride": 2,
        "index_offset": 1,
        "expected_size": 2,
        "expected_per_class": 1,
    }
    values.update(overrides)
    return LabelingSpec(
        fixed_indices_artifact=FixedIndicesArtifactSpec(**values),
    )


def _write_permutations(path, rows) -> None:
    normalized = [np.asarray(row, dtype=np.int64) for row in rows]
    offsets = np.zeros(len(normalized) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([row.size for row in normalized], dtype=np.int64)
    values = np.concatenate(normalized) if normalized else np.empty(0, dtype=np.int64)
    metadata = json.dumps(
        {
            "format": "ragged_int64_v1",
            "row_count": len(normalized),
            "schema_version": 1,
            "source_key": "perm",
            "source_sha256": _SOURCE_SHA256,
        },
        sort_keys=True,
    ).encode()
    np.savez(
        path,
        metadata_json=np.frombuffer(metadata, dtype=np.uint8),
        offsets=offsets,
        values=values,
    )


def test_fixed_indices_artifact_selects_exact_seed_row(tmp_path) -> None:
    path = tmp_path / "permutations.npz"
    _write_permutations(path, [[0, 2], [1, 3], [2, 4], [0, 5]])
    selected = select_labeled(
        train_idx=np.arange(6, dtype=np.int64),
        y=np.array([0, 0, 0, 1, 1, 1]),
        spec=_artifact_spec(path),
        rng=np.random.default_rng(99),
        run_seed=1,
    )
    np.testing.assert_array_equal(selected, [0, 5])


@pytest.mark.parametrize("run_seed", [None, True, -1])
def test_fixed_indices_artifact_requires_a_non_negative_integer_seed(tmp_path, run_seed) -> None:
    path = tmp_path / "permutations.npz"
    _write_permutations(path, [[0, 1], [0, 1]])
    with pytest.raises(SamplingValidationError, match="non-negative run_seed"):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=_artifact_spec(path),
            rng=np.random.default_rng(0),
            run_seed=run_seed,
        )


def test_fixed_indices_artifact_authenticates_before_loading(tmp_path) -> None:
    path = tmp_path / "permutations.npz"
    _write_permutations(path, [[0, 1], [0, 1]])
    spec = _artifact_spec(path)
    bad = LabelingSpec(
        fixed_indices_artifact=FixedIndicesArtifactSpec(
            **{**spec.fixed_indices_artifact.__dict__, "sha256": "0" * 64}
        )
    )
    with pytest.raises(SamplingValidationError, match="SHA-256 differs"):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=bad,
            rng=np.random.default_rng(0),
            run_seed=0,
        )

    missing = LabelingSpec(
        fixed_indices_artifact=FixedIndicesArtifactSpec(
            path=str(tmp_path / "missing.npz"),
            sha256="0" * 64,
            source_sha256=_SOURCE_SHA256,
        )
    )
    with pytest.raises(SamplingValidationError, match="artifact is missing"):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=missing,
            rng=np.random.default_rng(0),
            run_seed=0,
        )


def test_fixed_indices_artifact_rejects_malformed_archives(tmp_path) -> None:
    wrong_key = tmp_path / "wrong-key.npz"
    _write_permutations(wrong_key, [[0, 1], [0, 1]])
    with np.load(wrong_key, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads(payload["metadata_json"].tobytes())
    metadata["source_key"] = "other"
    payload["metadata_json"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    np.savez(wrong_key, **payload)
    with pytest.raises(SamplingValidationError, match="source key differs"):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=_artifact_spec(wrong_key),
            rng=np.random.default_rng(0),
            run_seed=0,
        )


def _rewrite_safe_artifact(
    path, *, arrays=None, metadata_changes=None, metadata_value=None
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    if arrays:
        payload.update(arrays)
    if metadata_value is not None:
        encoded = json.dumps(metadata_value).encode()
        payload["metadata_json"] = np.frombuffer(encoded, dtype=np.uint8)
    elif metadata_changes:
        metadata = json.loads(payload["metadata_json"].tobytes())
        metadata.update(metadata_changes)
        payload["metadata_json"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    np.savez(path, **payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"arrays": {"offsets": np.array([], dtype=np.int64)}}, "offsets are invalid"),
        (
            {"arrays": {"offsets": np.array([0, 3, 2, 4], dtype=np.int64)}},
            "not monotonic",
        ),
        ({"metadata_changes": {"row_count": True}}, "row count is invalid"),
        ({"metadata_changes": {"row_count": 3}}, "row count is invalid"),
        ({"arrays": {"values": np.array([[0, 1]], dtype=np.int64)}}, "one-dimensional"),
        ({"arrays": {"offsets": np.array([0.0, 2.0, 4.0])}}, "integer array"),
        ({"metadata_changes": {"schema_version": 2}}, "schema_version must be 1"),
        ({"metadata_changes": {"format": "unsafe"}}, "format must be"),
        ({"metadata_value": []}, "metadata must be a mapping"),
    ],
)
def test_fixed_indices_artifact_rejects_invalid_safe_payload(
    tmp_path,
    mutation,
    message,
) -> None:
    path = tmp_path / "invalid-safe.npz"
    _write_permutations(path, [[0, 1], [0, 1]])
    _rewrite_safe_artifact(path, **mutation)

    with pytest.raises(SamplingValidationError, match=message):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=_artifact_spec(path),
            rng=np.random.default_rng(0),
            run_seed=0,
        )


def test_fixed_indices_artifact_rejects_identity_drift_and_missing_row(tmp_path) -> None:
    path = tmp_path / "identity.npz"
    _write_permutations(path, [[0, 1], [0, 1]])

    with pytest.raises(SamplingValidationError, match="source SHA-256 differs"):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=_artifact_spec(path, source_sha256="0" * 64),
            rng=np.random.default_rng(0),
            run_seed=0,
        )
    with pytest.raises(SamplingValidationError, match="has no row"):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=_artifact_spec(path),
            rng=np.random.default_rng(0),
            run_seed=5,
        )


def test_fixed_indices_artifact_rejects_missing_array_and_invalid_metadata(tmp_path) -> None:
    path = tmp_path / "missing-array.npz"
    _write_permutations(path, [[0, 1], [0, 1]])
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files if key != "values"}
    np.savez(path, **payload)
    with pytest.raises(SamplingValidationError, match="has no 'values' array"):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=_artifact_spec(path),
            rng=np.random.default_rng(0),
            run_seed=0,
        )

    path = tmp_path / "invalid-metadata.npz"
    _write_permutations(path, [[0, 1], [0, 1]])
    _rewrite_safe_artifact(
        path,
        arrays={"metadata_json": np.array(["not-uint8"])},
    )
    with pytest.raises(SamplingValidationError, match="uint8 vector"):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=_artifact_spec(path),
            rng=np.random.default_rng(0),
            run_seed=0,
        )

    path = tmp_path / "invalid-json.npz"
    _write_permutations(path, [[0, 1], [0, 1]])
    _rewrite_safe_artifact(
        path,
        arrays={"metadata_json": np.frombuffer(b"{", dtype=np.uint8)},
    )
    with pytest.raises(SamplingValidationError, match="metadata is invalid"):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=_artifact_spec(path),
            rng=np.random.default_rng(0),
            run_seed=0,
        )

    scalar = tmp_path / "scalar.npz"
    _write_permutations(scalar, [[0, 1], [0, 1]])
    with np.load(scalar, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["offsets"] = np.array(1)
    np.savez(scalar, **payload)
    with pytest.raises(SamplingValidationError, match="one-dimensional integer array"):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=_artifact_spec(scalar),
            rng=np.random.default_rng(0),
            run_seed=0,
        )

    legacy = tmp_path / "legacy-object.npz"
    values = np.empty(2, dtype=object)
    values[:] = [np.array([0, 1]), np.array([0, 1])]
    np.savez(legacy, perm=values)
    with pytest.raises(SamplingValidationError, match="metadata_json"):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=_artifact_spec(legacy),
            rng=np.random.default_rng(0),
            run_seed=0,
        )

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not an archive")
    with pytest.raises(SamplingValidationError, match="cannot load"):
        select_labeled(
            train_idx=np.arange(2),
            y=np.array([0, 1]),
            spec=_artifact_spec(corrupt),
            rng=np.random.default_rng(0),
            run_seed=0,
        )


def test_fixed_indices_artifact_validates_row_contract(tmp_path) -> None:
    path = tmp_path / "permutations.npz"
    _write_permutations(path, [[0, 1], [0, 1]])
    common = {
        "train_idx": np.arange(4),
        "y": np.array([0, 0, 1, 1]),
        "rng": np.random.default_rng(0),
        "run_seed": 0,
    }
    with pytest.raises(SamplingValidationError, match="wrong size"):
        select_labeled(
            **common,
            spec=_artifact_spec(path, expected_size=3),
        )
    with pytest.raises(SamplingValidationError, match="expected per-class count"):
        select_labeled(
            **common,
            spec=_artifact_spec(path, expected_per_class=1),
        )

    duplicates = tmp_path / "duplicates.npz"
    _write_permutations(duplicates, [[0, 0], [0, 0]])
    with pytest.raises(SamplingValidationError, match="duplicates"):
        select_labeled(
            **common,
            spec=_artifact_spec(duplicates, expected_per_class=None),
        )

    outside = tmp_path / "outside.npz"
    _write_permutations(outside, [[0, 3], [0, 3]])
    with pytest.raises(SamplingValidationError, match="subset of train"):
        select_labeled(
            train_idx=np.array([0, 1, 2]),
            y=np.array([0, 0, 1, 1]),
            spec=_artifact_spec(outside, expected_per_class=None),
            rng=np.random.default_rng(0),
            run_seed=0,
        )


def test_fixed_indices_artifact_internal_guard() -> None:
    with pytest.raises(AssertionError, match="required"):
        _select_from_artifact(
            spec=LabelingSpec(),
            run_seed=0,
            train_idx=np.arange(1),
            y=np.zeros(1, dtype=np.int64),
        )


def test_label_fraction_per_class_min() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    train_idx = np.arange(6, dtype=np.int64)
    rng = np.random.default_rng(0)
    labeled = select_labeled(
        train_idx=train_idx,
        y=y,
        spec=LabelingSpec(mode="fraction", value=0.1, per_class=True, min_per_class=1),
        rng=rng,
    )

    assert labeled.size >= 2
    assert set(np.unique(y[labeled]).tolist()) == {0, 1}


def test_label_count_balanced() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    train_idx = np.arange(6, dtype=np.int64)
    rng = np.random.default_rng(0)
    labeled = select_labeled(
        train_idx=train_idx,
        y=y,
        spec=LabelingSpec(mode="count", value=4, strategy="balanced"),
        rng=rng,
    )
    assert labeled.size == 4


@pytest.mark.parametrize("selection_order", ["choice", "permutation"])
def test_class_counts_selects_exact_seed_deterministic_quota(selection_order: str) -> None:
    train_idx = np.arange(2, 42, dtype=np.int64)
    y = np.array([9, 9] + [0] * 25 + [1] * 15, dtype=np.int64)
    spec = LabelingSpec(
        mode="count",
        value=12,
        strategy="random",
        class_counts={"0": 9, "1": 3},
        selection_order=selection_order,  # type: ignore[arg-type]
    )

    first = select_labeled(
        train_idx=train_idx,
        y=y,
        spec=spec,
        rng=np.random.default_rng(17),
    )
    replay = select_labeled(
        train_idx=train_idx,
        y=y,
        spec=spec,
        rng=np.random.default_rng(17),
    )
    other = select_labeled(
        train_idx=train_idx,
        y=y,
        spec=spec,
        rng=np.random.default_rng(18),
    )

    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, other)
    assert first.size == 12
    assert np.count_nonzero(y[first] == 0) == 9
    assert np.count_nonzero(y[first] == 1) == 3
    assert np.setdiff1d(first, train_idx).size == 0


def test_class_counts_allows_an_explicit_zero_quota() -> None:
    selected = select_labeled(
        train_idx=np.arange(6),
        y=np.array([0, 0, 0, 1, 1, 1]),
        spec=LabelingSpec(mode="count", value=2, class_counts={"0": 2, "1": 0}),
        rng=np.random.default_rng(5),
    )

    assert selected.size == 2
    assert np.all(np.array([0, 0, 0, 1, 1, 1])[selected] == 0)


def test_class_counts_rejects_empty_train_missing_labels_and_unavailable_quota() -> None:
    with pytest.raises(SamplingValidationError, match="empty train"):
        select_labeled(
            train_idx=np.array([], dtype=np.int64),
            y=np.array([], dtype=np.int64),
            spec=LabelingSpec(mode="count", value=1, class_counts={"0": 1}),
            rng=np.random.default_rng(0),
        )

    common = {
        "train_idx": np.arange(4),
        "y": np.array([0, 0, 1, 1]),
        "rng": np.random.default_rng(0),
    }
    with pytest.raises(SamplingValidationError, match="exactly match"):
        select_labeled(
            **common,
            spec=LabelingSpec(mode="count", value=2, class_counts={"0": 2}),
        )
    with pytest.raises(SamplingValidationError, match="only 2 are available"):
        select_labeled(
            **common,
            spec=LabelingSpec(mode="count", value=4, class_counts={"0": 3, "1": 1}),
        )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            LabelingSpec(mode="fraction", value=1.0, class_counts={"0": 1}),
            "mode='count'",
        ),
        (
            LabelingSpec(mode="count", value=2, class_counts={"0": True, "1": 1}),
            "non-negative integers",
        ),
        (
            LabelingSpec(mode="count", value=1, class_counts={"0": -1, "1": 2}),
            "non-negative integers",
        ),
        (
            LabelingSpec(mode="count", value=1, class_counts={"0": 0, "1": 0}),
            "at least one sample",
        ),
        (
            LabelingSpec(mode="count", value=3, class_counts={"0": 1, "1": 1}),
            "sum to labeling.value",
        ),
        (
            LabelingSpec(
                mode="count",
                value=2,
                class_counts={"0": 1, "1": 1},
                selection_order="invalid",  # type: ignore[arg-type]
            ),
            "Unknown labeling selection_order",
        ),
        (
            LabelingSpec(mode="count", value=2, class_counts={0: 1, "0": 1}),
            "duplicate labels",
        ),
    ],
)
def test_class_counts_direct_specs_are_validated_fail_closed(
    spec: LabelingSpec,
    message: str,
) -> None:
    with pytest.raises(SamplingValidationError, match=message):
        select_labeled(
            train_idx=np.arange(4),
            y=np.array([0, 0, 1, 1]),
            spec=spec,
            rng=np.random.default_rng(0),
        )


def test_permutation_selection_is_nested_across_label_budgets() -> None:
    y = np.repeat(np.arange(3), 10)
    train_idx = np.arange(y.size, dtype=np.int64)
    small = select_labeled(
        train_idx=train_idx,
        y=y,
        spec=LabelingSpec(
            mode="per_class",
            value=2,
            strategy="balanced",
            selection_order="permutation",
        ),
        rng=np.random.default_rng(17),
    )
    large = select_labeled(
        train_idx=train_idx,
        y=y,
        spec=LabelingSpec(
            mode="per_class",
            value=5,
            strategy="balanced",
            selection_order="permutation",
        ),
        rng=np.random.default_rng(17),
    )

    assert set(small).issubset(set(large))


def test_label_count_random_is_uniform_over_full_train_partition() -> None:
    train_idx = np.arange(100, dtype=np.int64)
    y_imbalanced = np.array([0] * 99 + [1])
    y_relabelled = np.arange(100, dtype=np.int64)
    spec = LabelingSpec(mode="count", value=12, strategy="random", min_per_class=0)

    first = select_labeled(
        train_idx=train_idx,
        y=y_imbalanced,
        spec=spec,
        rng=np.random.default_rng(7),
    )
    second = select_labeled(
        train_idx=train_idx,
        y=y_relabelled,
        spec=spec,
        rng=np.random.default_rng(7),
    )

    assert first.size == 12
    assert np.array_equal(first, second)


def test_label_count_random_samples_once_without_class_allocation() -> None:
    train_idx = np.arange(10, dtype=np.int64)
    rng = MagicMock()
    rng.choice.return_value = np.array([0, 1, 2, 3], dtype=np.int64)

    labeled = select_labeled(
        train_idx=train_idx,
        y=np.array([0] * 9 + [1]),
        spec=LabelingSpec(mode="count", value=4, strategy="random", min_per_class=0),
        rng=rng,
    )

    assert labeled.tolist() == [0, 1, 2, 3]
    rng.choice.assert_called_once_with(train_idx, size=4, replace=False)


def test_label_count_random_permutation_uses_prefix() -> None:
    train_idx = np.arange(10, dtype=np.int64)
    spec = LabelingSpec(
        mode="count",
        value=4,
        strategy="random",
        min_per_class=0,
        selection_order="permutation",
    )
    expected = np.sort(np.random.default_rng(9).permutation(train_idx)[:4])
    actual = select_labeled(
        train_idx=train_idx,
        y=np.zeros(10, dtype=np.int64),
        spec=spec,
        rng=np.random.default_rng(9),
    )
    np.testing.assert_array_equal(actual, expected)


def test_fixed_indices_validation() -> None:
    y = np.array([0, 1, 0])
    train_idx = np.array([0, 1, 2])
    rng = np.random.default_rng(0)

    ok = select_labeled(train_idx=train_idx, y=y, spec=LabelingSpec(fixed_indices=[0, 2]), rng=rng)
    assert ok.tolist() == [0, 2]

    with pytest.raises(SamplingValidationError):
        select_labeled(train_idx=train_idx, y=y, spec=LabelingSpec(fixed_indices=[99]), rng=rng)


def test_labeling_empty_train():
    """Test select_labeled with empty train_idx."""
    spec = LabelingSpec()
    rng = np.random.default_rng(0)
    res = select_labeled(train_idx=np.array([]), y=np.array([]), spec=spec, rng=rng)
    assert res.size == 0
    assert res.dtype == np.int64


def test_labeling_fixed_duplicates():
    """Test select_labeled with duplicate fixed_indices."""
    spec = LabelingSpec(fixed_indices=np.array([0, 0]))
    rng = np.random.default_rng(0)
    train_idx = np.array([0, 1])
    y = np.array([0, 0])
    with pytest.raises(SamplingValidationError, match="fixed_indices contains duplicates"):
        select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)


def test_labeling_fill_loop_all_full():
    """Test fill loop when all classes are already full (total < target)."""

    y = np.zeros(1, dtype=int)
    train_idx = np.arange(1)

    spec = LabelingSpec(mode="count", value=10)
    rng = np.random.default_rng(42)

    res = select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)

    assert res.size == 1


def test_labeling_fill_loop_spread_deficit():
    """Test fill loop where deficit is spread across multiple classes.

    Scenario:
    C0..C9: 1 sample each.
    C10..C11: 10 samples each.
    Mode per_class value=5. Target=60.
    Init: 5 each.
    Cap: C0..C9 -> 1 (Lost 40). C10..C11 -> 5.
    Total=20. Deficit=40.
    Gaps: C10=5, C11=5.
    We fill C10 (hit else), fill C11 (hit else).
    """
    y = np.concatenate(
        [
            np.arange(10),
            np.full(10, 10),
            np.full(10, 11),
        ]
    )
    train_idx = np.arange(30)

    spec = LabelingSpec(mode="per_class", value=5)
    rng = np.random.default_rng(42)

    res = select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)

    assert res.size == 30


def test_labeling_invalid_fraction():
    """Test select_labeled with invalid fraction."""
    spec = LabelingSpec(mode="fraction", value=1.5)
    rng = np.random.default_rng(0)
    train_idx = np.array([0, 1])
    y = np.array([0, 0])
    with pytest.raises(ValueError, match="label fraction must be in"):
        select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)


def test_labeling_unknown_mode():
    """Test select_labeled with unknown mode."""
    spec = LabelingSpec(mode="invalid")  # type: ignore
    rng = np.random.default_rng(0)
    train_idx = np.array([0, 1])
    y = np.array([0, 0])
    with pytest.raises(ValueError, match="Unknown labeling mode"):
        select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)


def test_labeling_unknown_strategy():
    spec = LabelingSpec(strategy="invalid")  # type: ignore
    rng = np.random.default_rng(0)
    train_idx = np.array([0, 1])
    y = np.array([0, 0])

    with pytest.raises(ValueError, match="Unknown labeling strategy"):
        select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)


def test_labeling_proportional_remainder():
    """Test proportional allocation with remainder."""

    spec = LabelingSpec(mode="count", value=4)
    rng = np.random.default_rng(0)
    train_idx = np.arange(30)
    y = np.concatenate([np.zeros(10), np.ones(10), np.full(10, 2)])
    res = select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)
    assert res.size == 4

    y_res = y[res]
    counts = np.bincount(y_res.astype(int))
    assert np.sum(counts == 2) == 1
    assert np.sum(counts == 1) == 2


def test_labeling_adjust_down_min_per_class():
    """Test reducing allocation when min_per_class pushes total > target."""

    spec = LabelingSpec(mode="count", value=10, min_per_class=5)
    rng = np.random.default_rng(0)

    train_idx = np.arange(110)
    y = np.concatenate([np.zeros(100), np.ones(10)])

    res = select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)

    y_res = y[res]
    c0_count = np.sum(y_res == 0)
    c1_count = np.sum(y_res == 1)

    assert c1_count == 5

    assert c0_count == 5
    assert res.size == 10


def test_labeling_adjust_up_cap():
    """Test increasing allocation when cap reduces total < target."""

    spec = LabelingSpec(mode="per_class", value=5)
    rng = np.random.default_rng(0)

    train_idx = np.arange(102)
    y = np.concatenate([np.zeros(2), np.ones(100)])

    res = select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)

    y_res = y[res]
    c0_count = np.sum(y_res == 0)
    c1_count = np.sum(y_res == 1)

    assert c0_count == 2
    assert c1_count == 8
    assert res.size == 10


def test_labeling_zero_selection():
    """Test class with 0 selection (n_sel <= 0)."""

    spec = LabelingSpec(mode="count", value=0, min_per_class=0)
    rng = np.random.default_rng(0)
    train_idx = np.arange(10)
    y = np.zeros(10)
    res = select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)
    assert res.size == 0


def test_class_counts_empty():
    classes, counts = _class_counts(np.array([], dtype=np.int64))
    assert classes.size == 0
    assert counts.size == 0


def test_class_counts_integer_bincount():
    classes, counts = _class_counts(np.array([0, 1, 1, 3], dtype=np.int64))
    assert classes.tolist() == [0, 1, 3]
    assert counts.tolist() == [1, 2, 1]


def test_class_counts_integer_out_of_range():
    classes, counts = _class_counts(np.array([-1, 2_000_001], dtype=np.int64))
    assert set(classes.tolist()) == {-1, 2_000_001}
    assert counts.tolist() == [1, 1]


def test_labeling_defensive_guards():
    """Test defensive guards at the end of select_labeled."""
    spec = LabelingSpec(mode="fraction", value=0.5)
    train_idx = np.arange(10)
    y = np.zeros(10)

    mock_rng = MagicMock()

    mock_rng.choice.return_value = np.arange(20)

    with pytest.raises(SamplingValidationError, match="labeled size cannot exceed train size"):
        select_labeled(train_idx=train_idx, y=y, spec=spec, rng=mock_rng)

    mock_rng.choice.return_value = np.array([0, 0])

    with pytest.raises(SamplingValidationError, match="labeled contains duplicates"):
        select_labeled(train_idx=train_idx, y=y, spec=spec, rng=mock_rng)


def test_labeling_fill_loop_exhaustion():
    """Test hitting the else branch in the fill loop (total < target).

    Scenario:
    C0: 1 sample.
    C1: 5 samples.
    Target = 6.
    Mode per_class value=4 -> Target=8.
    Init: [4, 4].
    Cap: [1, 4]. Total=5. Deficit=3.
    Gaps: C0:0, C1:1.
    Order: [C1, C0].
    We fill C1 (add 1). C1 becomes 5. Total=6.
    Then we hit else for C1 (full).
    Then we hit else for C0 (full).
    Loop ends.
    """
    y = np.array([0] * 1 + [1] * 5)
    train_idx = np.arange(6)

    spec = LabelingSpec(mode="per_class", value=4)
    rng = np.random.default_rng(42)

    res = select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)

    assert res.size == 6
    assert np.unique(y[res]).size == 2


def test_labeling_min_per_class_skip():
    """Test skipping min_per_class enforcement when counts < min_per_class.

    Scenario:
    C0: 2 samples.
    min_per_class = 5.
    """
    y = np.zeros(2, dtype=int)
    train_idx = np.arange(2)

    spec = LabelingSpec(mode="count", value=1, min_per_class=5)
    rng = np.random.default_rng(42)

    res = select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)

    assert res.size == 1


def test_labeling_balanced_strategy():
    """Test balanced strategy allocation."""

    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    train_idx = np.arange(30)

    spec = LabelingSpec(mode="count", value=4, strategy="balanced")
    rng = np.random.default_rng(42)

    res = select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)

    assert res.size == 4

    y_sel = y[res]
    counts = np.bincount(y_sel)

    assert np.array_equal(np.sort(counts[counts > 0]), [1, 1, 2])


def test_labeling_fixed_subset_validation():
    """Test validation that fixed_indices is a subset of train_idx."""
    spec = LabelingSpec(fixed_indices=[0, 100])
    rng = np.random.default_rng(0)
    train_idx = np.array([0, 1])
    y = np.array([0, 0])

    with pytest.raises(SamplingValidationError, match="fixed_indices must be a subset"):
        select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)


def test_labeling_enforce_min_per_class():
    """Test enforcement of min_per_class when counts allow it.

    Scenario:
    C0: 10 samples.
    min_per_class = 5.
    Target = 2.
    Proportional allocation gives < 5.
    Should be boosted to 5.
    """
    y = np.zeros(10, dtype=int)
    train_idx = np.arange(10)

    spec = LabelingSpec(mode="count", value=2, min_per_class=5)
    rng = np.random.default_rng(42)

    res = select_labeled(train_idx=train_idx, y=y, spec=spec, rng=rng)

    assert res.size == 5
