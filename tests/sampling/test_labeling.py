from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from modssc.sampling.errors import SamplingValidationError
from modssc.sampling.labeling import _class_counts, select_labeled
from modssc.sampling.plan import LabelingSpec


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
