from __future__ import annotations

import numpy as np
import pytest

from modssc.sampling.imbalance import apply_imbalance
from modssc.sampling.plan import ImbalanceSpec


def test_none() -> None:
    y = np.array([0, 0, 1, 1])
    idx = np.arange(4, dtype=np.int64)
    out = apply_imbalance(
        idx=idx, y=y, spec=ImbalanceSpec(kind="none"), rng=np.random.default_rng(0)
    )
    assert out.tolist() == [0, 1, 2, 3]


def test_subsample_max_per_class() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    idx = np.arange(6, dtype=np.int64)
    out = apply_imbalance(
        idx=idx,
        y=y,
        spec=ImbalanceSpec(kind="subsample_max_per_class", max_per_class=2),
        rng=np.random.default_rng(0),
    )

    assert out.size == 4


def test_long_tail() -> None:
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    idx = np.arange(30, dtype=np.int64)
    out = apply_imbalance(
        idx=idx,
        y=y,
        spec=ImbalanceSpec(kind="long_tail", alpha=0.5, min_per_class=1),
        rng=np.random.default_rng(0),
    )
    assert out.size >= 3


def test_invalid_params() -> None:
    y = np.array([0, 1])
    idx = np.arange(2, dtype=np.int64)
    with pytest.raises(ValueError):
        apply_imbalance(
            idx=idx,
            y=y,
            spec=ImbalanceSpec(kind="subsample_max_per_class", max_per_class=0),
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ValueError):
        apply_imbalance(
            idx=idx,
            y=y,
            spec=ImbalanceSpec(kind="long_tail", alpha=1.0),
            rng=np.random.default_rng(0),
        )


def test_imbalance_none_or_empty():
    """Test early return for kind='none' or empty index."""
    spec = ImbalanceSpec(kind="none")
    idx = np.array([0, 1, 2])
    y = np.array([0, 0, 1])
    rng = np.random.default_rng(0)
    res = apply_imbalance(idx=idx, y=y, spec=spec, rng=rng)
    np.testing.assert_array_equal(res, idx)

    spec = ImbalanceSpec(kind="subsample_max_per_class", max_per_class=5)
    idx = np.array([], dtype=np.int64)
    res = apply_imbalance(idx=idx, y=y, spec=spec, rng=rng)
    assert res.size == 0


def test_imbalance_subsample_missing_max_per_class():
    """Test ValueError when max_per_class is missing for subsample_max_per_class."""
    spec = ImbalanceSpec(kind="subsample_max_per_class", max_per_class=None)
    idx = np.array([0, 1, 2])
    y = np.array([0, 0, 1])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="max_per_class is required"):
        apply_imbalance(idx=idx, y=y, spec=spec, rng=rng)


def test_imbalance_subsample_invalid_cap():
    """Test ValueError when max_per_class < 1."""
    spec = ImbalanceSpec(kind="subsample_max_per_class", max_per_class=0)
    idx = np.array([0, 1, 2])
    y = np.array([0, 0, 1])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="max_per_class must be >= 1"):
        apply_imbalance(idx=idx, y=y, spec=spec, rng=rng)


def test_imbalance_subsample_small_class():
    """Test subsample_max_per_class when class size <= cap."""

    spec = ImbalanceSpec(kind="subsample_max_per_class", max_per_class=5)
    idx = np.array([0, 1, 2])
    y = np.array([0, 0, 0])
    rng = np.random.default_rng(0)
    res = apply_imbalance(idx=idx, y=y, spec=spec, rng=rng)
    assert len(res) == 3
    np.testing.assert_array_equal(res, idx)


def test_imbalance_long_tail_missing_alpha():
    """Test ValueError when alpha is missing for long_tail."""
    spec = ImbalanceSpec(kind="long_tail", alpha=None)
    idx = np.array([0, 1, 2])
    y = np.array([0, 0, 1])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="alpha is required"):
        apply_imbalance(idx=idx, y=y, spec=spec, rng=rng)


def test_imbalance_long_tail_invalid_alpha():
    """Test ValueError when alpha is not in (0, 1)."""
    spec = ImbalanceSpec(kind="long_tail", alpha=1.5)
    idx = np.array([0, 1, 2])
    y = np.array([0, 0, 1])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="alpha must be in"):
        apply_imbalance(idx=idx, y=y, spec=spec, rng=rng)


def test_imbalance_long_tail_full_selection():
    """Test long_tail when desired count >= class size."""

    spec = ImbalanceSpec(kind="long_tail", alpha=0.9, min_per_class=10)
    idx = np.array([0, 1, 2])
    y = np.array([0, 0, 0])
    rng = np.random.default_rng(0)
    res = apply_imbalance(idx=idx, y=y, spec=spec, rng=rng)
    assert len(res) == 3
    np.testing.assert_array_equal(res, idx)


def test_imbalance_long_tail_partial_selection():
    """Test long_tail when desired count < class size."""

    idx = np.arange(200)
    y = np.concatenate([np.zeros(100), np.ones(100)])

    spec = ImbalanceSpec(kind="long_tail", alpha=0.5, min_per_class=1)
    rng = np.random.default_rng(0)

    res = apply_imbalance(idx=idx, y=y, spec=spec, rng=rng)

    y_res = y[res]
    counts = np.bincount(y_res.astype(int))

    assert 100 in counts
    assert 50 in counts
    assert len(res) == 150


def test_imbalance_unknown_kind():
    """Test ValueError for unknown imbalance kind."""

    spec = ImbalanceSpec(kind="invalid_kind")  # type: ignore
    idx = np.array([0])
    y = np.array([0])
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="Unknown imbalance kind"):
        apply_imbalance(idx=idx, y=y, spec=spec, rng=rng)
