"""Test coverage for sampling/splitters.py."""

from __future__ import annotations

import numpy as np
import pytest

from modssc.sampling.splitters import (
    _class_counts,
    make_holdout_split,
    make_kfold_split,
    plain_kfold,
    random_split,
    stratified_holdout,
    stratified_kfold,
)


def test_random_split_edge_cases():
    """Test random_split with edge cases."""
    rng = np.random.default_rng(0)
    indices = np.arange(10)

    keep, holdout = random_split(indices, n_holdout=0, rng=rng)
    assert keep.size == 10
    assert holdout.size == 0

    keep, holdout = random_split(indices, n_holdout=-5, rng=rng)
    assert keep.size == 10
    assert holdout.size == 0

    keep, holdout = random_split(indices, n_holdout=10, rng=rng)
    assert keep.size == 0
    assert holdout.size == 10

    keep, holdout = random_split(indices, n_holdout=15, rng=rng)
    assert keep.size == 0
    assert holdout.size == 10


def test_class_counts_empty():
    classes, counts = _class_counts(np.array([], dtype=np.int64))
    assert classes.size == 0
    assert counts.size == 0


def test_class_counts_integer_bincount():
    classes, counts = _class_counts(np.array([0, 1, 1, 3], dtype=np.int64))
    assert classes.tolist() == [0, 1, 3]
    assert counts.tolist() == [1, 2, 1]


def test_class_counts_out_of_range_fallback():
    classes, counts = _class_counts(np.array([-1, 2, 2], dtype=np.int64))
    assert classes.tolist() == [-1, 2]
    assert counts.tolist() == [1, 2]


def test_stratified_holdout_edge_cases():
    """Test stratified_holdout with edge cases."""
    rng = np.random.default_rng(0)
    indices = np.arange(10)
    y = np.zeros(10)

    keep, holdout = stratified_holdout(indices, y, n_holdout=0, rng=rng)
    assert keep.size == 10
    assert holdout.size == 0

    keep, holdout = stratified_holdout(indices, y, n_holdout=10, rng=rng)
    assert keep.size == 0
    assert holdout.size == 10


def test_stratified_holdout_distribution():
    """Test stratified_holdout distribution logic."""
    rng = np.random.default_rng(42)

    y = np.array([0] * 7 + [1] * 3)
    indices = np.arange(10)

    keep, holdout = stratified_holdout(indices, y, n_holdout=3, rng=rng)

    assert holdout.size == 3
    y_holdout = y[holdout]
    unique, counts = np.unique(y_holdout, return_counts=True)
    counts_dict = dict(zip(unique, counts, strict=True))
    assert counts_dict.get(0, 0) == 2
    assert counts_dict.get(1, 0) == 1


def test_make_holdout_split_valid():
    """Test make_holdout_split with valid inputs."""
    rng = np.random.default_rng(0)
    y = np.zeros(20)

    split = make_holdout_split(
        n_samples=20,
        y=y,
        test_fraction=0.2,
        val_fraction=0.2,
        stratify=True,
        rng=rng,
    )

    assert split["test"].size == 4
    assert split["val"].size == 3
    assert split["train"].size == 13

    split = make_holdout_split(
        n_samples=20,
        y=y,
        test_fraction=0.2,
        val_fraction=0.2,
        stratify=False,
        rng=rng,
    )
    assert split["test"].size == 4
    assert split["val"].size == 3
    assert split["train"].size == 13


def test_make_holdout_split_invalid():
    """Test make_holdout_split with invalid n_samples."""
    rng = np.random.default_rng(0)
    y = np.zeros(10)
    with pytest.raises(ValueError, match="n_samples must be >= 0"):
        make_holdout_split(
            n_samples=-1, y=y, test_fraction=0.2, val_fraction=0.0, stratify=False, rng=rng
        )


def test_make_kfold_split_random_val():
    """Test make_kfold_split with val_fraction > 0 and stratify=False."""
    rng = np.random.default_rng(0)
    y = np.zeros(20)

    split = make_kfold_split(
        n_samples=20,
        y=y,
        k=2,
        fold=0,
        stratify=False,
        shuffle=False,
        val_fraction=0.2,
        rng=rng,
    )
    assert split["test"].size == 10
    assert split["val"].size == 2
    assert split["train"].size == 8


def test_make_kfold_split_stratified_val():
    """Test make_kfold_split with val_fraction > 0 and stratify=True."""
    rng = np.random.default_rng(0)
    y = np.zeros(20)
    split = make_kfold_split(
        n_samples=20,
        y=y,
        k=2,
        fold=0,
        stratify=True,
        shuffle=True,
        val_fraction=0.2,
        rng=rng,
    )
    assert split["test"].size == 10
    assert split["val"].size == 2
    assert split["train"].size == 8


def test_make_kfold_split_invalid():
    """Test make_kfold_split with invalid inputs."""
    rng = np.random.default_rng(0)
    y = np.zeros(10)

    with pytest.raises(ValueError, match="k must be >= 2"):
        make_kfold_split(
            n_samples=10, y=y, k=1, fold=0, stratify=False, shuffle=False, val_fraction=0.0, rng=rng
        )

    with pytest.raises(ValueError, match="fold must satisfy"):
        make_kfold_split(
            n_samples=10, y=y, k=2, fold=2, stratify=False, shuffle=False, val_fraction=0.0, rng=rng
        )


def test_plain_kfold_shuffle():
    """Test plain_kfold with shuffle=True."""
    rng = np.random.default_rng(0)
    indices = np.arange(10)
    folds = plain_kfold(indices, k=2, rng=rng, shuffle=True)
    assert len(folds) == 2
    assert folds[0].size == 5
    assert folds[1].size == 5
    union = np.sort(np.concatenate(folds))
    assert np.array_equal(union, indices)


def test_stratified_kfold_shuffle():
    """Test stratified_kfold with shuffle=True."""
    rng = np.random.default_rng(0)

    indices = np.arange(12)
    y = np.array([0] * 6 + [1] * 6)
    folds = stratified_kfold(indices, y, k=2, rng=rng, shuffle=True)
    assert len(folds) == 2
    assert folds[0].size == 6
    assert folds[1].size == 6
    union = np.sort(np.concatenate(folds))
    assert np.array_equal(union, indices)

    y_fold0 = y[folds[0]]
    assert np.sum(y_fold0 == 0) == 3
    assert np.sum(y_fold0 == 1) == 3


def test_plain_kfold_no_shuffle():
    """Test plain_kfold with shuffle=False."""
    rng = np.random.default_rng(0)
    indices = np.arange(10)
    folds = plain_kfold(indices, k=2, rng=rng, shuffle=False)
    assert len(folds) == 2

    assert np.array_equal(folds[0], np.arange(5))
    assert np.array_equal(folds[1], np.arange(5, 10))


def test_stratified_kfold_no_shuffle():
    """Test stratified_kfold with shuffle=False."""
    rng = np.random.default_rng(0)

    indices = np.arange(12)
    y = np.array([0] * 6 + [1] * 6)
    folds = stratified_kfold(indices, y, k=2, rng=rng, shuffle=False)
    assert len(folds) == 2
    assert folds[0].size == 6
    assert folds[1].size == 6

    assert np.array_equal(folds[0], np.array([0, 1, 2, 6, 7, 8]))
    assert np.array_equal(folds[1], np.array([3, 4, 5, 9, 10, 11]))


def test_make_kfold_split_no_val():
    """Test make_kfold_split with val_fraction=0.0."""
    rng = np.random.default_rng(0)
    y = np.zeros(10)
    split = make_kfold_split(
        n_samples=10,
        y=y,
        k=2,
        fold=0,
        stratify=False,
        shuffle=False,
        val_fraction=0.0,
        rng=rng,
    )
    assert split["test"].size == 5
    assert split["val"].size == 0
    assert split["train"].size == 5


def test_make_kfold_split_empty():
    """Test make_kfold_split with n_samples=0."""
    rng = np.random.default_rng(0)
    y = np.zeros(0)
    split = make_kfold_split(
        n_samples=0,
        y=y,
        k=2,
        fold=0,
        stratify=True,
        shuffle=False,
        val_fraction=0.0,
        rng=rng,
    )
    assert split["test"].size == 0
    assert split["val"].size == 0
    assert split["train"].size == 0
