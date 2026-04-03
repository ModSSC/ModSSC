from __future__ import annotations

import numpy as np
import pytest

from modssc.sampling.splitters import make_kfold_split


def test_kfold_basic() -> None:
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    rng = np.random.default_rng(0)
    parts = make_kfold_split(
        n_samples=8, y=y, k=4, fold=1, stratify=True, shuffle=True, val_fraction=0.0, rng=rng
    )
    train, val, test = parts["train"], parts["val"], parts["test"]
    assert val.size == 0
    assert test.size == 2
    assert train.size == 6
    assert np.intersect1d(train, test).size == 0


def test_kfold_with_val_fraction() -> None:
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    rng = np.random.default_rng(0)
    parts = make_kfold_split(
        n_samples=8, y=y, k=4, fold=0, stratify=True, shuffle=True, val_fraction=0.25, rng=rng
    )
    assert parts["val"].size > 0
    assert np.intersect1d(parts["train"], parts["val"]).size == 0
    assert np.intersect1d(parts["train"], parts["test"]).size == 0


def test_kfold_invalid_k_fold() -> None:
    y = np.zeros(10)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        make_kfold_split(
            n_samples=10, y=y, k=1, fold=0, stratify=False, shuffle=False, val_fraction=0.0, rng=rng
        )
    with pytest.raises(ValueError):
        make_kfold_split(
            n_samples=10, y=y, k=5, fold=5, stratify=False, shuffle=False, val_fraction=0.0, rng=rng
        )
