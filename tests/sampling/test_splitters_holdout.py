from __future__ import annotations

import numpy as np

from modssc.sampling.splitters import make_holdout_split


def test_holdout_sizes_and_disjoint() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    rng = np.random.default_rng(0)
    parts = make_holdout_split(
        n_samples=6, y=y, test_fraction=0.33, val_fraction=0.25, stratify=True, rng=rng
    )

    train, val, test = parts["train"], parts["val"], parts["test"]
    assert train.dtype.kind in ("i", "u")
    assert np.intersect1d(train, val).size == 0
    assert np.intersect1d(train, test).size == 0
    assert np.intersect1d(val, test).size == 0
    assert train.size + val.size + test.size == 6


def test_holdout_no_stratify() -> None:
    y = np.arange(10)
    rng = np.random.default_rng(0)
    parts = make_holdout_split(
        n_samples=10, y=y, test_fraction=0.2, val_fraction=0.0, stratify=False, rng=rng
    )
    assert parts["val"].size == 0
    assert parts["test"].size == 2
