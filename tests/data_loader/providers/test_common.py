from __future__ import annotations

import numpy as np

from modssc.data_loader.providers.common import (
    apply_class_filter,
    apply_limits,
    apply_limits_to_split,
    limit_samples,
    normalize_filter,
)
from modssc.data_loader.types import Split


def test_normalize_filter_variants():
    assert normalize_filter(None) is None
    assert set(normalize_filter({1, 2})) == {1, 2}
    assert normalize_filter("x") == ["x"]


def test_apply_class_filter():
    X = np.array(["a", "b", "c"])
    y = np.array([0, 1, 0])
    X_f, y_f = apply_class_filter(X, y, class_filter=[0])
    assert X_f.tolist() == ["a", "c"]
    assert y_f.tolist() == [0, 0]


def test_limit_samples_with_and_without_seed():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])

    X_empty, y_empty = limit_samples(X, y, max_samples=0, seed=None)
    assert X_empty.size == 0
    assert y_empty.size == 0

    X_take_no_seed, y_take_no_seed = limit_samples(X, y, max_samples=2, seed=None)
    assert X_take_no_seed.shape == (2, 1)
    assert y_take_no_seed.shape == (2,)

    X_take, y_take = limit_samples(X, y, max_samples=2, seed=123)
    assert X_take.shape == (2, 1)
    assert y_take.shape == (2,)


def test_apply_limits_combines_filter_and_limit():
    X = np.array([[10], [20], [30], [40]])
    y = np.array([0, 1, 1, 0])
    X_l, y_l = apply_limits(X, y, class_filter=[0], max_samples=1, seed=7)
    assert X_l.shape == (1, 1)
    assert y_l.tolist() == [0]


def test_apply_limits_to_split_none_and_values():
    assert apply_limits_to_split(None, class_filter=None, max_samples=None, seed=None) is None

    split = Split(X=np.array(["a", "b", "c"]), y=np.array([0, 1, 0]))
    out = apply_limits_to_split(split, class_filter=[0], max_samples=1, seed=11)
    assert out is not None
    assert out.X.shape == (1,)
    assert out.y.shape == (1,)
    assert int(out.y[0]) == 0
