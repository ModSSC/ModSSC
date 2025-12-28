import numpy as np

from modssc.sampling.stats import _class_counts, class_distribution


def test_class_counts_empty():
    classes, counts = _class_counts(np.array([], dtype=np.int64))
    assert classes.size == 0
    assert counts.size == 0


def test_class_counts_unique_non_int():
    classes, counts = _class_counts(np.array(["a", "b", "a"], dtype=object))
    assert set(classes.tolist()) == {"a", "b"}
    assert counts.sum() == 3


def test_class_counts_out_of_range_fallback():
    classes, counts = _class_counts(np.array([-1, 5, 5], dtype=np.int64))
    assert classes.tolist() == [-1, 5]
    assert counts.tolist() == [1, 2]


def test_class_distribution_empty_idx():
    res = class_distribution(np.array([0, 1]), np.array([], dtype=np.int64))
    assert res == {"n": 0, "classes": {}}
