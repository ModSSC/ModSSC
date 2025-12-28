from __future__ import annotations

import numpy as np
import pytest

from modssc.data_loader.numpy_adapter import dataset_to_numpy, split_to_numpy, to_numpy
from modssc.data_loader.types import LoadedDataset, Split


def test_to_numpy_ndarray_passthrough() -> None:
    arr = np.array([1, 2, 3])
    out = to_numpy(arr)
    assert out is arr


def test_to_numpy_uses_to_numpy_method() -> None:
    class Obj:
        def to_numpy(self):
            return [1, 2, 3]

    out = to_numpy(Obj())
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))


def test_to_numpy_detach_cpu_numpy_chain() -> None:
    class WithNumpy:
        def numpy(self):
            return np.array([4, 5])

    class WithCpu:
        def cpu(self):
            return WithNumpy()

    class WithDetach:
        def detach(self):
            return WithCpu()

    out = to_numpy(WithDetach())
    np.testing.assert_array_equal(out, np.array([4, 5]))


def test_to_numpy_cpu_without_numpy_fallback() -> None:
    class CpuOnly:
        def cpu(self):
            return [7, 8]

    out = to_numpy(CpuOnly())
    np.testing.assert_array_equal(out, np.array([7, 8]))


def test_to_numpy_allow_object_fallback() -> None:
    class BadArray:
        def __array__(self, dtype=None):
            raise TypeError("nope")

    out = to_numpy(BadArray(), allow_object=True)
    assert out.dtype == object
    assert out.shape == (1,)
    assert out[0].__class__.__name__ == "BadArray"

    with pytest.raises(TypeError):
        to_numpy(BadArray(), allow_object=False)


def test_split_to_numpy_and_dataset_to_numpy() -> None:
    X = np.asarray(["a", "b"], dtype=object)
    y = np.asarray([0, 1])
    split = Split(
        X=X, y=y, edges=np.array([[0, 1], [1, 0]]), masks={"train": np.array([True, False])}
    )
    s2 = split_to_numpy(split)
    assert s2.edges is not None
    assert s2.masks is not None
    assert s2.X.dtype == object

    ds = LoadedDataset(train=split, test=Split(X=np.zeros((1, 2)), y=np.array([1])), meta={"x": 1})
    ds2 = dataset_to_numpy(ds)
    assert ds2.meta["x"] == 1
    assert ds2.train.edges is not None


class BadObject:
    def __init__(self, fail_detach=False, fail_cpu=False, fail_numpy=False, fail_array=True):
        self._fail_detach = fail_detach
        self._fail_cpu = fail_cpu
        self._fail_numpy = fail_numpy
        self._fail_array = fail_array

    def detach(self):
        if self._fail_detach:
            raise RuntimeError("detach failed")
        return self

    def cpu(self):
        if self._fail_cpu:
            raise RuntimeError("cpu failed")
        return self

    def numpy(self):
        if self._fail_numpy:
            raise RuntimeError("numpy failed")
        return np.array([1, 2])

    def __array__(self, dtype=None):
        if self._fail_array:
            raise RuntimeError("asarray failed")
        return np.array([1, 2], dtype=dtype)


def test_to_numpy_detach_fails():
    obj = BadObject(fail_detach=True, fail_cpu=True, fail_numpy=True, fail_array=True)

    res = to_numpy(obj, allow_object=True)
    assert res.dtype == object
    assert res[0] is obj


def test_to_numpy_cpu_fails():
    obj = BadObject(fail_detach=False, fail_cpu=True, fail_numpy=True, fail_array=True)

    res = to_numpy(obj, allow_object=True)
    assert res.dtype == object
    assert res[0] is obj


def test_to_numpy_numpy_fails():
    obj = BadObject(fail_detach=False, fail_cpu=False, fail_numpy=True, fail_array=True)

    res = to_numpy(obj, allow_object=True)
    assert res.dtype == object
    assert res[0] is obj


def test_to_numpy_success_after_detach_fail():
    obj = BadObject(fail_detach=True, fail_cpu=False, fail_numpy=False)

    res = to_numpy(obj)
    assert np.array_equal(res, np.array([1, 2]))


def test_to_numpy_success_after_cpu_fail():
    obj = BadObject(fail_detach=False, fail_cpu=True, fail_numpy=False)

    res = to_numpy(obj)
    assert np.array_equal(res, np.array([1, 2]))


def test_to_numpy_no_allow_object():
    obj = BadObject(fail_detach=True, fail_cpu=True, fail_numpy=True, fail_array=True)

    with pytest.raises(RuntimeError, match="asarray failed"):
        to_numpy(obj, allow_object=False)
