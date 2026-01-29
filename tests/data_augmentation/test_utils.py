from __future__ import annotations

import numpy as np
import pytest

from modssc.data_augmentation.utils import (
    copy_like,
    ensure_edge_index_2xE,
    is_torch_tensor,
    resolve_backend,
    split_image_channels_last,
    to_numpy,
)


def test_is_torch_tensor() -> None:
    class MockTensor:
        __module__ = "torch.tensor"
        shape = (1,)
        dtype = float
        device = "cpu"

    assert is_torch_tensor(MockTensor())
    assert not is_torch_tensor(np.array([1]))
    assert not is_torch_tensor([1])
    assert not is_torch_tensor({"x": MockTensor()})


def test_to_numpy() -> None:
    arr = np.array([1, 2])
    assert to_numpy(arr) is arr

    assert np.array_equal(to_numpy([1, 2]), arr)

    class MockTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([1, 2])

    assert np.array_equal(to_numpy(MockTensor()), arr)


def test_resolve_backend() -> None:
    class MockTensor:
        __module__ = "torch.tensor"
        shape = (1,)
        dtype = float
        device = "cpu"

    assert resolve_backend(MockTensor(), "auto") == "torch"
    assert resolve_backend(np.array([1]), "auto") == "numpy"
    assert resolve_backend(None, "tf") == "tf"


def test_ensure_edge_index_2xE() -> None:
    ei = np.zeros((2, 10), dtype=np.int64)
    assert ensure_edge_index_2xE(ei).shape == (2, 10)

    ei_t = np.zeros((10, 2), dtype=np.int64)
    assert ensure_edge_index_2xE(ei_t).shape == (2, 10)

    with pytest.raises(ValueError, match="must be 2D"):
        ensure_edge_index_2xE(np.zeros((10,)))

    with pytest.raises(ValueError, match="must have shape"):
        ensure_edge_index_2xE(np.zeros((3, 3)))


def test_ensure_edge_index_2xE_torch_mock() -> None:
    class MockTorchTensor:
        __module__ = "torch.tensor"

        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr
            self.shape = arr.shape
            self.ndim = arr.ndim
            self.dtype = arr.dtype
            self.device = "cpu"

        def t(self):
            return MockTorchTensor(self._arr.T)

    ei = MockTorchTensor(np.zeros((10, 2), dtype=np.int64))
    out = ensure_edge_index_2xE(ei)
    assert isinstance(out, MockTorchTensor)
    assert out.shape == (2, 10)

    ei2 = MockTorchTensor(np.zeros((2, 4), dtype=np.int64))
    out2 = ensure_edge_index_2xE(ei2)
    assert out2.shape == (2, 4)


def test_ensure_edge_index_2xE_torch_invalid_shapes() -> None:
    class MockTorchTensor:
        __module__ = "torch.tensor"

        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr
            self.shape = arr.shape
            self.ndim = arr.ndim
            self.dtype = arr.dtype
            self.device = "cpu"

        def t(self):
            return MockTorchTensor(self._arr.T)

    with pytest.raises(ValueError, match="must be 2D"):
        ensure_edge_index_2xE(MockTorchTensor(np.zeros((3,), dtype=np.int64)))

    with pytest.raises(ValueError, match="must have shape"):
        ensure_edge_index_2xE(MockTorchTensor(np.zeros((3, 3), dtype=np.int64)))


def test_copy_like() -> None:
    arr = np.array([1, 2])
    cp = copy_like(arr)
    assert np.array_equal(cp, arr)
    assert cp is not arr

    class MockTensor:
        __module__ = "torch.tensor"
        shape = (1,)
        dtype = float
        device = "cpu"

        def clone(self):
            return "cloned"

    assert copy_like(MockTensor()) == "cloned"

    class Copyable:
        def copy(self):
            return "copied"

    assert copy_like(Copyable()) == "copied"

    obj = object()
    assert copy_like(obj) is obj


def test_split_image_channels_last() -> None:
    img = np.zeros((10, 10))
    _, layout = split_image_channels_last(img)
    assert layout == "hw"

    img = np.zeros((3, 10, 10))
    _, layout = split_image_channels_last(img)
    assert layout == "chw"

    img = np.zeros((10, 10, 3))
    _, layout = split_image_channels_last(img)
    assert layout == "hwc"

    with pytest.raises(ValueError, match="Expected image with ndim 2 or 3"):
        split_image_channels_last(np.zeros((10,)))
