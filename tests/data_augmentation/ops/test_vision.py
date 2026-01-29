from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from modssc.data_augmentation.ops.vision import (
    Cutout,
    GaussianNoise,
    RandomCropPad,
    RandomHorizontalFlip,
    _numpy_hw_layout,
    _torch_hw_layout,
)
from modssc.data_augmentation.types import AugmentationContext


@pytest.fixture
def ctx():
    return AugmentationContext(seed=0, epoch=0, sample_id=0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_vision_layout_helpers():
    assert _numpy_hw_layout(np.zeros((10, 10))) == (10, 10, "hw")

    assert _numpy_hw_layout(np.zeros((10, 10, 3))) == (10, 10, "hwc")

    assert _numpy_hw_layout(np.zeros((3, 10, 10))) == (10, 10, "chw")

    class MockTensor:
        def __init__(self, shape):
            self.shape = shape

    assert _torch_hw_layout(MockTensor((10, 10))) == (10, 10, "hw")

    assert _torch_hw_layout(MockTensor((10, 10, 3))) == (10, 10, "hwc")

    assert _torch_hw_layout(MockTensor((3, 10, 10))) == (10, 10, "chw")

    with pytest.raises(ValueError):
        _torch_hw_layout(MockTensor((10,)))


def test_random_horizontal_flip_numpy_chw(ctx, rng):
    op = RandomHorizontalFlip(p=1.0)

    x = np.array([[[1, 2], [3, 4]]])

    out = op.apply(x, rng=rng, ctx=ctx)

    expected = np.array([[[2, 1], [4, 3]]])
    np.testing.assert_array_equal(out, expected)


def test_random_horizontal_flip_numpy_hwc(ctx, rng):
    op = RandomHorizontalFlip(p=1.0)

    x = np.array([[[1], [2]], [[3], [4]]])

    out = op.apply(x, rng=rng, ctx=ctx)

    expected = np.array([[[2], [1]], [[4], [3]]])
    np.testing.assert_array_equal(out, expected)


def test_random_horizontal_flip_numpy_hw(ctx, rng):
    op = RandomHorizontalFlip(p=1.0)

    x = np.array([[1, 2], [3, 4]])

    out = op.apply(x, rng=rng, ctx=ctx)

    expected = np.array([[2, 1], [4, 3]])
    np.testing.assert_array_equal(out, expected)


def test_random_horizontal_flip_numpy_chw_forced(ctx, rng):
    op = RandomHorizontalFlip(p=1.0)
    x = np.zeros((1, 2, 2))

    with patch("modssc.data_augmentation.ops.vision._numpy_hw_layout", return_value=(2, 2, "chw")):
        op.apply(x, rng=rng, ctx=ctx)

    op = RandomHorizontalFlip(p=1.0)

    x = np.array([[1, 2], [3, 4]])

    out = op.apply(x, rng=rng, ctx=ctx)

    expected = np.array([[2, 1], [4, 3]])
    np.testing.assert_array_equal(out, expected)


def test_vision_random_horizontal_flip(ctx, rng):
    with pytest.raises(ValueError):
        RandomHorizontalFlip(p=1.1).apply(np.zeros((2, 2)), rng=rng, ctx=ctx)

    op = RandomHorizontalFlip(p=0)
    arr = np.array([[1, 2], [3, 4]])
    assert np.array_equal(op.apply(arr, rng=rng, ctx=ctx), arr)

    op = RandomHorizontalFlip(p=1)

    out = op.apply(arr, rng=rng, ctx=ctx)
    assert np.array_equal(out, [[2, 1], [4, 3]])

    arr_hwc = np.zeros((2, 2, 1))
    arr_hwc[0, 0, 0] = 1
    arr_hwc[0, 1, 0] = 2
    out_hwc = op.apply(arr_hwc, rng=rng, ctx=ctx)
    assert out_hwc[0, 0, 0] == 2
    assert out_hwc[0, 1, 0] == 1

    arr_chw = np.zeros((1, 2, 2))
    arr_chw[0, 0, 0] = 1
    arr_chw[0, 0, 1] = 2
    out_chw = op.apply(arr_chw, rng=rng, ctx=ctx)
    assert out_chw[0, 0, 0] == 2
    assert out_chw[0, 0, 1] == 1


def test_vision_gaussian_noise(ctx, rng):
    with pytest.raises(ValueError):
        GaussianNoise(std=-1).apply(np.zeros((2, 2)), rng=rng, ctx=ctx)

    op = GaussianNoise(std=0)
    arr = np.zeros((2, 2))
    assert op.apply(arr, rng=rng, ctx=ctx) is arr


def test_vision_cutout(ctx, rng):
    with pytest.raises(ValueError):
        Cutout(frac=1.1).apply(np.zeros((10, 10)), rng=rng, ctx=ctx)

    op = Cutout(frac=0)
    arr = np.zeros((10, 10))
    assert op.apply(arr, rng=rng, ctx=ctx) is arr

    op = Cutout(frac=0.5, fill=1.0)
    arr = np.zeros((10, 10))
    out = op.apply(arr, rng=rng, ctx=ctx)
    assert out.sum() > 0
    assert out.sum() < 100

    arr = np.zeros((10, 10, 1))
    out = op.apply(arr, rng=rng, ctx=ctx)
    assert out.sum() > 0

    arr = np.zeros((1, 10, 10))
    out = op.apply(arr, rng=rng, ctx=ctx)
    assert out.sum() > 0


def test_vision_cutout_conflicting_and_empty(ctx, rng):
    with pytest.raises(ValueError, match="Use either frac"):
        Cutout(frac=0.5, length=8).apply(np.zeros((10, 10)), rng=rng, ctx=ctx)
    with pytest.raises(ValueError, match="Use either frac"):
        Cutout(frac=0.5, n_holes=2).apply(np.zeros((10, 10)), rng=rng, ctx=ctx)

    arr = np.zeros((4, 4))
    op = Cutout(length=0, n_holes=1)
    assert op.apply(arr, rng=rng, ctx=ctx) is arr

    op = Cutout(length=2, n_holes=0)
    assert op.apply(arr, rng=rng, ctx=ctx) is arr


def test_vision_cutout_length_path_numpy(ctx, rng):
    op = Cutout(length=2, n_holes=1, fill=1.0)
    arr = np.zeros((6, 6))
    out = op.apply(arr, rng=rng, ctx=ctx)
    assert out.shape == arr.shape


def test_vision_cutout_length_path_torch(ctx, rng):
    torch = pytest.importorskip("torch")
    op = Cutout(length=2, n_holes=1, fill=1.0)
    x = torch.zeros((6, 6))
    out = op.apply(x, rng=rng, ctx=ctx)
    assert out.shape == x.shape


def test_vision_random_crop_pad(ctx, rng):
    with pytest.raises(ValueError):
        RandomCropPad(pad=-1).apply(np.zeros((10, 10)), rng=rng, ctx=ctx)

    with pytest.raises(ValueError, match="Use either pad or padding"):
        RandomCropPad(pad=2, padding=3).apply(np.zeros((10, 10)), rng=rng, ctx=ctx)

    op = RandomCropPad(pad=0)
    arr = np.zeros((10, 10))
    assert op.apply(arr, rng=rng, ctx=ctx) is arr

    op = RandomCropPad(pad=2)

    arr = np.zeros((10, 10))
    out = op.apply(arr, rng=rng, ctx=ctx)
    assert out.shape == (10, 10)

    arr = np.zeros((10, 10, 3))
    out = op.apply(arr, rng=rng, ctx=ctx)
    assert out.shape == (10, 10, 3)

    arr = np.zeros((3, 10, 10))
    out = op.apply(arr, rng=rng, ctx=ctx)
    assert out.shape == (3, 10, 10)

    op_padding = RandomCropPad(padding=2)
    arr = np.zeros((10, 10))
    out = op_padding.apply(arr, rng=rng, ctx=ctx)
    assert out.shape == (10, 10)
