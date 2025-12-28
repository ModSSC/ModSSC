from __future__ import annotations

import numpy as np

from modssc.data_augmentation import AugmentationContext
from modssc.data_augmentation.registry import get_op
from modssc.data_augmentation.utils import make_numpy_rng


def test_horizontal_flip_numpy() -> None:
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    op = get_op("vision.random_horizontal_flip", p=1.0)
    ctx = AugmentationContext(seed=0, epoch=0, sample_id=0, modality="vision")
    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    out = op.apply(img, rng=rng, ctx=ctx)
    assert np.array_equal(out, np.flip(img, axis=1))


def test_cutout_masks_some_values() -> None:
    img = np.ones((8, 8, 3), dtype=np.float32)
    op = get_op("vision.cutout", frac=0.5, fill=0.0)
    ctx = AugmentationContext(seed=0, epoch=0, sample_id=0, modality="vision")
    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    out = op.apply(img, rng=rng, ctx=ctx)
    assert out.shape == img.shape
    assert (out == 0.0).any()
    assert (out == 1.0).any()


def test_random_crop_pad_preserves_shape() -> None:
    img = np.arange(3 * 4 * 1, dtype=np.float32).reshape(3, 4, 1)
    op = get_op("vision.random_crop_pad", pad=2)
    ctx = AugmentationContext(seed=0, epoch=0, sample_id=1, modality="vision")
    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    out = op.apply(img, rng=rng, ctx=ctx)
    assert out.shape == img.shape


def test_gaussian_noise_preserves_dtype() -> None:
    img = np.zeros((4, 4, 3), dtype=np.float32)
    op = get_op("vision.gaussian_noise", std=0.1)
    ctx = AugmentationContext(seed=0, epoch=0, sample_id=2, modality="vision")
    rng = make_numpy_rng(seed=ctx.seed, epoch=ctx.epoch, sample_id=ctx.sample_id)
    out = op.apply(img, rng=rng, ctx=ctx)
    assert out.dtype == np.float32
