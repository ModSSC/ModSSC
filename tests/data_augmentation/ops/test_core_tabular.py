from __future__ import annotations

import numpy as np
import pytest

from modssc.data_augmentation.ops.core import EnsureFloat32, Identity
from modssc.data_augmentation.ops.tabular import FeatureDropout, GaussianNoise
from modssc.data_augmentation.types import AugmentationContext


@pytest.fixture
def ctx():
    return AugmentationContext(seed=0, epoch=0, sample_id=0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_core_identity(ctx, rng):
    op = Identity()
    assert op.apply("test", rng=rng, ctx=ctx) == "test"


def test_core_ensure_float32(ctx, rng):
    op = EnsureFloat32()

    arr = np.array([1, 2], dtype=np.int64)
    out = op.apply(arr, rng=rng, ctx=ctx)
    assert out.dtype == np.float32
    assert np.array_equal(out, arr)

    arr_f32 = np.array([1.0, 2.0], dtype=np.float32)
    out = op.apply(arr_f32, rng=rng, ctx=ctx)
    assert out is arr_f32

    class MockTensor:
        __module__ = "torch.tensor"
        shape = (1,)
        dtype = "float64"
        device = "cpu"

        def to(self, dtype):
            return "converted"

    pass


def test_tabular_gaussian_noise(ctx, rng):
    with pytest.raises(ValueError):
        GaussianNoise(std=-1).apply(np.zeros(5), rng=rng, ctx=ctx)

    op = GaussianNoise(std=0)
    arr = np.zeros(5)
    assert op.apply(arr, rng=rng, ctx=ctx) is arr

    op = GaussianNoise(std=1.0)
    out = op.apply(np.zeros(1000), rng=rng, ctx=ctx)
    assert np.abs(out.mean()) < 0.2
    assert np.abs(out.std() - 1.0) < 0.2


def test_tabular_feature_dropout(ctx, rng):
    with pytest.raises(ValueError):
        FeatureDropout(p=-0.1).apply(np.zeros(5), rng=rng, ctx=ctx)
    with pytest.raises(ValueError):
        FeatureDropout(p=1.1).apply(np.zeros(5), rng=rng, ctx=ctx)

    op = FeatureDropout(p=0)
    arr = np.ones(5)
    assert op.apply(arr, rng=rng, ctx=ctx) is arr

    op = FeatureDropout(p=0.5)
    arr = np.ones(1000)
    out = op.apply(arr, rng=rng, ctx=ctx)

    zeros = (out == 0).sum()
    assert 400 < zeros < 600
