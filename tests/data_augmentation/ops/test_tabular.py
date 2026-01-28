from __future__ import annotations

import numpy as np
import pytest

from modssc.data_augmentation.ops.tabular import FeatureDropout, GaussianNoise, SwapNoise
from modssc.data_augmentation.types import AugmentationContext


@pytest.fixture
def ctx():
    return AugmentationContext(seed=0, epoch=0, sample_id=0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


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


def test_tabular_swap_noise_numpy(ctx, rng):
    with pytest.raises(ValueError):
        SwapNoise(p=-0.1).apply(np.zeros((2, 2)), rng=rng, ctx=ctx)
    with pytest.raises(ValueError):
        SwapNoise(p=1.1).apply(np.zeros((2, 2)), rng=rng, ctx=ctx)

    arr = np.ones((4, 4), dtype=np.float32)
    op = SwapNoise(p=0.0)
    assert op.apply(arr, rng=rng, ctx=ctx) is arr

    op = SwapNoise(p=1.0)
    out = op.apply(arr, rng=rng, ctx=ctx)
    assert out.shape == arr.shape
    assert not np.allclose(out, arr)


def test_tabular_swap_noise_torch(ctx, rng):
    torch = pytest.importorskip("torch")
    x = torch.zeros((3, 3), dtype=torch.float32)
    op = SwapNoise(p=1.0)
    out = op.apply(x, rng=rng, ctx=ctx)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)
