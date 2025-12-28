from __future__ import annotations

import numpy as np
import pytest

from modssc.data_augmentation.ops.audio import AddNoise, TimeShift
from modssc.data_augmentation.types import AugmentationContext


@pytest.fixture
def ctx():
    return AugmentationContext(seed=0, epoch=0, sample_id=0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_audio_add_noise(ctx, rng):
    with pytest.raises(ValueError):
        AddNoise(std=-1).apply(np.zeros(10), rng=rng, ctx=ctx)

    op = AddNoise(std=0)
    arr = np.zeros(10)
    assert op.apply(arr, rng=rng, ctx=ctx) is arr

    op = AddNoise(std=1.0)
    out = op.apply(np.zeros(1000), rng=rng, ctx=ctx)
    assert np.abs(out.std() - 1.0) < 0.2


def test_audio_time_shift(ctx, rng):
    with pytest.raises(ValueError):
        TimeShift(max_frac=1.1).apply(np.zeros(10), rng=rng, ctx=ctx)

    op = TimeShift(max_frac=0)
    arr = np.arange(10)
    assert op.apply(arr, rng=rng, ctx=ctx) is arr

    op = TimeShift(max_frac=0.5)
    arr = np.zeros(10)
    arr[0] = 1

    out = op.apply(arr, rng=rng, ctx=ctx)
    assert out.sum() == 1

    assert out.shape == (10,)
