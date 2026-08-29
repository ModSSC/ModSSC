from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from modssc.data_augmentation.ops.vision import (
    Cutout,
    GaussianNoise,
    RandAugment,
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


def test_vision_randaugment_is_deterministic_and_preserves_layout(ctx):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    op = RandAugment(num_ops=2, magnitude=10)
    image = torch.linspace(0.0, 1.0, 3 * 16 * 16).reshape(3, 16, 16)

    out_a = op.apply(image, rng=np.random.default_rng(42), ctx=ctx)
    out_b = op.apply(image, rng=np.random.default_rng(42), ctx=ctx)

    assert out_a.shape == image.shape
    assert out_a.dtype == image.dtype
    assert out_a.device == image.device
    assert bool(torch.isfinite(out_a).all())
    assert float(out_a.min()) >= 0.0
    assert float(out_a.max()) <= 1.0
    torch.testing.assert_close(out_a, out_b)


def test_vision_randaugment_float_matches_uint8_for_solarize_policy(ctx):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    class FixedPolicyRng:
        @staticmethod
        def integers(*_args, **_kwargs):
            # Torchvision's policy selected with torch seed 2 includes Solarize.
            return 2

    image_uint8 = torch.arange(3 * 32 * 32, dtype=torch.int64).remainder(256).to(torch.uint8)
    image_uint8 = image_uint8.reshape(3, 32, 32)
    image_float = image_uint8.to(torch.float64).div(255.0)
    op = RandAugment(num_ops=2, magnitude=10)

    output_uint8 = op.apply(image_uint8, rng=FixedPolicyRng(), ctx=ctx)
    output_float = op.apply(image_float, rng=FixedPolicyRng(), ctx=ctx)

    assert output_uint8.dtype == torch.uint8
    assert output_float.dtype == torch.float64
    assert output_float.device == image_float.device
    torch.testing.assert_close(output_float, output_uint8.to(torch.float64).div(255.0))


def test_vision_randaugment_float_is_safe_across_policy_seeds(ctx):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    image = torch.linspace(0.0, 1.0, 3 * 16 * 16, dtype=torch.float32).reshape(3, 16, 16)
    op = RandAugment(num_ops=2, magnitude=10)

    for seed in range(32):
        output = op.apply(image, rng=np.random.default_rng(seed), ctx=ctx)
        assert output.shape == image.shape
        assert output.dtype == image.dtype
        assert output.device == image.device
        assert bool(torch.isfinite(output).all())
        assert float(output.min()) >= 0.0
        assert float(output.max()) <= 1.0


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(np.full((3, 8, 8), -0.01, dtype=np.float32), id="below-zero"),
        pytest.param(np.full((3, 8, 8), 1.01, dtype=np.float32), id="above-one"),
        pytest.param(np.full((3, 8, 8), np.nan, dtype=np.float32), id="not-finite"),
    ],
)
def test_vision_randaugment_rejects_invalid_float_range(ctx, image):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    with pytest.raises(ValueError, match=r"finite|\[0, 1\]"):
        RandAugment(num_ops=1, magnitude=0).apply(
            torch.from_numpy(image), rng=np.random.default_rng(0), ctx=ctx
        )


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((3, 0, 8), "non-empty"),
        ((2, 8, 8), "one or three"),
    ],
)
def test_vision_randaugment_rejects_invalid_image_shapes(ctx, shape, message):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    with pytest.raises(ValueError, match=message):
        RandAugment(num_ops=1, magnitude=0).apply(
            torch.zeros(shape, dtype=torch.uint8), rng=np.random.default_rng(0), ctx=ctx
        )


@pytest.mark.parametrize("fill", [-0.1, 1.1, float("nan"), (0.0, 1.1, 0.0)])
def test_vision_randaugment_validates_float_fill(ctx, fill):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    with pytest.raises(ValueError, match="fill"):
        RandAugment(num_ops=1, magnitude=0, fill=fill).apply(
            torch.zeros((3, 8, 8), dtype=torch.float32),
            rng=np.random.default_rng(0),
            ctx=ctx,
        )


@pytest.mark.parametrize("fill", [0.5, (0.0, 0.5, 1.0)])
def test_vision_randaugment_scales_float_fill(ctx, fill):
    torch = pytest.importorskip("torch")

    captured = {}

    def build_transform(**kwargs):
        captured.update(kwargs)
        return lambda value: value

    fake_torchvision = SimpleNamespace(
        transforms=SimpleNamespace(
            InterpolationMode=SimpleNamespace(NEAREST="nearest"),
            RandAugment=build_transform,
        )
    )
    real_import = __import__("importlib").import_module

    def fake_import(name):
        return fake_torchvision if name == "torchvision" else real_import(name)

    with patch("importlib.import_module", side_effect=fake_import):
        output = RandAugment(num_ops=1, magnitude=0, fill=fill).apply(
            torch.zeros((3, 8, 8), dtype=torch.float32),
            rng=np.random.default_rng(0),
            ctx=ctx,
        )

    expected = 128 if isinstance(fill, float) else [0, 128, 255]
    assert captured["fill"] == expected
    assert output.dtype == torch.float32


def test_vision_randaugment_rejects_transform_shape_change(ctx):
    torch = pytest.importorskip("torch")

    fake_torchvision = SimpleNamespace(
        transforms=SimpleNamespace(
            InterpolationMode=SimpleNamespace(NEAREST="nearest"),
            RandAugment=lambda **_kwargs: lambda value: value[..., :-1],
        )
    )
    real_import = __import__("importlib").import_module

    def fake_import(name):
        return fake_torchvision if name == "torchvision" else real_import(name)

    with (
        patch("importlib.import_module", side_effect=fake_import),
        pytest.raises(ValueError, match="changed the image shape"),
    ):
        RandAugment(num_ops=1, magnitude=0).apply(
            torch.zeros((3, 8, 8), dtype=torch.uint8),
            rng=np.random.default_rng(0),
            ctx=ctx,
        )


@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 3)])
def test_vision_randaugment_preserves_numpy_layouts_and_tuple_fill(ctx, shape):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    op = RandAugment(num_ops=1, magnitude=0, fill=(0.0,))

    output = op.apply(image, rng=np.random.default_rng(3), ctx=ctx)

    assert isinstance(output, np.ndarray)
    assert output.shape == image.shape


def test_vision_randaugment_seeds_cuda_policy_context(ctx, monkeypatch):
    class FakeTensor:
        device = SimpleNamespace(type="cuda")
        dtype = "uint8"

        def __init__(self, shape=(8, 8)):
            self.shape = shape

        def unsqueeze(self, _dim):
            return FakeTensor((1, *self.shape))

        def squeeze(self, _dim):
            return FakeTensor(self.shape[1:])

        def is_floating_point(self):
            return False

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((8, 8), dtype=np.uint8)

    calls = []
    fake_torch = SimpleNamespace(
        from_numpy=lambda _value: FakeTensor(),
        random=SimpleNamespace(fork_rng=lambda **_kwargs: nullcontext()),
        manual_seed=lambda seed: calls.append(("cpu", seed)),
        cuda=SimpleNamespace(manual_seed_all=lambda seed: calls.append(("cuda", seed))),
    )

    def fake_transform(value):
        return value

    fake_torchvision = SimpleNamespace(
        transforms=SimpleNamespace(
            InterpolationMode=SimpleNamespace(NEAREST="nearest"),
            RandAugment=lambda **_kwargs: fake_transform,
        )
    )

    def fake_import(name):
        return fake_torch if name == "torch" else fake_torchvision

    monkeypatch.setattr("importlib.import_module", fake_import)
    output = RandAugment(num_ops=1, magnitude=0).apply(
        np.zeros((8, 8), dtype=np.uint8),
        rng=np.random.default_rng(7),
        ctx=ctx,
    )

    assert output.shape == (8, 8)
    assert [kind for kind, _ in calls] == ["cpu", "cuda"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_ops": 0}, "num_ops"),
        ({"num_magnitude_bins": 1}, "num_magnitude_bins"),
        ({"magnitude": 31}, "magnitude"),
    ],
)
def test_vision_randaugment_validates_parameters(ctx, rng, kwargs, message):
    with pytest.raises(ValueError, match=message):
        RandAugment(**kwargs).apply(np.zeros((8, 8), dtype=np.uint8), rng=rng, ctx=ctx)


from ._vision_numpy import *  # noqa: E402,F401,F403
