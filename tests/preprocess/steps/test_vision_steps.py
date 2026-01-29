from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.vision.openclip import OpenClipStep
from modssc.preprocess.store import ArtifactStore


def test_channels_order_step():
    from modssc.preprocess.steps.vision.channels_order import ChannelsOrderStep

    step = ChannelsOrderStep(order="NCHW")
    step_invalid = ChannelsOrderStep(order="noop")
    store = ArtifactStore()
    rng = np.random.default_rng(42)

    img_nhwc = np.zeros((2, 32, 32, 3))
    store.set("raw.X", img_nhwc)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 3, 32, 32)
    res = step_invalid.transform(store, rng=rng)
    assert res["raw.X"] is img_nhwc

    step_nhwc = ChannelsOrderStep(order="NHWC")
    img_nchw = np.zeros((2, 3, 32, 32))
    store.set("raw.X", img_nchw)
    res = step_nhwc.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 32, 32, 3)

    img_gray = np.zeros((2, 10, 10))
    store.set("raw.X", img_gray)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 1, 10, 10)
    res = step_nhwc.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 10, 10, 1)

    img_single = np.zeros((32, 32, 3))
    store.set("raw.X", img_single)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (3, 32, 32)

    store.set("raw.X", np.zeros((10,)))
    res = step.transform(store, rng=rng)
    assert res["raw.X"].ndim == 1

    img_ambiguous = np.zeros((2, 10, 10, 10))
    store.set("raw.X", img_ambiguous)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 10, 10, 10)

    img_ambiguous_ch = np.zeros((2, 3, 7, 3))
    store.set("raw.X", img_ambiguous_ch)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 3, 3, 7)

    img_nchw_2 = np.zeros((2, 3, 32, 32))
    store.set("raw.X", img_nchw_2)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 3, 32, 32)


def test_ensure_num_channels_step():
    from modssc.preprocess.steps.vision.ensure_num_channels import EnsureNumChannelsStep

    step = EnsureNumChannelsStep(num_channels=3)
    store = ArtifactStore()
    rng = np.random.default_rng(42)

    img_1c = np.zeros((2, 32, 32, 1))
    store.set("raw.X", img_1c)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 32, 32, 3)

    img_4c = np.zeros((2, 32, 32, 4))
    store.set("raw.X", img_4c)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 32, 32, 3)

    img_2c = np.zeros((2, 32, 32, 2))
    store.set("raw.X", img_2c)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 32, 32, 3)

    img_nchw = np.zeros((2, 1, 32, 32))
    store.set("raw.X", img_nchw)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 3, 32, 32)

    img_single = np.zeros((10, 10, 1))
    store.set("raw.X", img_single)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (10, 10, 3)

    img_3c = np.zeros((2, 8, 8, 3))
    store.set("raw.X", img_3c)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 8, 8, 3)

    store.set("raw.X", np.zeros((10,)))
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (10,)

    step_4 = EnsureNumChannelsStep(num_channels=4)
    img_ambiguous_ch = np.zeros((2, 3, 7, 3))
    store.set("raw.X", img_ambiguous_ch)
    res = step_4.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 3, 7, 4)


def test_openclip_step_caches_encoder():
    with patch("modssc.preprocess.steps.vision.openclip.load_encoder") as mock_load:
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.zeros((1, 2), dtype=np.float32)
        mock_load.return_value = mock_encoder

        step = OpenClipStep(device="cpu")
        store = ArtifactStore()
        rng = np.random.default_rng(0)
        store.set("raw.X", np.zeros((1, 2, 2, 3), dtype=np.float32))

        res = step.transform(store, rng=rng)
        res2 = step.transform(store, rng=rng)

        assert res["features.X"].shape == (1, 2)
        assert res2["features.X"].shape == (1, 2)
        assert mock_load.call_count == 1


def test_normalize_step():
    from modssc.preprocess.steps.vision.normalize import NormalizeStep

    step = NormalizeStep(mean=(0.5,), std=(0.5,))
    store = ArtifactStore()
    rng = np.random.default_rng(42)

    img = np.ones((2, 10, 10, 1), dtype=np.float32) * 0.5
    store.set("raw.X", img)
    res = step.transform(store, rng=rng)

    assert np.allclose(res["raw.X"], 0.0)

    img_nchw = np.ones((2, 1, 10, 10), dtype=np.float32)
    store.set("raw.X", img_nchw)
    res = step.transform(store, rng=rng)

    assert np.allclose(res["raw.X"], 1.0)

    img_single = np.ones((10, 10, 1), dtype=np.float32) * 0.5
    store.set("raw.X", img_single)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (10, 10, 1)
    assert np.allclose(res["raw.X"], 0.0)

    img_gray = np.zeros((2, 10, 10), dtype=np.float32)
    store.set("raw.X", img_gray)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 1, 10, 10)

    store.set("raw.X", np.zeros((10,)))
    res = step.transform(store, rng=rng)
    assert res["raw.X"].ndim == 1


def test_open_clip_step():
    with patch("modssc.preprocess.steps.vision.openclip.load_encoder") as mock_load:
        from modssc.preprocess.steps.vision.openclip import OpenClipStep

        mock_encoder = MagicMock()
        mock_load.return_value = mock_encoder
        mock_encoder.encode.return_value = np.zeros((2, 512), dtype=np.float32)

        step = OpenClipStep(device="cpu")
        store = ArtifactStore()
        rng = np.random.default_rng(42)
        store.set("raw.X", np.zeros((2, 32, 32, 3)))

        res = step.transform(store, rng=rng)
        assert res["features.X"].shape == (2, 512)
        mock_load.assert_called_with("openclip:ViT-B-32/openai", device="cpu")

        step = OpenClipStep(device=None)
        step.transform(store, rng=rng)
        mock_load.assert_called_with("openclip:ViT-B-32/openai")


def test_resize_step():
    from modssc.preprocess.steps.vision.resize import ResizeStep

    step = ResizeStep(height=4, width=4)
    store = ArtifactStore()
    rng = np.random.default_rng(42)

    img = np.zeros((2, 8, 8, 3))
    store.set("raw.X", img)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 4, 4, 3)

    img_nchw = np.zeros((2, 3, 8, 8))
    store.set("raw.X", img_nchw)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 3, 4, 4)

    img_gray = np.zeros((2, 8, 8))
    store.set("raw.X", img_gray)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 1, 4, 4)

    img_ambiguous_ch = np.zeros((2, 3, 7, 3))
    store.set("raw.X", img_ambiguous_ch)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (2, 4, 4, 3)

    img_single = np.zeros((8, 8, 3))
    store.set("raw.X", img_single)
    res = step.transform(store, rng=rng)
    assert res["raw.X"].shape == (4, 4, 3)

    img_match = np.zeros((2, 4, 4, 3))
    store.set("raw.X", img_match)
    res = step.transform(store, rng=rng)
    # With strict uint8 policy, we expect a copy if input wasn't uint8
    assert res["raw.X"].dtype == np.uint8
    assert res["raw.X"].shape == img_match.shape
    assert np.all(res["raw.X"] == img_match.astype(np.uint8))

    store.set("raw.X", np.zeros((10,)))
    res = step.transform(store, rng=rng)
    assert res["raw.X"].ndim == 1


def test_zca_whitening_step():
    from modssc.preprocess.steps.vision.zca_whitening import ZcaWhiteningStep

    step = ZcaWhiteningStep(max_features=100)
    store = ArtifactStore()
    rng = np.random.default_rng(42)

    X = np.random.randn(10, 5).astype(np.float32)
    X[:, 1] = X[:, 0] * 0.9 + 0.1
    store.set("raw.X", X)

    step.fit(store, fit_indices=np.arange(10), rng=rng)
    assert step.mean_ is not None
    assert step.W_ is not None
    assert step.orig_shape_ == (5,)

    res = step.transform(store, rng=rng)
    out = res["raw.X"]
    assert out.shape == (10, 5)

    np.cov(out.T)

    step_unfit = ZcaWhiteningStep()
    with pytest.raises(PreprocessValidationError, match="called before fit"):
        step_unfit.transform(store, rng=rng)

    store.set("raw.X", np.random.randn(10, 4))
    with pytest.raises(PreprocessValidationError, match="dimension mismatch"):
        step.transform(store, rng=rng)

    store.set("raw.X", np.zeros((10,)))
    with pytest.raises(PreprocessValidationError, match="at least 2 dimensions"):
        step.transform(store, rng=rng)

    store.set("raw.X", np.random.randn(10, 5))
    with pytest.raises(PreprocessValidationError, match="empty selection"):
        step.fit(store, fit_indices=np.array([]), rng=rng)

    step_small = ZcaWhiteningStep(max_features=2)
    store.set("raw.X", np.random.randn(10, 5))
    with pytest.raises(PreprocessValidationError, match="too large"):
        step_small.fit(store, fit_indices=np.arange(10), rng=rng)

    store.set("raw.X", np.zeros((10,)))
    with pytest.raises(PreprocessValidationError, match="at least 2 dimensions"):
        step.fit(store, fit_indices=np.arange(10), rng=rng)
