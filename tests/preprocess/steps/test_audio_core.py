from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.audio.load_waveform import (
    LoadWaveformStep,
    _as_numpy_waveform,
    _is_path_like,
    _pad_waveform,
    _trim_waveform,
)
from modssc.preprocess.steps.audio.wav2vec2 import Wav2Vec2Step
from modssc.preprocess.store import ArtifactStore


def test_wav2vec2_transform():
    """Test Wav2Vec2Step transform with mocked encoder."""
    step = Wav2Vec2Step(model_id="test-model", batch_size=2, device="cpu")
    store = ArtifactStore()
    store.set("raw.X", ["audio1", "audio2"])

    rng = np.random.default_rng(42)

    with patch("modssc.preprocess.steps.audio.wav2vec2.load_encoder") as mock_load:
        mock_encoder = MagicMock()
        mock_load.return_value = mock_encoder

        mock_encoder.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        result = step.transform(store, rng=rng)

        mock_load.assert_called_once_with("test-model", device="cpu")

        mock_encoder.encode.assert_called_once()
        call_args = mock_encoder.encode.call_args
        assert call_args[0][0] == ["audio1", "audio2"]
        assert call_args[1]["batch_size"] == 2
        assert call_args[1]["rng"] is rng

        assert "features.X" in result
        np.testing.assert_array_almost_equal(result["features.X"], [[0.1, 0.2], [0.3, 0.4]])


def test_wav2vec2_transform_no_device():
    """Test Wav2Vec2Step transform without explicit device."""
    step = Wav2Vec2Step(model_id="test-model")
    store = ArtifactStore()
    store.set("raw.X", ["audio1"])
    rng = np.random.default_rng(42)

    with patch("modssc.preprocess.steps.audio.wav2vec2.load_encoder") as mock_load:
        mock_encoder = MagicMock()
        mock_load.return_value = mock_encoder
        mock_encoder.encode.return_value = [[0.1]]

        step.transform(store, rng=rng)
        step.transform(store, rng=rng)

        mock_load.assert_called_once_with("test-model")


def test_is_path_like_variants():
    assert _is_path_like("file.wav")
    assert _is_path_like(b"file.wav")
    assert _is_path_like(bytearray(b"file.wav"))
    assert _is_path_like(Path("file.wav"))
    assert not _is_path_like(123)


def test_as_numpy_waveform_variants():
    wave = np.array([1.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(_as_numpy_waveform(wave, mono=True), wave)

    stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    mono = _as_numpy_waveform(stereo, mono=True)
    np.testing.assert_allclose(mono, np.array([1.5, 3.5], dtype=np.float32))

    mono2 = _as_numpy_waveform(stereo, mono=False)
    np.testing.assert_allclose(mono2, np.array([1.5, 3.5], dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="1D or 2D"):
        _as_numpy_waveform(np.zeros((1, 2, 3)), mono=True)


def test_trim_and_pad_waveform():
    wave = np.arange(6, dtype=np.float32)
    np.testing.assert_allclose(_trim_waveform(wave, max_length=4, trim="start"), [0, 1, 2, 3])
    np.testing.assert_allclose(_trim_waveform(wave, max_length=4, trim="center"), [1, 2, 3, 4])
    np.testing.assert_allclose(_trim_waveform(wave, max_length=4, trim="end"), [2, 3, 4, 5])

    padded = _pad_waveform(np.array([1.0, 2.0], dtype=np.float32), max_length=4, pad_value=-1)
    np.testing.assert_allclose(padded, [1.0, 2.0, -1.0, -1.0])


def test_load_waveform_transform_numpy_2d():
    store = ArtifactStore()
    store.set("raw.X", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    step = LoadWaveformStep()
    rng = np.random.default_rng(0)

    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (2, 2)


def test_load_waveform_transform_paths_with_resample(tmp_path):
    store = ArtifactStore()
    store.set("raw.X", [str(tmp_path / "a.wav")])

    class DummyFunctional:
        @staticmethod
        def resample(waveform, sr, target):
            return waveform

    class DummyTorchAudio:
        functional = DummyFunctional()

        @staticmethod
        def load(path):
            return np.zeros((1, 4), dtype=np.float32), 8000

    step = LoadWaveformStep(target_sample_rate=16000, max_length=4)
    rng = np.random.default_rng(0)

    with patch("modssc.preprocess.steps.audio.load_waveform.require", return_value=DummyTorchAudio):
        res = step.transform(store, rng=rng)

    assert res["features.X"].shape == (1, 4)


def test_load_waveform_load_path_resample(tmp_path):
    class DummyFunctional:
        @staticmethod
        def resample(waveform, sr, target):
            return waveform

    class DummyTorchAudio:
        functional = DummyFunctional()

        @staticmethod
        def load(path):
            return np.zeros((1, 2), dtype=np.float32), 8000

    step = LoadWaveformStep(target_sample_rate=16000)
    with patch("modssc.preprocess.steps.audio.load_waveform.require", return_value=DummyTorchAudio):
        wave, sr = step._load_path(tmp_path / "a.wav")
    assert wave.shape == (2,)
    assert sr == 16000


def test_load_waveform_transform_single_item_path(tmp_path):
    store = ArtifactStore()
    store.set("raw.X", str(tmp_path / "a.wav"))

    class DummyTorchAudio:
        @staticmethod
        def load(path):
            return np.zeros((1, 2), dtype=np.float32), 16000

    step = LoadWaveformStep()
    rng = np.random.default_rng(0)
    with patch("modssc.preprocess.steps.audio.load_waveform.require", return_value=DummyTorchAudio):
        res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (1, 2)


def test_load_waveform_transform_invalid_trim():
    store = ArtifactStore()
    store.set("raw.X", [np.zeros(4, dtype=np.float32)])
    step = LoadWaveformStep(trim="bad")
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match="trim must be one of"):
        step.transform(store, rng=rng)


def test_load_waveform_transform_invalid_max_length():
    store = ArtifactStore()
    store.set("raw.X", [np.zeros(4, dtype=np.float32)])
    step = LoadWaveformStep(max_length=0)
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match="max_length must be"):
        step.transform(store, rng=rng)


def test_load_waveform_transform_variable_lengths_without_max():
    store = ArtifactStore()
    store.set("raw.X", [np.zeros(2, dtype=np.float32), np.zeros(3, dtype=np.float32)])
    step = LoadWaveformStep()
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match="requires max_length"):
        step.transform(store, rng=rng)


def test_load_waveform_transform_target_sample_rate_missing():
    store = ArtifactStore()
    store.set("raw.X", [np.zeros(2, dtype=np.float32)])
    step = LoadWaveformStep(target_sample_rate=16000)
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match="target_sample_rate"):
        step.transform(store, rng=rng)


def test_load_waveform_transform_empty():
    store = ArtifactStore()
    store.set("raw.X", [])
    step = LoadWaveformStep()
    rng = np.random.default_rng(0)
    res = step.transform(store, rng=rng)
    assert res["features.X"].shape == (0, 0)


def test_load_waveform_transform_numpy_scalar_error():
    store = ArtifactStore()
    store.set("raw.X", np.array(1.0))
    step = LoadWaveformStep()
    rng = np.random.default_rng(0)
    with pytest.raises(PreprocessValidationError, match="1D or 2D"):
        step.transform(store, rng=rng)
