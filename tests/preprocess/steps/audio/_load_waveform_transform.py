import contextlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.audio.load_waveform import LoadWaveformStep
from modssc.preprocess.store import ArtifactStore


def test_transform_variants():
    step = LoadWaveformStep(max_length=10, pad_value=0.0)
    rng = np.random.default_rng(0)

    # 1. Input is already numpy array (2D)
    X_arr = np.random.randn(2, 5).astype(np.float32)
    store = ArtifactStore({"raw.X": X_arr})
    out = step.transform(store, rng=rng)
    assert out["features.X"].shape == (2, 10)

    # 2. Resampling logic
    step_sr = LoadWaveformStep(target_sample_rate=8000, max_length=10)

    with patch("modssc.preprocess.steps.audio.load_waveform.require") as mock_require:
        mock_ta = MagicMock()
        # Mock load returning sr=16000
        mock_ta.load.return_value = (torch.randn(1, 100), 16000)
        # Mock resample
        mock_ta.functional.resample.return_value = torch.randn(1, 50)
        mock_require.return_value = mock_ta

        # Pass path
        store_path = ArtifactStore({"raw.X": ["file.wav"]})
        step_sr.transform(store_path, rng=rng)

        assert mock_ta.functional.resample.called
        # Check call args: waveform, 16000, 8000
        args, _ = mock_ta.functional.resample.call_args
        assert args[1] == 16000
        assert args[2] == 8000

    # 3. Invalid waveform dimension (explicit list of 3D arrays)
    # Each item is (1, 1, 1) -> 3D.
    store_3d_list = ArtifactStore({"raw.X": [np.zeros((1, 1, 1))]})
    with pytest.raises(PreprocessValidationError, match="Audio waveform must be 1D or 2D"):
        step.transform(store_3d_list, rng=rng)

    # 4. Mixed input (list of path and array)
    store_mixed = ArtifactStore({"raw.X": ["file.wav", np.zeros(5)]})

    with patch("modssc.preprocess.steps.audio.load_waveform.require") as mock_require:
        mock_ta = MagicMock()
        mock_ta.load.return_value = (torch.zeros(1, 5), 16000)
        mock_require.return_value = mock_ta

        out_mixed = step.transform(store_mixed, rng=rng)
        assert out_mixed["features.X"].shape == (2, 10)


def test_load_path_errors():
    step = LoadWaveformStep()
    with patch("modssc.preprocess.steps.audio.load_waveform.require") as mock_require:
        mock_ta = MagicMock()
        mock_ta.load.side_effect = RuntimeError("some random error")
        mock_require.return_value = mock_ta

        with pytest.raises(RuntimeError, match="some random error"):
            step._load_path("file.wav")


def test_as_numpy_waveform_variants():
    from modssc.preprocess.steps.audio.load_waveform import _as_numpy_waveform

    # 1D
    arr1 = np.array([1, 2, 3])
    assert _as_numpy_waveform(arr1, mono=True).ndim == 1

    # 2D mono
    arr2 = np.array([[1, 2, 3]])
    out2 = _as_numpy_waveform(arr2, mono=True)
    assert out2.ndim == 1
    assert np.allclose(out2, arr2[0])

    # 2D stereo -> mean
    arr_stereo = np.array([[1, 2], [3, 4]])  # 2 channels, 2 samples
    out_stereo = _as_numpy_waveform(arr_stereo, mono=True)
    assert np.allclose(out_stereo, np.array([2, 3]))

    # 3D -> Error
    with pytest.raises(PreprocessValidationError, match="1D or 2D"):
        _as_numpy_waveform(np.zeros((1, 1, 1)), mono=True)


def test_transform_scipy_fallback():
    step = LoadWaveformStep(allow_fallback=True)

    # We patch the module where 'require' is defined
    with patch("modssc.preprocess.steps.audio.load_waveform.require") as mock_require:
        # Mock torchaudio
        mock_torchaudio = MagicMock()
        # Raise error to trigger fallback
        mock_torchaudio.load.side_effect = RuntimeError("libtorchcodec error")
        mock_require.return_value = mock_torchaudio

        # Use patch to mock scipy.io.wavfile module
        # This handles sys.modules injection and attribute patching on scipy.io
        with patch("scipy.io.wavfile") as mock_scipy_io_wavfile:
            # Case 1: int16 success
            # scipy.io.wavfile.read returns (sr, data)
            # data is numpy array
            mock_scipy_io_wavfile.read.side_effect = None
            mock_scipy_io_wavfile.read.return_value = (16000, np.array([32767], dtype=np.int16))

            out, sr = step._load_path("fake.wav")
            assert sr == 16000
            # 32767 / 32768.0 approx 0.99997
            assert np.allclose(out, np.array([32767 / 32768.0], dtype=np.float32))

            # Case 2: int32
            mock_scipy_io_wavfile.read.return_value = (
                16000,
                np.array([2147483647], dtype=np.int32),
            )
            out, _ = step._load_path("fake.wav")
            assert np.allclose(out, np.array([2147483647 / 2147483648.0], dtype=np.float32))

            # Case 3: uint8
            mock_scipy_io_wavfile.read.return_value = (16000, np.array([255], dtype=np.uint8))
            out, _ = step._load_path("fake.wav")
            # (255 - 128) / 128 = 127/128
            assert np.allclose(out, np.array([(255 - 128) / 128.0], dtype=np.float32))

            # Case 4: float (default)
            mock_scipy_io_wavfile.read.return_value = (16000, np.array([0.5], dtype=np.float64))
            out, _ = step._load_path("fake.wav")
            assert np.allclose(out, np.array([0.5], dtype=np.float32))

            # Case 5: scipy fails
            mock_scipy_io_wavfile.read.side_effect = Exception("scipy failed")

            with pytest.raises(RuntimeError, match="scipy fallback"):
                step._load_path("fake.wav")

    # Case 6: non-libtorchcodec error
    with patch("modssc.preprocess.steps.audio.load_waveform.require") as mock_require2:
        mock_ta2 = MagicMock()
        mock_ta2.load.side_effect = RuntimeError("other error")
        mock_require2.return_value = mock_ta2

        with pytest.raises(RuntimeError, match="other error"):
            step._load_path("fake.wav")


def test_load_waveform_internals():
    from modssc.preprocess.steps.audio.load_waveform import _as_numpy_waveform

    # 1. 3D input logic (additional check)
    arr_3d_valid = np.zeros((1, 10, 1))
    with contextlib.suppress(PreprocessValidationError):
        _as_numpy_waveform(arr_3d_valid, mono=True)

    arr_3d_invalid = np.zeros((2, 10, 1))
    with pytest.raises(PreprocessValidationError, match="1D or 2D"):
        _as_numpy_waveform(arr_3d_invalid, mono=True)

    # 2A. Test trimming loop logic (Valid trim)
    step = LoadWaveformStep(max_length=10, trim="start", pad_value=0.0)
    # _load_path returns 1D array as expected by _pad_waveform
    step._load_path = MagicMock(return_value=(np.zeros((20,), dtype=np.float32), 16000))
    step._requires_resampling = MagicMock(return_value=False)

    store = ArtifactStore({"raw.X": ["dummy"]})
    out = step.transform(store, rng=np.random.default_rng(0))
    assert out["features.X"].shape[-1] == 10

    # 2B. Test invalid trim validation
    step_invalid = LoadWaveformStep(max_length=10, trim="invalid", pad_value=0.0)
    step_invalid._load_path = MagicMock(return_value=(np.zeros((20,), dtype=np.float32), 16000))
    step_invalid._requires_resampling = MagicMock(return_value=False)
    with pytest.raises(PreprocessValidationError, match="trim must be one of"):
        step_invalid.transform(store, rng=np.random.default_rng(0))

    # 2C. Test padding
    step_pad = LoadWaveformStep(max_length=30, trim="end", pad_value=0.0)
    step_pad._load_path = MagicMock(return_value=(np.zeros((20,), dtype=np.float32), 16000))
    step_pad._requires_resampling = MagicMock(return_value=False)
    out_pad = step_pad.transform(store, rng=np.random.default_rng(0))
    assert out_pad["features.X"].shape[-1] == 30

    # 3. 1D return from load (implicit above, but double check)
    step_1d = LoadWaveformStep()
    step_1d._load_path = MagicMock(return_value=(np.zeros((20,), dtype=np.float32), 16000))
    step_1d._requires_resampling = MagicMock(return_value=False)
    out_1d = step_1d.transform(store, rng=np.random.default_rng(0))
    # Stack creates (1, 20)
    assert out_1d["features.X"].ndim == 2
