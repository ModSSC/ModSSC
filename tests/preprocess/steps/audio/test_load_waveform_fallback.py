from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.audio.load_waveform import LoadWaveformStep


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


def test_transform_scipy_fallback_requires_opt_in():
    step = LoadWaveformStep(allow_fallback=False)
    with patch("modssc.preprocess.steps.audio.load_waveform.require") as mock_require:
        mock_torchaudio = MagicMock()
        mock_torchaudio.load.side_effect = RuntimeError("libtorchcodec error")
        mock_require.return_value = mock_torchaudio

        with pytest.raises(PreprocessValidationError, match="allow_fallback"):
            step._load_path("fake.wav")
