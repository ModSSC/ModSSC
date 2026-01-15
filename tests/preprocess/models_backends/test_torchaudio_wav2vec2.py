from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import torch
except Exception:
    torch = None

from modssc.preprocess.errors import OptionalDependencyError
from modssc.preprocess.models_backends.torchaudio_wav2vec2 import Wav2Vec2Encoder


def test_load_safe_scipy_fallback():
    # Mock torchaudio requirements during init
    with patch("modssc.preprocess.models_backends.torchaudio_wav2vec2.require") as mock_require:
        # Mock torchaudio.pipelines
        mock_torchaudio = MagicMock()
        mock_bundle = MagicMock()
        mock_bundle.get_model.return_value = MagicMock()
        mock_torchaudio.pipelines.WAV2VEC2_BASE = mock_bundle

        # When require is called, return mocks
        def side_effect(module, **kwargs):
            if module == "torch":
                return MagicMock()
            if module == "torchaudio":
                return mock_torchaudio
            return MagicMock()

        mock_require.side_effect = side_effect

        backend = Wav2Vec2Encoder()

    # Mock self._torchaudio.load to raise RuntimeError
    backend._torchaudio.load.side_effect = RuntimeError("libtorchcodec error")

    # Mock scipy
    with patch.dict("sys.modules", {"scipy.io.wavfile": MagicMock()}):
        import scipy.io.wavfile

        # Case 1: int16
        scipy.io.wavfile.read.return_value = (16000, np.array([32767, -32768], dtype=np.int16))
        out = backend._load_safe("fake.wav")
        assert out.dtype == np.float32
        # 32767 -> ~0.9999, -32768 -> -1.0
        assert np.allclose(out, np.array([[32767 / 32768], [-1.0]]).T)

        # Case 2: int32
        scipy.io.wavfile.read.return_value = (16000, np.array([2147483647], dtype=np.int32))
        out2 = backend._load_safe("fake.wav")
        assert np.allclose(out2, np.array([[2147483647 / 2147483648]]).T)

        # Case 3: uint8
        scipy.io.wavfile.read.return_value = (16000, np.array([255, 0], dtype=np.uint8))
        out3 = backend._load_safe("fake.wav")
        # 255 -> 127/128, 0 -> -128/128 = -1.0
        assert np.allclose(out3, np.array([[(255 - 128) / 128], [-1.0]]).T)

        # Case 4: float32
        scipy.io.wavfile.read.return_value = (16000, np.array([0.5], dtype=np.float32))
        out4 = backend._load_safe("fake.wav")
        assert np.allclose(out4, np.array([[0.5]]).T)

        # Case 5: scipy fails
        scipy.io.wavfile.read.side_effect = Exception("scipy failed")
        with pytest.raises(RuntimeError, match="scipy fallback"):
            backend._load_safe("fake.wav")

    # Case 6: non-libtorchcodec error
    backend._torchaudio.load.side_effect = RuntimeError("other error")
    with pytest.raises(RuntimeError, match="other error"):
        backend._load_safe("fake.wav")


def test_wav2vec2_init_errors():
    # Test OptionalDependencyError
    with patch("modssc.preprocess.models_backends.torchaudio_wav2vec2.require") as mock_require:
        mock_require.side_effect = OptionalDependencyError("torch", "preprocess-audio")
        with pytest.raises(OptionalDependencyError):
            Wav2Vec2Encoder()

    # Test invalid bundle
    with patch("modssc.preprocess.models_backends.torchaudio_wav2vec2.require") as mock_require:
        mock_torchaudio = MagicMock()
        del mock_torchaudio.pipelines.BAD_BUNDLE
        mock_require.return_value = mock_torchaudio

        # Mock torch to pass first require
        def side_effect(module, **kwargs):
            if module == "torch":
                return MagicMock()
            return mock_torchaudio

        mock_require.side_effect = side_effect

        with pytest.raises(ValueError, match="Unknown torchaudio pipeline bundle"):
            Wav2Vec2Encoder(bundle="BAD_BUNDLE")


def test_wav2vec2_encode():
    # Mock require to return mocks
    with patch("modssc.preprocess.models_backends.torchaudio_wav2vec2.require") as mock_require:
        mock_torchaudio = MagicMock()

        # Setup mock model
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        mock_param = MagicMock()
        mock_param.device = "cpu"
        # Use side_effect to return a new iterator every time parameters() is called
        mock_model.parameters.side_effect = lambda: iter([mock_param])

        def forward(x):
            return (torch.ones((x.shape[0], 5, 10)), None)

        mock_model.side_effect = forward

        mock_bundle = MagicMock()
        mock_bundle.get_model.return_value = mock_model
        mock_torchaudio.pipelines.WAV2VEC2_BASE = mock_bundle

        def side_effect(module, **kwargs):
            if module == "torch":
                return torch  # Use real torch for logic
            if module == "torchaudio":
                return mock_torchaudio
            return MagicMock()

        mock_require.side_effect = side_effect

        encoder = Wav2Vec2Encoder(device="cpu")

        X = [np.random.randn(100).astype(np.float32)]
        out = encoder.encode(X)
        assert out.shape == (1, 10)

        # Test validation error 2D > 1 channel
        with pytest.raises(ValueError, match="expects 1D waveforms"):
            encoder.encode([np.zeros((2, 100))])

        # Test 2D (1, T) ok
        out_2d_ok = encoder.encode([np.zeros((1, 100))])
        assert out_2d_ok.shape == (1, 10)

        # Test path input
        with patch.object(encoder, "_load_safe") as mock_load:
            mock_load.return_value = np.random.randn(100).astype(np.float32)
            out_path = encoder.encode(["path1"])
            assert out_path.shape == (1, 10)
            assert mock_load.called
