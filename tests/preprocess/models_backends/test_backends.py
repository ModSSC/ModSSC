from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.preprocess.errors import OptionalDependencyError


def test_encoder_protocol():
    class MyEncoder:
        def encode(self, X, *, batch_size=32, rng=None):
            return np.array([])

    assert issubclass(MyEncoder, object)


def test_sentence_transformer_missing_dep():
    with patch("modssc.preprocess.models_backends.sentence_transformers.require") as mock_require:
        mock_require.side_effect = OptionalDependencyError("missing", "purpose")
        from modssc.preprocess.models_backends.sentence_transformers import (
            SentenceTransformerEncoder,
        )

        with pytest.raises(OptionalDependencyError):
            SentenceTransformerEncoder()


def test_sentence_transformer_encode():
    mock_st_module = MagicMock()
    mock_model = MagicMock()
    mock_st_module.SentenceTransformer.return_value = mock_model

    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    with patch(
        "modssc.preprocess.models_backends.sentence_transformers.require",
        return_value=mock_st_module,
    ):
        from modssc.preprocess.models_backends.sentence_transformers import (
            SentenceTransformerEncoder,
        )

        encoder = SentenceTransformerEncoder(model_name="test-model")

        mock_st_module.SentenceTransformer.assert_called_with("test-model", device=None)

        texts = ["hello", "world"]
        res = encoder.encode(texts, batch_size=2)

        assert res.shape == (2, 2)
        assert res.dtype == np.float32
        mock_model.encode.assert_called_with(
            texts,
            batch_size=2,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )


def test_open_clip_missing_dep():
    with patch("modssc.preprocess.models_backends.open_clip.require") as mock_require:
        mock_require.side_effect = OptionalDependencyError("missing", "purpose")
        from modssc.preprocess.models_backends.open_clip import OpenClipEncoder

        with pytest.raises(OptionalDependencyError):
            OpenClipEncoder()


def test_open_clip_encode():
    mock_open_clip = MagicMock()
    mock_torch = MagicMock()

    mock_model = MagicMock()

    mock_model.to.return_value = mock_model

    mock_preprocess = MagicMock(return_value="processed_image")
    mock_open_clip.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)

    mock_tensor = MagicMock()
    mock_torch.stack.return_value = mock_tensor
    mock_tensor.to.return_value = mock_tensor

    mock_emb1 = MagicMock()
    mock_emb1.cpu.return_value = mock_emb1
    mock_emb1.numpy.return_value = np.array([[0.1]], dtype=np.float32)

    mock_emb2 = MagicMock()
    mock_emb2.cpu.return_value = mock_emb2
    mock_emb2.numpy.return_value = np.array([[0.1], [0.2]], dtype=np.float32)

    with (
        patch.dict("sys.modules", {"PIL": MagicMock(), "PIL.Image": MagicMock()}),
        patch("modssc.preprocess.models_backends.open_clip.require") as mock_require,
    ):

        def require_side_effect(module, **kwargs):
            if module == "open_clip":
                return mock_open_clip
            if module == "torch":
                return mock_torch
            return MagicMock()

        mock_require.side_effect = require_side_effect

        from modssc.preprocess.models_backends.open_clip import OpenClipEncoder

        encoder = OpenClipEncoder()

        mock_model.encode_image.return_value = mock_emb1
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        res = encoder.encode(img)
        assert res.shape == (1, 1)

        mock_model.encode_image.return_value = mock_emb2
        batch = np.zeros((2, 10, 10, 3), dtype=np.uint8)
        res = encoder.encode(batch)
        assert res.shape == (2, 1)

        mock_model.encode_image.return_value = mock_emb2
        res = encoder.encode([img, img])
        assert res.shape == (2, 1)

        mock_model.encode_image.return_value = mock_emb1
        img_chw = np.zeros((3, 10, 10), dtype=np.uint8)
        res = encoder.encode(img_chw)
        assert res.shape == (1, 1)

        mock_model.encode_image.return_value = mock_emb1
        img_chw_1 = np.zeros((1, 10, 10), dtype=np.uint8)
        res = encoder.encode(img_chw_1)
        assert res.shape == (1, 1)

        mock_model.encode_image.return_value = mock_emb2
        res = encoder.encode([img_chw, img_chw])
        assert res.shape == (2, 1)

        mock_model.encode_image.return_value = mock_emb1
        img_float = np.zeros((10, 10, 3), dtype=np.float32)
        res = encoder.encode(img_float)
        assert res.shape == (1, 1)

        mock_model.encode_image.return_value = mock_emb1
        gen = (img for _ in range(1))
        res = encoder.encode(gen)
        assert res.shape == (1, 1)

        res = encoder.encode([])
        assert res.shape == (0, 0)

        chw = np.zeros((3, 10, 10), dtype=np.uint8)
        encoder.encode(chw)


def test_wav2vec2_missing_dep():
    with patch("modssc.preprocess.models_backends.torchaudio_wav2vec2.require") as mock_require:
        mock_require.side_effect = OptionalDependencyError("missing", "purpose")
        from modssc.preprocess.models_backends.torchaudio_wav2vec2 import Wav2Vec2Encoder

        with pytest.raises(OptionalDependencyError):
            Wav2Vec2Encoder()


def test_wav2vec2_unknown_bundle():
    mock_torch = MagicMock()
    mock_torchaudio = MagicMock()

    mock_torchaudio.pipelines = MagicMock(spec=[])

    with patch("modssc.preprocess.models_backends.torchaudio_wav2vec2.require") as mock_require:

        def require_side_effect(module, **kwargs):
            if module == "torch":
                return mock_torch
            if module == "torchaudio":
                return mock_torchaudio
            return MagicMock()

        mock_require.side_effect = require_side_effect

        from modssc.preprocess.models_backends.torchaudio_wav2vec2 import Wav2Vec2Encoder

        with pytest.raises(ValueError, match="Unknown torchaudio pipeline bundle"):
            Wav2Vec2Encoder(bundle="UNKNOWN")


def test_wav2vec2_encode():
    mock_torch = MagicMock()
    mock_torchaudio = MagicMock()

    mock_bundle = MagicMock()
    mock_model = MagicMock()

    mock_model.to.return_value = mock_model

    mock_bundle.get_model.return_value = mock_model
    mock_torchaudio.pipelines.WAV2VEC2_BASE = mock_bundle

    mock_tensor = MagicMock()
    mock_torch.from_numpy.return_value = mock_tensor
    mock_tensor.unsqueeze.return_value = mock_tensor
    mock_tensor.to.return_value = mock_tensor

    mock_feat = MagicMock()
    mock_model.return_value = (mock_feat, None)
    mock_feat.mean.return_value = mock_feat
    mock_feat.cpu.return_value = mock_feat
    mock_feat.numpy.return_value = np.array([[0.5]], dtype=np.float32)

    mock_torchaudio.load.return_value = (MagicMock(numpy=lambda: np.zeros(100)), 16000)

    with patch("modssc.preprocess.models_backends.torchaudio_wav2vec2.require") as mock_require:

        def require_side_effect(module, **kwargs):
            if module == "torch":
                return mock_torch
            if module == "torchaudio":
                return mock_torchaudio
            return MagicMock()

        mock_require.side_effect = require_side_effect

        from modssc.preprocess.models_backends.torchaudio_wav2vec2 import Wav2Vec2Encoder

        encoder = Wav2Vec2Encoder()

        res = encoder.encode(["/tmp/fake.wav"])
        assert res.shape == (1, 1)
        mock_torchaudio.load.assert_called()

        arr = np.zeros(100, dtype=np.float32)
        res = encoder.encode([arr])
        assert res.shape == (1, 1)

        arr2d = np.zeros((1, 100), dtype=np.float32)
        res = encoder.encode([arr2d])
        assert res.shape == (1, 1)

        with pytest.raises(ValueError, match="wav2vec2 expects 1D waveforms"):
            encoder.encode([np.zeros((2, 100))])

        res = encoder.encode([])
        assert res.shape == (0, 0)


def test_base_encoder_protocol():
    from modssc.preprocess.models_backends.base import Encoder

    assert isinstance(Encoder, type)
