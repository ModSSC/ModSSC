import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import modssc.cache.model as model_cache
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

    with (
        patch.dict(os.environ, {}, clear=True),
        patch(
            "modssc.preprocess.models_backends.sentence_transformers.require",
            return_value=mock_st_module,
        ),
    ):
        from modssc.preprocess.models_backends.sentence_transformers import (
            SentenceTransformerEncoder,
        )

        encoder = SentenceTransformerEncoder(model_name="test-model")

        mock_st_module.SentenceTransformer.assert_called_with(
            "test-model",
            device=None,
            cache_folder=None,
            local_files_only=False,
        )

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


def test_sentence_transformer_respects_cache_and_offline_env():
    mock_st_module = MagicMock()
    mock_model = MagicMock()
    mock_st_module.SentenceTransformer.return_value = mock_model

    with (
        patch.dict(
            os.environ,
            {
                "SENTENCE_TRANSFORMERS_HOME": "/tmp/st-cache",
                "HF_HUB_OFFLINE": "1",
            },
            clear=True,
        ),
        patch(
            "modssc.preprocess.models_backends.sentence_transformers.require",
            return_value=mock_st_module,
        ),
    ):
        from modssc.preprocess.models_backends.sentence_transformers import (
            SentenceTransformerEncoder,
        )

        SentenceTransformerEncoder(model_name="test-model", device="cpu")

        mock_st_module.SentenceTransformer.assert_called_with(
            "test-model",
            device="cpu",
            cache_folder=str(Path("/tmp/st-cache").resolve()),
            local_files_only=True,
        )


def test_sentence_transformer_wraps_model_load_error():
    mock_st_module = MagicMock()
    mock_st_module.SentenceTransformer.side_effect = RuntimeError("offline")

    with (
        patch.dict(os.environ, {}, clear=True),
        patch(
            "modssc.preprocess.models_backends.sentence_transformers.require",
            return_value=mock_st_module,
        ),
    ):
        from modssc.preprocess.models_backends.sentence_transformers import (
            SentenceTransformerEncoder,
        )

        with pytest.raises(RuntimeError, match="Failed to load SentenceTransformer model"):
            SentenceTransformerEncoder(model_name="broken-model")


def test_open_clip_missing_dep():
    with patch("modssc.preprocess.models_backends.open_clip.require") as mock_require:
        mock_require.side_effect = OptionalDependencyError("missing", "purpose")
        from modssc.preprocess.models_backends.open_clip import OpenClipEncoder

        with pytest.raises(OptionalDependencyError):
            OpenClipEncoder()


def test_open_clip_wraps_model_load_error():
    mock_open_clip = MagicMock()
    mock_open_clip.create_model_and_transforms.side_effect = RuntimeError("offline")

    with patch("modssc.preprocess.models_backends.open_clip.require") as mock_require:

        def require_side_effect(module, **kwargs):
            del kwargs
            if module == "open_clip":
                return mock_open_clip
            if module == "torch":
                return MagicMock()
            return MagicMock()

        mock_require.side_effect = require_side_effect

        from modssc.preprocess.models_backends.open_clip import OpenClipEncoder

        with pytest.raises(RuntimeError, match="Failed to load OpenCLIP model"):
            OpenClipEncoder(model_name="broken-model")


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
        mock_open_clip.create_model_and_transforms.assert_called_with(
            "ViT-B-32",
            pretrained="openai",
            cache_dir=None,
        )

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


def test_open_clip_respects_cache_env():
    mock_open_clip = MagicMock()
    mock_torch = MagicMock()
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_open_clip.create_model_and_transforms.return_value = (mock_model, None, MagicMock())

    with (
        patch.dict(os.environ, {"MODSSC_OPENCLIP_CACHE_DIR": "/tmp/openclip"}, clear=True),
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

        OpenClipEncoder()

        mock_open_clip.create_model_and_transforms.assert_called_with(
            "ViT-B-32",
            pretrained="openai",
            cache_dir=str(Path("/tmp/openclip").resolve()),
        )


def test_model_cache_resolves_from_modssc_cache_root():
    with patch.dict(os.environ, {"MODSSC_CACHE_ROOT": "/tmp/modssc-cache"}, clear=True):
        root = Path("/tmp/modssc-cache").resolve() / "models"
        assert model_cache.resolve_model_cache_root() == root
        assert model_cache.resolve_hf_home() == root / "hf"
        assert model_cache.resolve_sentence_transformers_cache() == str(
            root / "hf" / "sentence_transformers"
        )
        assert model_cache.resolve_openclip_cache_dir() == str(root / "open_clip")


def test_model_cache_prefers_hf_and_transformers_envs():
    with patch.dict(os.environ, {"HF_HOME": "/tmp/hf-home"}, clear=True):
        assert model_cache.resolve_hf_home() == Path("/tmp/hf-home").resolve()

    with patch.dict(os.environ, {"TRANSFORMERS_CACHE": "/tmp/transformers-cache"}, clear=True):
        assert model_cache.resolve_sentence_transformers_cache() == str(
            Path("/tmp/transformers-cache").resolve()
        )


def test_torchvision_image_encode(monkeypatch):
    torch = pytest.importorskip("torch")
    from modssc.preprocess.models_backends import torchvision_image as backend

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    def fake_require(module, **kwargs):
        del kwargs
        if module == "torch":
            return torch
        return MagicMock()

    monkeypatch.setattr(backend, "require", fake_require)
    monkeypatch.setattr(
        backend.image_pretrained_backend, "_load_model", lambda *_a, **_k: DummyModel()
    )
    monkeypatch.setattr(backend.image_pretrained_backend, "_infer_in_channels", lambda *_a, **_k: 3)

    encoder = backend.TorchvisionImageEncoder()

    chw = np.zeros((2, 3, 8, 8), dtype=np.float32)
    res_chw = encoder.encode(chw, batch_size=1)
    assert res_chw.shape == (2, 4)
    assert res_chw.dtype == np.float32

    hwc = np.zeros((2, 8, 8, 3), dtype=np.float32)
    res_hwc = encoder.encode(hwc, batch_size=2)
    assert res_hwc.shape == (2, 4)

    batched_gray = np.zeros((5, 8, 8), dtype=np.float32)
    res_batched_gray = encoder.encode(batched_gray, batch_size=2)
    assert res_batched_gray.shape == (5, 4)

    gray = np.zeros((2, 8, 8, 1), dtype=np.float32)
    res_gray = encoder.encode(gray, batch_size=2)
    assert res_gray.shape == (2, 4)

    single_chw = np.zeros((3, 8, 8), dtype=np.float32)
    res_single_chw = encoder.encode(single_chw, batch_size=1)
    assert res_single_chw.shape == (1, 4)

    gray_list = [np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32)]
    res_list = encoder.encode(gray_list, batch_size=2)
    assert res_list.shape == (2, 4)

    gray_gen = (np.zeros((8, 8), dtype=np.float32) for _ in range(1))
    res_gen = encoder.encode(gray_gen, batch_size=1)
    assert res_gen.shape == (1, 4)

    res_empty = encoder.encode([])
    assert res_empty.shape == (0, 0)


def test_torchvision_image_encode_with_unknown_input_channels(monkeypatch):
    torch = pytest.importorskip("torch")
    from modssc.preprocess.models_backends import torchvision_image as backend

    class StemlessModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(1, 2)

        def forward(self, x):
            x = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
            return self.head(x)

    def fake_require(module, **kwargs):
        del kwargs
        if module == "torch":
            return torch
        return MagicMock()

    monkeypatch.setattr(backend, "require", fake_require)
    monkeypatch.setattr(
        backend.image_pretrained_backend, "_load_model", lambda *_a, **_k: StemlessModel()
    )
    monkeypatch.setattr(
        backend.image_pretrained_backend, "_infer_in_channels", lambda *_a, **_k: None
    )

    encoder = backend.TorchvisionImageEncoder()

    res_single_chw = encoder.encode(np.zeros((3, 8, 8), dtype=np.float32), batch_size=1)
    assert res_single_chw.shape == (1, 1)

    res_single_hwc = encoder.encode(np.zeros((8, 8, 3), dtype=np.float32), batch_size=1)
    assert res_single_hwc.shape == (1, 1)

    res_batched_gray = encoder.encode(np.zeros((5, 8, 8), dtype=np.float32), batch_size=2)
    assert res_batched_gray.shape == (5, 1)


def test_torchvision_image_helpers(monkeypatch):
    torch = pytest.importorskip("torch")
    from modssc.preprocess.errors import PreprocessValidationError
    from modssc.preprocess.models_backends import torchvision_image as backend

    class WithClassifierLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(4, 2)

    model = WithClassifierLinear()
    assert backend._resolve_feature_module(model, torch) is model.classifier

    class WithClassifierSequential(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(4, 2))

    model = WithClassifierSequential()
    assert backend._resolve_feature_module(model, torch) is model.classifier[1]

    class WithClassifierSequentialNoLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.ReLU())
            self.head = torch.nn.Linear(4, 2)

    model = WithClassifierSequentialNoLinear()
    assert backend._resolve_feature_module(model, torch) is model.head

    class WithEmptyClassifierSequential(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Sequential()
            self.head = torch.nn.Linear(4, 2)

    model = WithEmptyClassifierSequential()
    assert backend._resolve_feature_module(model, torch) is model.head

    class WithClassifierOther(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.ReLU()
            self.head = torch.nn.Linear(4, 2)

    model = WithClassifierOther()
    assert backend._resolve_feature_module(model, torch) is model.head

    class HeadsContainer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 2)

    class WithHeadsHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = HeadsContainer()

    model = WithHeadsHead()
    assert backend._resolve_feature_module(model, torch) is model.heads.head

    class WithHeadsLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = torch.nn.Linear(4, 2)

    model = WithHeadsLinear()
    assert backend._resolve_feature_module(model, torch) is model.heads

    class WithHeadsOther(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = torch.nn.Sequential(torch.nn.ReLU())
            self.head = torch.nn.Linear(4, 2)

    model = WithHeadsOther()
    assert backend._resolve_feature_module(model, torch) is model.head

    class WithHeadLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 2)

    model = WithHeadLinear()
    assert backend._resolve_feature_module(model, torch) is model.head

    with pytest.raises(PreprocessValidationError, match="Unable to resolve classifier head"):
        backend._resolve_feature_module(torch.nn.Identity(), torch)

    arr_2d = np.zeros((6, 7), dtype=np.float32)
    assert backend._to_nchw(arr_2d).shape == (1, 6, 7)

    arr_chw = np.zeros((3, 6, 7), dtype=np.float32)
    np.testing.assert_array_equal(backend._to_nchw(arr_chw), arr_chw)

    arr_hwc = np.zeros((6, 7, 3), dtype=np.float32)
    assert backend._to_nchw(arr_hwc).shape == (3, 6, 7)

    assert backend._is_single_image_3d(np.zeros((3, 6, 7), dtype=np.float32)) is True
    assert backend._is_single_image_3d(np.zeros((6, 7, 3), dtype=np.float32)) is True
    assert backend._is_single_image_3d(np.zeros((5, 6, 7), dtype=np.float32)) is False
    assert backend._is_single_image_3d(np.zeros((6, 7), dtype=np.float32)) is False

    with pytest.raises(PreprocessValidationError, match="expects 2D/3D images"):
        backend._to_nchw(np.zeros((1, 2, 3, 4), dtype=np.float32))

    with pytest.raises(PreprocessValidationError, match="Could not infer image channel layout"):
        backend._to_nchw(np.zeros((2, 6, 7), dtype=np.float32))

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    def fake_require(module, **kwargs):
        del kwargs
        if module == "torch":
            return torch
        return MagicMock()

    monkeypatch.setattr(backend, "require", fake_require)
    monkeypatch.setattr(
        backend.image_pretrained_backend, "_load_model", lambda *_a, **_k: DummyModel()
    )
    monkeypatch.setattr(backend.image_pretrained_backend, "_infer_in_channels", lambda *_a, **_k: 3)

    encoder = backend.TorchvisionImageEncoder(auto_channel_repeat=False)
    split_2d = encoder._split_samples(np.zeros((8, 8), dtype=np.float32))
    assert len(split_2d) == 1

    split_hwc = encoder._split_samples(np.zeros((8, 8, 3), dtype=np.float32))
    assert len(split_hwc) == 1

    split_other = encoder._split_samples(np.zeros((1, 2, 3, 4, 5), dtype=np.float32))
    assert len(split_other) == 1

    with pytest.raises(PreprocessValidationError, match="Model expects 3 channels, got 1"):
        encoder.encode([np.zeros((8, 8), dtype=np.float32)])


def test_torchvision_image_encode_batch_edge_cases(monkeypatch):
    torch = pytest.importorskip("torch")
    from modssc.preprocess.errors import PreprocessValidationError
    from modssc.preprocess.models_backends import torchvision_image as backend

    class SpatialHeadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x).permute(0, 2, 3, 1)
            return self.fc(x).mean(dim=(1, 2))

    def fake_require(module, **kwargs):
        del kwargs
        if module == "torch":
            return torch
        return MagicMock()

    monkeypatch.setattr(backend, "require", fake_require)
    monkeypatch.setattr(
        backend.image_pretrained_backend, "_load_model", lambda *_a, **_k: SpatialHeadModel()
    )
    monkeypatch.setattr(backend.image_pretrained_backend, "_infer_in_channels", lambda *_a, **_k: 3)

    encoder = backend.TorchvisionImageEncoder()
    spatial_feats = encoder.encode(np.zeros((1, 3, 8, 8), dtype=np.float32), batch_size=1)
    assert spatial_feats.shape == (1, 8 * 8 * 4)

    batch = encoder._prepare_batch([np.zeros((8, 8, 3), dtype=np.float32)])

    class HookHandle:
        def __init__(self):
            self.removed = False

        def remove(self):
            self.removed = True

    class HookModule:
        def __init__(self):
            self.hook = None
            self.handle = HookHandle()

        def register_forward_hook(self, hook):
            self.hook = hook
            return self.handle

    hook_module = HookModule()
    encoder._feature_module = hook_module
    encoder._model = lambda _batch: hook_module.hook(None, (), None)

    with pytest.raises(PreprocessValidationError, match="failed to capture penultimate features"):
        encoder._encode_batch(batch)
    assert hook_module.handle.removed is True

    silent_module = HookModule()
    encoder._feature_module = silent_module
    encoder._model = lambda _batch: None

    with pytest.raises(PreprocessValidationError, match="failed to capture penultimate features"):
        encoder._encode_batch(batch)
    assert silent_module.handle.removed is True


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
