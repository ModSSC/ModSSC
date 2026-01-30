from __future__ import annotations

import sys
import types

import pytest
import torch

from modssc.inductive.deep import bundles
from modssc.inductive.deep.types import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError


def test_normalize_hidden_sizes_and_activation() -> None:
    assert bundles._normalize_hidden_sizes(None, None, default=(4, 2)) == (4, 2)
    assert bundles._normalize_hidden_sizes(None, 3, default=(4, 2)) == (3,)
    assert bundles._normalize_hidden_sizes(5, None, default=(4, 2)) == (5,)
    assert bundles._normalize_hidden_sizes([2, 1], None, default=(4, 2)) == (2, 1)
    with pytest.raises(InductiveValidationError, match="hidden_sizes must be"):
        bundles._normalize_hidden_sizes("bad", None, default=(4, 2))

    assert isinstance(bundles._make_activation("relu", torch), torch.nn.ReLU)
    assert isinstance(bundles._make_activation("gelu", torch), torch.nn.GELU)
    assert isinstance(bundles._make_activation("tanh", torch), torch.nn.Tanh)
    with pytest.raises(InductiveValidationError, match="Unknown activation"):
        bundles._make_activation("bad", torch)


def test_infer_input_dim_and_take_sample() -> None:
    with pytest.raises(InductiveValidationError, match="torch.Tensor"):
        bundles._infer_input_dim([1, 2, 3])
    with pytest.raises(InductiveValidationError, match="at least one dimension"):
        bundles._infer_input_dim(torch.tensor(1.0))

    assert bundles._infer_input_dim(torch.randn(4)) == 4
    assert bundles._infer_input_dim(torch.randn(2, 5)) == 5

    x = torch.randn(2, 3)
    sample = bundles._take_sample(x)
    assert sample.shape[0] == 1

    x1 = torch.randn(1, 3)
    assert bundles._take_sample(x1).shape == x1.shape

    obj = {"x": 1}
    assert bundles._take_sample(obj) is obj


def test_maybe_ema() -> None:
    model = torch.nn.Linear(2, 2)
    ema = bundles._maybe_ema(model, enabled=True)
    assert ema is not None
    assert all(not p.requires_grad for p in ema.parameters())
    assert bundles._maybe_ema(model, enabled=False) is None


def test_build_mlp_and_logreg_bundles() -> None:
    sample = torch.randn(3, 4)
    bundle = bundles._build_mlp_bundle(
        sample,
        num_classes=3,
        params={"hidden_sizes": (4,), "dropout": 0.2},
        seed=0,
        ema=True,
    )
    assert isinstance(bundle, TorchModelBundle)
    assert bundle.ema_model is not None
    no_dropout = bundles._build_mlp_bundle(
        sample,
        num_classes=2,
        params={"hidden_sizes": (4,), "dropout": 0.0},
        seed=0,
        ema=False,
    )
    assert isinstance(no_dropout, TorchModelBundle)

    with pytest.raises(InductiveValidationError, match="hidden_sizes must be positive"):
        bundles._build_mlp_bundle(
            sample,
            num_classes=2,
            params={"hidden_sizes": (-1,)},
            seed=0,
            ema=False,
        )

    logreg = bundles._build_mlp_bundle(
        sample,
        num_classes=2,
        params={},
        seed=0,
        ema=False,
        force_hidden_sizes=(),
    )
    assert isinstance(logreg.model, torch.nn.Sequential)
    assert len(list(logreg.model)) == 1
    assert isinstance(logreg.model[0], torch.nn.Linear)


def test_build_mlp_feature_bundle() -> None:
    sample = torch.randn(2, 3)
    bundle = bundles._build_mlp_feature_bundle(
        sample,
        num_classes=2,
        params={"hidden_sizes": (4,), "dropout": 0.0},
        seed=0,
        ema=False,
    )
    out = bundle.model(sample)
    assert set(out.keys()) == {"logits", "feat"}
    assert out["logits"].shape[0] == int(sample.shape[0])
    assert out["feat"].shape[0] == int(sample.shape[0])

    bundle_dropout = bundles._build_mlp_feature_bundle(
        sample,
        num_classes=2,
        params={"hidden_sizes": (4,), "dropout": 0.2},
        seed=0,
        ema=False,
    )
    assert any(isinstance(layer, torch.nn.Dropout) for layer in bundle_dropout.model.backbone)

    with pytest.raises(InductiveValidationError, match="hidden_sizes must be positive"):
        bundles._build_mlp_feature_bundle(
            sample,
            num_classes=2,
            params={"hidden_sizes": (-1,)},
            seed=0,
            ema=False,
        )


def test_image_audio_text_helpers_and_bundles() -> None:
    sample4 = torch.randn(2, 3, 4, 4)
    assert bundles._infer_image_shape(sample4, input_shape=None) == (3, 4, 4)
    sample3 = torch.randn(2, 4, 4)
    assert bundles._infer_image_shape(sample3, input_shape=None) == (1, 4, 4)
    with pytest.raises(InductiveValidationError, match="input_shape"):
        bundles._infer_image_shape(torch.randn(2, 12), input_shape=None)
    with pytest.raises(InductiveValidationError, match="input_shape must be"):
        bundles._infer_image_shape(torch.randn(2, 12), input_shape=(1, 2, 3, 4))
    assert bundles._infer_image_shape(torch.randn(2, 12), input_shape=(3, 4)) == (1, 3, 4)
    with pytest.raises(InductiveValidationError, match="requires 2D, 3D, or 4D"):
        bundles._infer_image_shape(torch.randn(1, 1, 1, 1, 1), input_shape=None)
    with pytest.raises(InductiveValidationError, match="requires torch.Tensor"):
        bundles._infer_image_shape([1, 2, 3], input_shape=None)

    img_bundle = bundles._build_image_cnn_bundle(
        sample4,
        num_classes=2,
        params={"conv_channels": (4,), "dropout": 0.1},
        seed=0,
        ema=False,
    )
    assert isinstance(img_bundle, TorchModelBundle)
    logits_img, feat_img = img_bundle.model(sample4)
    assert logits_img.shape[0] == 2
    assert feat_img.ndim == 2

    audio3 = torch.randn(2, 1, 8)
    assert bundles._infer_audio_channels(audio3, input_shape=None) == 1
    audio2 = torch.randn(2, 8)
    assert bundles._infer_audio_channels(audio2, input_shape=(2, 8)) == 2
    assert bundles._infer_audio_channels(audio2, input_shape=None) == 1
    with pytest.raises(InductiveValidationError, match="audio_cnn requires"):
        bundles._infer_audio_channels(torch.randn(1, 1, 1, 1), input_shape=None)
    with pytest.raises(InductiveValidationError, match="requires torch.Tensor"):
        bundles._infer_audio_channels([1, 2, 3], input_shape=None)

    audio_bundle = bundles._build_audio_cnn_bundle(
        audio3,
        num_classes=2,
        params={"conv_channels": (4,), "dropout": 0.1},
        seed=0,
        ema=False,
    )
    assert isinstance(audio_bundle, TorchModelBundle)

    text3 = torch.randn(2, 5, 4)
    assert bundles._infer_text_shape(text3, input_layout="channels_last") == (4, 5)
    assert bundles._infer_text_shape(text3, input_layout="channels_first") == (5, 4)
    with pytest.raises(InductiveValidationError, match="input_layout must be"):
        bundles._infer_text_shape(text3, input_layout="bad")
    assert bundles._infer_text_shape(torch.randn(2, 6), input_layout="channels_last") == (1, 6)
    with pytest.raises(InductiveValidationError, match="text_cnn requires"):
        bundles._infer_text_shape(torch.randn(1, 1, 1, 1), input_layout="channels_last")
    with pytest.raises(InductiveValidationError, match="requires torch.Tensor"):
        bundles._infer_text_shape([1, 2, 3], input_layout="channels_last")

    text_bundle = bundles._build_text_cnn_bundle(
        text3,
        num_classes=2,
        params={"kernel_sizes": (2,), "dropout": 0.1},
        seed=0,
        ema=False,
    )
    assert isinstance(text_bundle, TorchModelBundle)

    with pytest.raises(InductiveValidationError, match="kernel_sizes are larger"):
        bundles._build_text_cnn_bundle(
            torch.randn(2, 2, 2),
            num_classes=2,
            params={"kernel_sizes": (5,)},
            seed=0,
            ema=False,
        )


def test_parse_image_input_shape_and_prepare_audio_input() -> None:
    assert bundles._parse_image_input_shape(None) is None
    assert bundles._parse_image_input_shape((4, 5)) == (1, 4, 5)
    assert bundles._parse_image_input_shape((3, 4, 5)) == (3, 4, 5)
    with pytest.raises(InductiveValidationError, match="input_shape must be"):
        bundles._parse_image_input_shape("bad")
    with pytest.raises(InductiveValidationError, match="input_shape must be"):
        bundles._parse_image_input_shape((1, 2, 3, 4))

    x1 = bundles._prepare_audio_input(torch.randn(8), torch)
    x2 = bundles._prepare_audio_input(torch.randn(2, 8), torch)
    x3 = bundles._prepare_audio_input(torch.randn(2, 1, 8), torch)
    assert x1.ndim == 2
    assert x2.ndim == 2
    assert x3.ndim == 2
    with pytest.raises(InductiveValidationError, match="mono waveforms"):
        bundles._prepare_audio_input(torch.randn(2, 2, 8), torch)
    x4 = bundles._prepare_audio_input(torch.randn(1, 1, 1, 1), torch)
    assert x4.ndim == 3


def test_build_image_pretrained_bundle_wrapper(monkeypatch) -> None:
    from modssc.supervised.backends.torch import image_pretrained as ip

    class DummyModel(torch.nn.Module):
        def __init__(self, in_ch: int = 3):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_ch, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyModel())

    sample = torch.randn(2, 1, 4, 4)
    bundle = bundles._build_image_pretrained_bundle(
        sample,
        num_classes=2,
        params={
            "weights": None,
            "input_layout": "channels_first",
            "auto_channel_repeat": True,
            "freeze_backbone": True,
        },
        seed=0,
        ema=True,
    )
    wrapper = bundle.model
    assert bundle.ema_model is not None

    out = wrapper(sample)
    assert out.shape[0] == int(sample.shape[0])
    wrapper.train(True)
    assert wrapper.model.training is False
    assert wrapper.head.training is True

    with pytest.raises(InductiveValidationError, match="requires torch.Tensor"):
        wrapper._prepare([1, 2, 3])
    with pytest.raises(InductiveValidationError, match="input_layout must be"):
        bundle_bad = bundles._build_image_pretrained_bundle(
            sample,
            num_classes=2,
            params={"weights": None, "input_layout": "bad"},
            seed=0,
            ema=False,
        )
        bundle_bad.model._prepare(sample)
    with pytest.raises(InductiveValidationError, match="requires input_shape"):
        wrapper._prepare(torch.randn(2, 8))

    bundle_mismatch = bundles._build_image_pretrained_bundle(
        sample,
        num_classes=2,
        params={
            "weights": None,
            "auto_channel_repeat": False,
        },
        seed=0,
        ema=False,
    )
    with pytest.raises(InductiveValidationError, match="expects 3 channels"):
        bundle_mismatch.model._prepare(sample)

    bundle_unfrozen = bundles._build_image_pretrained_bundle(
        sample,
        num_classes=2,
        params={"weights": None, "freeze_backbone": False},
        seed=0,
        ema=False,
    )
    bundle_unfrozen.model.train(True)
    assert bundle_unfrozen.model.model.training is True


def test_image_pretrained_wrapper_branches(monkeypatch) -> None:
    from modssc.supervised.backends.torch import image_pretrained as ip

    class DummyModel(torch.nn.Module):
        def __init__(self, in_ch: int):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_ch, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyModel(3))

    sample = torch.randn(2, 4, 4, 3)
    bundle = bundles._build_image_pretrained_bundle(
        sample,
        num_classes=2,
        params={"weights": None, "input_layout": "channels_last"},
        seed=0,
        ema=False,
    )
    wrapper = bundle.model
    X4 = wrapper._prepare(sample)
    assert X4.shape[1:] == (3, 4, 4)

    X3 = torch.randn(2, 4, 4)
    X4b = wrapper._prepare(X3)
    assert X4b.shape[1] == 3

    wrapper.input_shape = (1, 2, 2)
    with pytest.raises(InductiveValidationError, match="input_shape does not match"):
        wrapper._prepare(torch.randn(2, 5))
    with pytest.raises(InductiveValidationError, match="requires 2D, 3D, or 4D"):
        wrapper._prepare(torch.randn(1, 1, 1, 1, 1))

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyModel(1))
    bundle2 = bundles._build_image_pretrained_bundle(
        torch.randn(2, 4),
        num_classes=2,
        params={"weights": None, "input_shape": (2, 2), "auto_channel_repeat": False},
        seed=0,
        ema=False,
    )
    X2 = bundle2.model._prepare(torch.randn(2, 4))
    assert X2.shape[1:] == (1, 2, 2)


def test_image_pretrained_wrapper_return_features(monkeypatch) -> None:
    from modssc.supervised.backends.torch import image_pretrained as ip

    class DummyModel(torch.nn.Module):
        def __init__(self, in_ch: int = 3):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_ch, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    sample = torch.randn(2, 3, 4, 4)
    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyModel())
    bundle = bundles._build_image_pretrained_bundle(
        sample,
        num_classes=2,
        params={"weights": None, "return_features": True, "input_layout": "channels_first"},
        seed=0,
        ema=False,
    )
    out = bundle.model(sample)
    assert set(out.keys()) == {"logits", "feat"}

    class DummyNoHead(DummyModel):
        def forward(self, x):
            x = self.conv(x)
            return x.mean(dim=(2, 3))

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyNoHead())
    bundle_no_head = bundles._build_image_pretrained_bundle(
        sample,
        num_classes=2,
        params={"weights": None, "return_features": True},
        seed=0,
        ema=False,
    )
    out_no_head = bundle_no_head.model(sample)
    assert torch.allclose(out_no_head["feat"], out_no_head["logits"])

    class DummyBad(DummyModel):
        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return {"logits": x}

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyBad())
    bundle_bad = bundles._build_image_pretrained_bundle(
        sample,
        num_classes=2,
        params={"weights": None, "return_features": True},
        seed=0,
        ema=False,
    )
    with pytest.raises(InductiveValidationError, match="return_features failed"):
        bundle_bad.model(sample)

    class NoInputHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(2))

        def forward(self):
            return self.param

    class DummyNoInputModel(torch.nn.Module):
        def __init__(self, in_ch: int = 3):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_ch, 4, kernel_size=1)
            self.fc = NoInputHead()

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc()

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyNoInputModel())
    monkeypatch.setattr(ip, "_replace_classifier", lambda model, _n, _t: model.fc)
    bundle_empty = bundles._build_image_pretrained_bundle(
        sample,
        num_classes=2,
        params={"weights": None, "return_features": True},
        seed=0,
        ema=False,
    )
    out_empty = bundle_empty.model(sample)
    assert set(out_empty.keys()) == {"logits", "feat"}


def test_build_audio_pretrained_bundle_wrapper(monkeypatch) -> None:
    from modssc.supervised.backends.torch import audio_pretrained as ap

    class DummyBackbone(torch.nn.Module):
        def __init__(self, feature_dim: int = 3):
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.param = torch.nn.Parameter(torch.zeros(1))

        def extract_features(self, x):
            feats = torch.zeros((int(x.shape[0]), 4, self.feature_dim), device=x.device)
            return (feats,)

    class DummyBundle:
        def __init__(self, model):
            self._model = model

        def get_model(self):
            return self._model

    monkeypatch.setattr(ap, "_load_bundle", lambda _name: DummyBundle(DummyBackbone()))
    monkeypatch.setattr(
        ap, "_extract_features", lambda _m, x, _t: torch.zeros((int(x.shape[0]), 3))
    )

    sample = torch.randn(2, 8)
    bundle = bundles._build_audio_pretrained_bundle(
        sample,
        num_classes=2,
        params={"freeze_backbone": True},
        seed=0,
        ema=True,
    )
    wrapper = bundle.model
    out = wrapper(sample)
    assert out.shape[0] == int(sample.shape[0])
    wrapper.train(True)

    bundle_features = bundles._build_audio_pretrained_bundle(
        sample,
        num_classes=2,
        params={"freeze_backbone": True, "return_features": True},
        seed=0,
        ema=False,
    )
    out_features = bundle_features.model(sample)
    assert set(out_features.keys()) == {"logits", "feat"}

    bundle_unfrozen = bundles._build_audio_pretrained_bundle(
        sample,
        num_classes=2,
        params={"freeze_backbone": False},
        seed=0,
        ema=False,
    )
    bundle_unfrozen.model.train(True)
    out = bundle_unfrozen.model(sample)
    assert out.shape[0] == int(sample.shape[0])


def test_build_torch_bundle_from_classifier(monkeypatch) -> None:
    sample = torch.randn(2, 3)

    with pytest.raises(InductiveValidationError, match="classifier_backend='torch'"):
        bundles.build_torch_bundle_from_classifier(
            classifier_id="mlp",
            classifier_backend="numpy",
            classifier_params=None,
            sample=sample,
            num_classes=2,
        )

    with pytest.raises(InductiveValidationError, match="requires torch.Tensor"):
        bundles.build_torch_bundle_from_classifier(
            classifier_id="mlp",
            classifier_backend="torch",
            classifier_params=None,
            sample=[1, 2, 3],
            num_classes=2,
        )

    with pytest.raises(InductiveValidationError, match="num_classes must be provided"):
        bundles.build_torch_bundle_from_classifier(
            classifier_id="mlp",
            classifier_backend="torch",
            classifier_params=None,
            sample=sample,
            num_classes=None,
        )

    with pytest.raises(InductiveValidationError, match="num_classes must be > 0"):
        bundles.build_torch_bundle_from_classifier(
            classifier_id="mlp",
            classifier_backend="torch",
            classifier_params=None,
            sample=sample,
            num_classes=0,
        )

    assert isinstance(
        bundles.build_torch_bundle_from_classifier(
            classifier_id="mlp",
            classifier_backend=None,
            classifier_params=None,
            sample=sample,
            num_classes=2,
        ),
        TorchModelBundle,
    )
    assert isinstance(
        bundles.build_torch_bundle_from_classifier(
            classifier_id="logreg",
            classifier_backend="auto",
            classifier_params=None,
            sample=sample,
            num_classes=2,
        ),
        TorchModelBundle,
    )
    mlp_features = bundles.build_torch_bundle_from_classifier(
        classifier_id="mlp",
        classifier_backend="torch",
        classifier_params={"return_features": True, "hidden_sizes": (4,), "dropout": 0.0},
        sample=sample,
        num_classes=2,
        ema=False,
    )
    out = mlp_features.model(sample)
    assert set(out.keys()) == {"logits", "feat"}

    logreg_features = bundles.build_torch_bundle_from_classifier(
        classifier_id="logreg",
        classifier_backend="torch",
        classifier_params={"return_features": True},
        sample=sample,
        num_classes=2,
        ema=False,
    )
    out_log = logreg_features.model(sample)
    assert torch.allclose(out_log["feat"], sample)
    assert isinstance(
        bundles.build_torch_bundle_from_classifier(
            classifier_id="image_cnn",
            classifier_backend="torch",
            classifier_params={"input_shape": (1, 3, 1)},
            sample=torch.randn(2, 3),
            num_classes=2,
            ema=False,
        ),
        TorchModelBundle,
    )
    assert isinstance(
        bundles.build_torch_bundle_from_classifier(
            classifier_id="audio_cnn",
            classifier_backend="torch",
            classifier_params=None,
            sample=torch.randn(2, 8),
            num_classes=2,
            ema=False,
        ),
        TorchModelBundle,
    )
    assert isinstance(
        bundles.build_torch_bundle_from_classifier(
            classifier_id="text_cnn",
            classifier_backend="torch",
            classifier_params={"kernel_sizes": (2,)},
            sample=torch.randn(2, 4, 3),
            num_classes=2,
            ema=False,
        ),
        TorchModelBundle,
    )

    dummy = TorchModelBundle(model=torch.nn.Linear(1, 1), optimizer=None)
    monkeypatch.setattr(bundles, "_build_image_pretrained_bundle", lambda *args, **kwargs: dummy)
    monkeypatch.setattr(bundles, "_build_audio_pretrained_bundle", lambda *args, **kwargs: dummy)
    assert (
        bundles.build_torch_bundle_from_classifier(
            classifier_id="image_pretrained",
            classifier_backend="torch",
            classifier_params=None,
            sample=sample,
            num_classes=2,
        )
        is dummy
    )
    assert (
        bundles.build_torch_bundle_from_classifier(
            classifier_id="audio_pretrained",
            classifier_backend="torch",
            classifier_params=None,
            sample=sample,
            num_classes=2,
        )
        is dummy
    )
    monkeypatch.setattr(bundles, "_build_lstm_bundle", lambda *args, **kwargs: dummy)
    assert (
        bundles.build_torch_bundle_from_classifier(
            classifier_id="lstm_scratch",
            classifier_backend="torch",
            classifier_params=None,
            sample=torch.randint(0, 3, (2, 3)),
            num_classes=2,
        )
        is dummy
    )
    monkeypatch.setattr(bundles, "_build_graphsage_bundle", lambda *args, **kwargs: dummy)
    assert (
        bundles.build_torch_bundle_from_classifier(
            classifier_id="graphsage_inductive",
            classifier_backend="torch",
            classifier_params=None,
            sample={"x": torch.randn(2, 3), "edge_index": torch.tensor([[0], [1]])},
            num_classes=2,
        )
        is dummy
    )

    with pytest.raises(InductiveValidationError, match="Unsupported torch classifier_id"):
        bundles.build_torch_bundle_from_classifier(
            classifier_id="unknown",
            classifier_backend="torch",
            classifier_params=None,
            sample=sample,
            num_classes=2,
        )


def test_prepare_audio_input_variants():
    x1 = torch.randn(8)
    out1 = bundles._prepare_audio_input(x1, torch)
    assert out1.shape == (1, 8)

    x2 = torch.randn(2, 8)
    out2 = bundles._prepare_audio_input(x2, torch)
    assert out2.shape == (2, 8)

    x3 = torch.randn(2, 1, 8)
    out3 = bundles._prepare_audio_input(x3, torch)
    assert out3.shape == (2, 8)

    with pytest.raises(InductiveValidationError, match="mono waveforms"):
        bundles._prepare_audio_input(torch.randn(2, 2, 8), torch)

    x4 = torch.randn(2, 1, 4, 4)
    out4 = bundles._prepare_audio_input(x4, torch)
    assert out4.shape == (2, 4, 4)

    with pytest.raises(InductiveValidationError, match="requires 1D, 2D, or 3D"):
        bundles._prepare_audio_input(torch.randn(2, 2, 4, 4), torch)
    with pytest.raises(InductiveValidationError, match="requires 1D, 2D, or 3D"):
        bundles._prepare_audio_input(torch.randn(1, 1, 1, 1, 1), torch)


def test_build_lstm_bundle_vocab_infer_and_bidirectional():
    sample = torch.tensor([[6, 5, 4], [3, 2, 1]], dtype=torch.int64)
    bundle = bundles._build_lstm_bundle(
        sample,
        num_classes=2,
        params={"vocab_size": 0, "bidirectional": True},
        seed=0,
        ema=False,
    )
    out = bundle.model(sample)
    assert out.shape == (2, 2)
    assert bundle.model.embedding.num_embeddings == int(sample.max().item()) + 1

    float_sample = torch.zeros((2, 3), dtype=torch.float32)
    bundle2 = bundles._build_lstm_bundle(
        float_sample,
        num_classes=2,
        params={"vocab_size": 0, "bidirectional": False},
        seed=0,
        ema=False,
    )
    out2 = bundle2.model(float_sample)
    assert out2.shape == (2, 2)
    assert bundle2.model.embedding.num_embeddings == 20000

    bundle3 = bundles._build_lstm_bundle(
        sample,
        num_classes=2,
        params={"vocab_size": 5, "bidirectional": True},
        seed=0,
        ema=False,
    )
    assert bundle3.model.embedding.num_embeddings == 5


def _install_fake_tg_nn(monkeypatch, *, with_sage: bool):
    nn_mod = types.ModuleType("torch_geometric.nn")
    if with_sage:

        class SAGEConv(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)

            def forward(self, x, edge_index):
                return self.lin(x)

        nn_mod.SAGEConv = SAGEConv

    tg = types.ModuleType("torch_geometric")
    tg.nn = nn_mod
    monkeypatch.setitem(sys.modules, "torch_geometric", tg)
    monkeypatch.setitem(sys.modules, "torch_geometric.nn", nn_mod)


def test_build_graphsage_bundle_dict(monkeypatch):
    _install_fake_tg_nn(monkeypatch, with_sage=True)
    sample = {"x": torch.randn(3, 4), "edge_index": torch.tensor([[0, 1], [1, 2]])}
    bundle = bundles._build_graphsage_bundle(
        sample,
        num_classes=2,
        params={"hidden_channels": 4, "num_layers": 3, "dropout": 0.0},
        seed=0,
        ema=False,
    )
    out = bundle.model(sample)
    assert set(out.keys()) == {"logits", "feat"}
    assert out["logits"].shape[0] == 3


def test_build_graphsage_bundle_hidden_sizes(monkeypatch):
    _install_fake_tg_nn(monkeypatch, with_sage=True)
    sample = {"x": torch.randn(4, 5), "edge_index": torch.tensor([[0, 1], [1, 2]])}
    bundle = bundles._build_graphsage_bundle(
        sample,
        num_classes=2,
        params={"hidden_sizes": [8, 4], "dropout": 0.0},
        seed=0,
        ema=False,
    )
    assert len(bundle.model.convs) == 3
    assert bundle.model.convs[0].lin.out_features == 8
    assert bundle.model.convs[1].lin.out_features == 4


def test_build_graphsage_bundle_hidden_sizes_int(monkeypatch):
    _install_fake_tg_nn(monkeypatch, with_sage=True)
    sample = {"x": torch.randn(2, 3), "edge_index": torch.tensor([[0], [1]])}
    bundle = bundles._build_graphsage_bundle(
        sample,
        num_classes=2,
        params={"hidden_sizes": 8, "dropout": 0.0},
        seed=0,
        ema=False,
    )
    assert len(bundle.model.convs) == 2


def test_build_graphsage_bundle_errors(monkeypatch):
    _install_fake_tg_nn(monkeypatch, with_sage=False)
    with pytest.raises(ImportError, match="torch_geometric is required"):
        bundles._build_graphsage_bundle(
            torch.randn(2, 3),
            num_classes=2,
            params={},
            seed=0,
            ema=False,
        )

    _install_fake_tg_nn(monkeypatch, with_sage=True)
    with pytest.raises(InductiveValidationError, match="hidden_sizes must be an int"):
        bundles._build_graphsage_bundle(
            {"x": torch.randn(2, 3), "edge_index": torch.tensor([[0], [1]])},
            num_classes=2,
            params={"hidden_sizes": "bad"},
            seed=0,
            ema=False,
        )
    with pytest.raises(InductiveValidationError, match="hidden_sizes must be positive"):
        bundles._build_graphsage_bundle(
            {"x": torch.randn(2, 3), "edge_index": torch.tensor([[0], [1]])},
            num_classes=2,
            params={"hidden_sizes": [-1]},
            seed=0,
            ema=False,
        )
    with pytest.raises(InductiveValidationError, match="num_layers must equal"):
        bundles._build_graphsage_bundle(
            {"x": torch.randn(2, 3), "edge_index": torch.tensor([[0], [1]])},
            num_classes=2,
            params={"hidden_sizes": [8, 4], "num_layers": 2},
            seed=0,
            ema=False,
        )
    with pytest.raises(InductiveValidationError, match="requires sample"):
        bundles._build_graphsage_bundle(
            object(),
            num_classes=2,
            params={},
            seed=0,
            ema=False,
        )


def test_build_graphsage_bundle_tensor_input_and_forward_error(monkeypatch):
    _install_fake_tg_nn(monkeypatch, with_sage=True)
    sample = torch.randn(3, 4)
    bundle = bundles._build_graphsage_bundle(
        sample,
        num_classes=2,
        params={"hidden_channels": 4, "num_layers": 3, "dropout": 0.0},
        seed=0,
        ema=False,
    )
    with pytest.raises(ValueError, match="expects a dict input"):
        bundle.model(sample)
