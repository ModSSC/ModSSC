from __future__ import annotations

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on optional torch install
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.supervised.backends.torch import audio_pretrained as ap
from modssc.supervised.errors import SupervisedValidationError


def _stub_bundle(monkeypatch, *, feature_dim: int = 3):
    class DummyBackbone(torch.nn.Module):
        def __init__(self):
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
    return DummyBackbone


def test_extract_features_variants() -> None:
    class DummyWithExtract(torch.nn.Module):
        def extract_features(self, x):
            feats = torch.zeros((int(x.shape[0]), 4, 3), device=x.device)
            return (feats,)

    x = torch.zeros((2, 8), dtype=torch.float32)
    out = ap._extract_features(DummyWithExtract(), x, torch)
    assert out.shape == (2, 3)

    class DummyCall(torch.nn.Module):
        def forward(self, x):
            return [torch.zeros((int(x.shape[0]), 5), device=x.device)]

    out2 = ap._extract_features(DummyCall(), x, torch)
    assert out2.shape == (2, 5)

    class DummyBad(torch.nn.Module):
        def forward(self, _x):
            return torch.zeros((3,), dtype=torch.float32)

    with pytest.raises(SupervisedValidationError, match="invalid feature shape"):
        ap._extract_features(DummyBad(), x, torch)

    class DummyNonTensor(torch.nn.Module):
        def forward(self, _x):
            return ["not_a_tensor"]

    with pytest.raises(SupervisedValidationError, match="Unexpected feature output"):
        ap._extract_features(DummyNonTensor(), x, torch)


def test_torchaudio_helper(monkeypatch) -> None:
    dummy = object()
    monkeypatch.setattr(ap, "optional_import", lambda *_a, **_k: dummy)
    assert ap._torchaudio() is dummy


def test_load_bundle_unknown(monkeypatch) -> None:
    class DummyAudio:
        pipelines = object()

    monkeypatch.setattr(ap, "_torchaudio", lambda: DummyAudio)
    with pytest.raises(SupervisedValidationError, match="Unknown torchaudio bundle"):
        ap._load_bundle("MISSING")


def test_load_bundle_success(monkeypatch) -> None:
    class DummyAudio:
        class pipelines:
            DUMMY = "ok"

    monkeypatch.setattr(ap, "_torchaudio", lambda: DummyAudio)
    assert ap._load_bundle("DUMMY") == "ok"


def test_audio_pretrained_fit_predict_with_stub_bundle(monkeypatch) -> None:
    _stub_bundle(monkeypatch)

    X = torch.randn(4, 8, dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

    clf = ap.TorchAudioPretrainedClassifier(max_epochs=1, batch_size=2)
    fit = clf.fit(X, y)
    assert fit.n_samples == int(X.shape[0])
    assert clf.supports_proba

    scores = clf.predict_scores(X)
    assert scores.shape == (int(X.shape[0]), int(torch.unique(y).numel()))

    proba = clf.predict_proba(X)
    assert torch.allclose(scores, proba)

    pred = clf.predict(X)
    assert pred.shape[0] == int(X.shape[0])


def test_audio_pretrained_prepare_x_invalid_channels() -> None:
    clf = ap.TorchAudioPretrainedClassifier()
    bad = torch.randn(2, 2, 8, dtype=torch.float32)
    with pytest.raises(SupervisedValidationError, match="expects mono waveforms"):
        clf._prepare_X(bad, torch)


def test_audio_pretrained_prepare_x_shapes() -> None:
    clf = ap.TorchAudioPretrainedClassifier()
    x1 = torch.randn(8, dtype=torch.float32)
    x2 = torch.randn(2, 8, dtype=torch.float32)
    x3 = torch.randn(2, 1, 8, dtype=torch.float32)

    out1 = clf._prepare_X(x1, torch)
    out2 = clf._prepare_X(x2, torch)
    out3 = clf._prepare_X(x3, torch)

    assert out1.ndim == 2
    assert out2.ndim == 2
    assert out3.ndim == 2


def test_audio_pretrained_prepare_x_errors() -> None:
    clf = ap.TorchAudioPretrainedClassifier()
    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor X"):
        clf._prepare_X([1, 2, 3], torch)
    with pytest.raises(SupervisedValidationError, match="1D, 2D, or 3D"):
        clf._prepare_X(torch.zeros((1, 1, 1, 1), dtype=torch.float32), torch)


def test_audio_pretrained_set_train_mode_paths() -> None:
    clf = ap.TorchAudioPretrainedClassifier(freeze_backbone=False)
    clf._set_train_mode()

    clf._backbone = torch.nn.Linear(1, 1)
    clf._head = None
    clf._set_train_mode()
    assert clf._backbone.training is True

    clf._head = torch.nn.Linear(1, 1)
    clf._set_train_mode()
    assert clf._head.training is True


def test_audio_pretrained_fit_validation_errors(monkeypatch) -> None:
    _stub_bundle(monkeypatch)
    X = torch.randn(2, 8, dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.int64)

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor X"):
        ap.TorchAudioPretrainedClassifier(max_epochs=1).fit([1, 2], y)
    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor y"):
        ap.TorchAudioPretrainedClassifier(max_epochs=1).fit(X, [0, 1])
    with pytest.raises(SupervisedValidationError, match="batch_size must be >= 1"):
        ap.TorchAudioPretrainedClassifier(batch_size=0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="max_epochs must be >= 1"):
        ap.TorchAudioPretrainedClassifier(max_epochs=0).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="lr must be > 0"):
        ap.TorchAudioPretrainedClassifier(lr=0.0, max_epochs=1).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="X must be non-empty"):
        ap.TorchAudioPretrainedClassifier(max_epochs=1).fit(
            torch.zeros((0, 8), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )
    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    y_mismatch = torch.empty((2,), dtype=torch.int64, device=mismatch_device)
    with pytest.raises(SupervisedValidationError, match="same device"):
        ap.TorchAudioPretrainedClassifier(max_epochs=1).fit(X, y_mismatch)

    clf = ap.TorchAudioPretrainedClassifier(max_epochs=1, batch_size=1)
    clf.fit(X, y.view(-1, 1))

    with pytest.raises(SupervisedValidationError, match="matching first dimension"):
        clf.fit(X, torch.tensor([0, 1, 0], dtype=torch.int64))


def test_audio_pretrained_freeze_backbone_false(monkeypatch) -> None:
    _stub_bundle(monkeypatch)
    X = torch.randn(2, 8, dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.int64)

    clf = ap.TorchAudioPretrainedClassifier(max_epochs=1, batch_size=1, freeze_backbone=False)
    clf.fit(X, y)

    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    X_mismatch = torch.empty((2, 8), dtype=torch.float32, device=mismatch_device)
    with pytest.raises(SupervisedValidationError, match="same device"):
        clf.predict_scores(X_mismatch)


def test_audio_pretrained_predict_requires_fit() -> None:
    X = torch.randn(2, 8, dtype=torch.float32)
    clf = ap.TorchAudioPretrainedClassifier()

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_scores(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(X)
