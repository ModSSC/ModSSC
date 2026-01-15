from __future__ import annotations

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on optional torch install
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.supervised.backends.torch import image_pretrained as ip
from modssc.supervised.errors import SupervisedValidationError


def test_supports_arg_and_select_weights() -> None:
    def fn(a, b=1):
        return a + b

    assert ip._supports_arg(fn, "a")
    assert not ip._supports_arg(fn, "missing")

    class DummyWeights:
        DEFAULT = "default"
        FOO = "foo"

    assert ip._select_weights(DummyWeights, "DEFAULT") == "default"
    assert ip._select_weights(DummyWeights, "foo") == "foo"

    class DummyWeightsLower:
        DEFAULT = "default"
        foo = "lower"

    assert ip._select_weights(DummyWeightsLower, "foo") == "lower"

    with pytest.raises(SupervisedValidationError, match="Unknown weights enum"):
        ip._select_weights(DummyWeights, "missing")


def test_torchvision_helper(monkeypatch) -> None:
    dummy = object()
    monkeypatch.setattr(ip, "optional_import", lambda *_a, **_k: dummy)
    assert ip._torchvision() is dummy


def test_supports_arg_signature_error(monkeypatch) -> None:
    def _boom(_fn):
        raise ValueError("boom")

    monkeypatch.setattr(ip.inspect, "signature", _boom)
    assert ip._supports_arg(lambda x: x, "x") is False


def test_resolve_weights_variants() -> None:
    class DummyWeights:
        DEFAULT = "default"
        BAR = "bar"

    class ModelsWithGet:
        def get_model_weights(self, _name):
            return DummyWeights

    models = ModelsWithGet()
    assert ip._resolve_weights(models, "model", None) is None
    assert ip._resolve_weights(models, "model", "none") is None
    assert ip._resolve_weights(models, "model", "bar") == "bar"

    class ModelsMap:
        ResNet18_Weights = DummyWeights

    assert ip._resolve_weights(ModelsMap(), "resnet18", "DEFAULT") == "default"

    class ModelsEmpty:
        pass

    assert ip._resolve_weights(ModelsEmpty(), "unknown_model", "DEFAULT") is ip._PRETRAINED_SENTINEL
    assert ip._resolve_weights(ModelsEmpty(), "model", {"w": 1}) == {"w": 1}


def test_infer_in_channels_and_replace_classifier() -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    model = DummyModel()
    assert ip._infer_in_channels(model, torch) == 3

    head = ip._replace_classifier(model, 5, torch)
    assert isinstance(head, torch.nn.Linear)
    assert model.fc.out_features == 5

    class SeqModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 2),
            )

    seq_model = SeqModel()
    head2 = ip._replace_classifier(seq_model, 3, torch)
    assert seq_model.classifier[-1].out_features == 3
    assert isinstance(head2, torch.nn.Linear)

    class SeqModelTailNonLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )

    seq_tail = SeqModelTailNonLinear()
    head2b = ip._replace_classifier(seq_tail, 6, torch)
    assert isinstance(head2b, torch.nn.Linear)
    assert seq_tail.classifier[0].out_features == 6

    class SeqModelNoLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Sequential(torch.nn.ReLU())
            self.head = torch.nn.Linear(4, 2)

    seq_no_linear = SeqModelNoLinear()
    head2c = ip._replace_classifier(seq_no_linear, 4, torch)
    assert isinstance(head2c, torch.nn.Linear)
    assert seq_no_linear.head.out_features == 4

    class ClassifierOther(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.ReLU()
            self.head = torch.nn.Linear(4, 2)

    classifier_other = ClassifierOther()
    head2d = ip._replace_classifier(classifier_other, 5, torch)
    assert isinstance(head2d, torch.nn.Linear)
    assert classifier_other.head.out_features == 5

    class ClassifierModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(4, 2)

    cls_model = ClassifierModel()
    head3 = ip._replace_classifier(cls_model, 3, torch)
    assert isinstance(head3, torch.nn.Linear)
    assert cls_model.classifier.out_features == 3

    class HeadsObj:
        def __init__(self):
            self.head = torch.nn.Linear(4, 2)

    class HeadsModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = HeadsObj()

    heads_model = HeadsModel()
    head4 = ip._replace_classifier(heads_model, 3, torch)
    assert isinstance(head4, torch.nn.Linear)
    assert heads_model.heads.head.out_features == 3

    class HeadsLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = torch.nn.Linear(4, 2)

    heads_linear = HeadsLinear()
    head5 = ip._replace_classifier(heads_linear, 3, torch)
    assert isinstance(head5, torch.nn.Linear)
    assert heads_linear.heads.out_features == 3

    class HeadsFallback(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = torch.nn.ReLU()
            self.head = torch.nn.Linear(4, 2)

    heads_fallback = HeadsFallback()
    head_fallback = ip._replace_classifier(heads_fallback, 7, torch)
    assert isinstance(head_fallback, torch.nn.Linear)
    assert heads_fallback.head.out_features == 7

    class HeadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 2)

    head_model = HeadModel()
    head6 = ip._replace_classifier(head_model, 3, torch)
    assert isinstance(head6, torch.nn.Linear)
    assert head_model.head.out_features == 3

    class NoHead(torch.nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(SupervisedValidationError, match="Unable to replace classifier"):
        ip._replace_classifier(NoHead(), 2, torch)

    class NoConv(torch.nn.Module):
        def forward(self, x):
            return x

    assert ip._infer_in_channels(NoConv(), torch) is None


def test_load_model_get_model_path(monkeypatch) -> None:
    class DummyWeights:
        DEFAULT = "ok"

    class DummyModels:
        def get_model_weights(self, _name):
            return DummyWeights

        def get_model(self, name, weights=None):
            return {"name": name, "weights": weights}

    class DummyTV:
        models = DummyModels()

    monkeypatch.setattr(ip, "_torchvision", lambda: DummyTV)
    out = ip._load_model("resnet18", "DEFAULT")
    assert out["weights"] == DummyWeights.DEFAULT


def test_load_model_pretrained_fallback(monkeypatch) -> None:
    class DummyModels:
        def __init__(self):
            self.called = None

        def foo(self, pretrained=False):
            self.called = bool(pretrained)
            return {"pretrained": pretrained}

    models = DummyModels()

    class DummyTV:
        pass

    DummyTV.models = models

    monkeypatch.setattr(ip, "_torchvision", lambda: DummyTV)
    out = ip._load_model("foo", "DEFAULT")
    assert out["pretrained"] is True


def test_load_model_get_model_weights_missing(monkeypatch) -> None:
    class DummyModels:
        def get_model(self, _name, weights=None):
            return {"weights": weights}

    class DummyTV:
        models = DummyModels()

    monkeypatch.setattr(ip, "_torchvision", lambda: DummyTV)
    with pytest.raises(SupervisedValidationError, match="Unable to resolve pretrained weights"):
        ip._load_model("unknown", "DEFAULT")


def test_load_model_weights_branches(monkeypatch) -> None:
    class DummyModels:
        def foo(self, weights=None):
            return {"weights": weights}

        def bar(self, pretrained=False):
            return {"pretrained": pretrained}

        def baz(self):
            return {"ok": True}

    class DummyTV:
        models = DummyModels()

    monkeypatch.setattr(ip, "_torchvision", lambda: DummyTV)

    out = ip._load_model("foo", {"w": 1})
    assert out["weights"] == {"w": 1}

    out2 = ip._load_model("bar", {"w": 1})
    assert out2["pretrained"] is True

    with pytest.raises(SupervisedValidationError, match="does not support pretrained weights"):
        ip._load_model("baz", {"w": 1})

    with pytest.raises(SupervisedValidationError, match="Unable to resolve pretrained weights"):
        ip._load_model("baz", "DEFAULT")


def test_load_model_no_weights_kwargs(monkeypatch) -> None:
    class DummyModels:
        def noop(self):
            return {"ok": True}

    class DummyTV:
        models = DummyModels()

    monkeypatch.setattr(ip, "_torchvision", lambda: DummyTV)
    out = ip._load_model("noop", None)
    assert out["ok"] is True


def test_load_model_unknown_model(monkeypatch) -> None:
    class DummyTV:
        models = object()

    monkeypatch.setattr(ip, "_torchvision", lambda: DummyTV)
    with pytest.raises(SupervisedValidationError, match="Unknown torchvision model"):
        ip._load_model("missing", None)


def test_image_pretrained_fit_predict_with_stub_model(monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyModel())

    X = torch.randn(4, 1, 4, 4, dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

    clf = ip.TorchImagePretrainedClassifier(max_epochs=1, batch_size=2, weights=None)
    fit = clf.fit(X, y)
    assert fit.n_samples == int(X.shape[0])
    assert clf.supports_proba

    scores = clf.predict_scores(X)
    assert scores.shape == (int(X.shape[0]), int(torch.unique(y).numel()))

    proba = clf.predict_proba(X)
    assert torch.allclose(scores, proba)

    pred = clf.predict(X)
    assert pred.shape[0] == int(X.shape[0])


def test_image_pretrained_channels_last_auto_repeat(monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyModel())

    X = torch.randn(2, 4, 4, 1, dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.int64)

    clf = ip.TorchImagePretrainedClassifier(
        max_epochs=1,
        batch_size=1,
        weights=None,
        input_layout="channels_last",
        auto_channel_repeat=True,
    )
    clf.fit(X, y)
    scores = clf.predict_scores(X)
    assert scores.shape[0] == int(X.shape[0])


def test_image_pretrained_channel_mismatch(monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyModel())

    X = torch.randn(2, 1, 4, 4, dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.int64)

    clf = ip.TorchImagePretrainedClassifier(
        max_epochs=1, batch_size=1, weights=None, auto_channel_repeat=False
    )
    with pytest.raises(SupervisedValidationError, match="expects 3 channels"):
        clf.fit(X, y)


def test_image_pretrained_prepare_x_variants(monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    monkeypatch.setattr(ip, "_load_model", lambda *_a, **_k: DummyModel())

    clf = ip.TorchImagePretrainedClassifier(weights=None, max_epochs=1, batch_size=1)
    clf._expected_in_channels = 3

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor X"):
        clf._prepare_X([1, 2, 3], torch, allow_infer=True)

    bad_layout = ip.TorchImagePretrainedClassifier(
        weights=None, max_epochs=1, batch_size=1, input_layout="bad"
    )
    with pytest.raises(SupervisedValidationError, match="input_layout must be"):
        bad_layout._prepare_X(torch.randn(2, 3, 4, 4), torch, allow_infer=True)

    with pytest.raises(SupervisedValidationError, match="requires 3D/4D inputs"):
        clf._prepare_X(torch.randn(2, 4), torch, allow_infer=True)

    clf.input_shape = (2, 2)
    X2 = torch.randn(2, 4)
    X4 = clf._prepare_X(X2, torch, allow_infer=True)
    assert X4.shape[1:] == (3, 2, 2)

    clf2 = ip.TorchImagePretrainedClassifier(input_shape=(1, 2, 2), weights=None)
    with pytest.raises(SupervisedValidationError, match="input_shape does not match"):
        clf2._prepare_X(torch.randn(2, 5), torch, allow_infer=True)

    clf3 = ip.TorchImagePretrainedClassifier(input_shape=(1, 2, 2), weights=None)
    clf3._input_shape = (1, 2, 2)
    with pytest.raises(SupervisedValidationError, match="X shape does not match"):
        clf3._prepare_X(torch.randn(2, 1, 3, 3), torch, allow_infer=False)

    clf4 = ip.TorchImagePretrainedClassifier(weights=None, auto_channel_repeat=True)
    clf4._expected_in_channels = 3
    X1 = torch.randn(2, 1, 2, 2)
    X4b = clf4._prepare_X(X1, torch, allow_infer=True)
    assert X4b.shape[1] == 3


def test_image_pretrained_prepare_x_extra_branches(monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    monkeypatch.setattr(ip, "_load_model", lambda *_a, **_k: DummyModel())

    clf = ip.TorchImagePretrainedClassifier(weights=None, max_epochs=1, batch_size=1)
    clf._expected_in_channels = None
    X3 = torch.randn(2, 4, 4)
    X4 = clf._prepare_X(X3, torch, allow_infer=True)
    assert X4.shape[1:] == (1, 4, 4)

    clf._expected_in_channels = 3
    X4_ok = clf._prepare_X(torch.randn(2, 3, 4, 4), torch, allow_infer=True)
    assert X4_ok.shape[1] == 3

    clf2 = ip.TorchImagePretrainedClassifier(weights=None, input_shape=(1, 2, 2))
    clf2._input_shape = (1, 2, 2)
    X2 = torch.randn(2, 4)
    X4b = clf2._prepare_X(X2, torch, allow_infer=True)
    assert X4b.shape[1:] == (1, 2, 2)

    bad_shape = ip.TorchImagePretrainedClassifier(weights=None, input_shape=(1, 2, 3, 4))
    with pytest.raises(SupervisedValidationError, match="input_shape must be"):
        bad_shape._prepare_X(torch.randn(2, 4), torch, allow_infer=True)

    with pytest.raises(SupervisedValidationError, match="2D, 3D, or 4D"):
        clf._prepare_X(torch.randn(1, 1, 1, 1, 1), torch, allow_infer=True)


def test_image_pretrained_set_train_mode_and_errors(monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    monkeypatch.setattr(ip, "_load_model", lambda *_a, **_k: DummyModel())

    clf = ip.TorchImagePretrainedClassifier(weights=None, freeze_backbone=True)
    clf._set_train_mode()

    clf._model = DummyModel()
    clf._head = clf._model.fc
    clf.freeze_backbone = False
    clf._set_train_mode()
    assert clf._model.training is True

    X = torch.randn(2, 1, 2, 2)
    y = torch.tensor([0, 1], dtype=torch.int64)

    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor X"):
        ip.TorchImagePretrainedClassifier(weights=None, max_epochs=1).fit([1, 2], y)
    with pytest.raises(SupervisedValidationError, match="requires torch.Tensor y"):
        ip.TorchImagePretrainedClassifier(weights=None, max_epochs=1).fit(X, [0, 1])
    with pytest.raises(SupervisedValidationError, match="input_layout must be"):
        ip.TorchImagePretrainedClassifier(weights=None, max_epochs=1, input_layout="bad").fit(X, y)
    with pytest.raises(SupervisedValidationError, match="batch_size must be >= 1"):
        ip.TorchImagePretrainedClassifier(weights=None, batch_size=0).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="max_epochs must be >= 1"):
        ip.TorchImagePretrainedClassifier(weights=None, max_epochs=0).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="lr must be > 0"):
        ip.TorchImagePretrainedClassifier(weights=None, lr=0.0).fit(X, y)
    with pytest.raises(SupervisedValidationError, match="X must be non-empty"):
        ip.TorchImagePretrainedClassifier(weights=None, max_epochs=1).fit(
            torch.zeros((0, 1, 2, 2)), torch.zeros((0,), dtype=torch.int64)
        )
    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    y_mismatch = torch.empty((2,), dtype=torch.int64, device=mismatch_device)
    with pytest.raises(SupervisedValidationError, match="same device"):
        ip.TorchImagePretrainedClassifier(weights=None, max_epochs=1).fit(X, y_mismatch)

    clf_ok = ip.TorchImagePretrainedClassifier(weights=None, max_epochs=1, batch_size=1)
    clf_ok.fit(X, y.view(-1, 1))


def test_image_pretrained_ambiguous_channel_input(monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyModel())

    # 2 channels, expecting 3
    X = torch.randn(2, 2, 4, 4)
    y = torch.tensor([0, 1], dtype=torch.int64)
    clf = ip.TorchImagePretrainedClassifier(
        weights=None, max_epochs=1, batch_size=1, auto_channel_repeat=True
    )

    with pytest.raises(SupervisedValidationError, match="Ambiguous 2-channel input"):
        clf.fit(X, y)


def test_image_pretrained_fit_mismatched_first_dim(monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyModel())

    X = torch.randn(2, 1, 2, 2)
    y = torch.tensor([0, 1, 0], dtype=torch.int64)
    clf = ip.TorchImagePretrainedClassifier(weights=None, max_epochs=1, batch_size=1)
    with pytest.raises(SupervisedValidationError, match="matching first dimension"):
        clf.fit(X, y)


def test_image_pretrained_freeze_backbone_false(monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyModel())

    X = torch.randn(2, 1, 2, 2)
    y = torch.tensor([0, 1], dtype=torch.int64)
    clf = ip.TorchImagePretrainedClassifier(
        weights=None, max_epochs=1, batch_size=1, freeze_backbone=False
    )
    clf.fit(X, y)


def test_image_pretrained_scores_validation(monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    monkeypatch.setattr(ip, "_load_model", lambda *_args, **_kwargs: DummyModel())

    X = torch.randn(2, 1, 2, 2)
    y = torch.tensor([0, 1], dtype=torch.int64)
    clf = ip.TorchImagePretrainedClassifier(weights=None, max_epochs=1, batch_size=1)

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_scores(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(X)

    clf.fit(X, y)
    mismatch_device = "cuda" if torch.cuda.is_available() else "meta"
    X_mismatch = torch.empty((2, 1, 2, 2), dtype=torch.float32, device=mismatch_device)
    with pytest.raises(SupervisedValidationError, match="same device"):
        clf.predict_scores(X_mismatch)
