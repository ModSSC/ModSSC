from __future__ import annotations

import copy

import pytest
import torch

from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods import (
    adamatch,
    fixmatch,
    flexmatch,
    free_match,
    mixmatch,
    softmatch,
    uda,
)
from modssc.inductive.methods.adamatch import AdaMatchMethod, AdaMatchSpec
from modssc.inductive.methods.fixmatch import FixMatchMethod, FixMatchSpec
from modssc.inductive.methods.flexmatch import FlexMatchMethod, FlexMatchSpec
from modssc.inductive.methods.free_match import FreeMatchMethod, FreeMatchSpec
from modssc.inductive.methods.mean_teacher import MeanTeacherMethod, MeanTeacherSpec
from modssc.inductive.methods.mixmatch import MixMatchMethod, MixMatchSpec
from modssc.inductive.methods.pi_model import PiModelMethod, PiModelSpec
from modssc.inductive.methods.softmatch import SoftMatchMethod, SoftMatchSpec
from modssc.inductive.methods.uda import UDAMethod, UDASpec
from modssc.inductive.types import DeviceSpec

from .conftest import (
    DummyDataset,
    make_model_bundle,
    make_numpy_dataset,
    make_torch_dataset,
    make_torch_ssl_dataset,
)

DEEP_METHODS = [
    PiModelMethod,
    MeanTeacherMethod,
    FixMatchMethod,
    FlexMatchMethod,
    UDAMethod,
    MixMatchMethod,
    AdaMatchMethod,
    FreeMatchMethod,
    SoftMatchMethod,
]


DEEP_METHOD_MODULES = {
    PiModelMethod: "modssc.inductive.methods.pi_model",
    MeanTeacherMethod: "modssc.inductive.methods.mean_teacher",
    FixMatchMethod: "modssc.inductive.methods.fixmatch",
    FlexMatchMethod: "modssc.inductive.methods.flexmatch",
    UDAMethod: "modssc.inductive.methods.uda",
    MixMatchMethod: "modssc.inductive.methods.mixmatch",
    AdaMatchMethod: "modssc.inductive.methods.adamatch",
    FreeMatchMethod: "modssc.inductive.methods.free_match",
    SoftMatchMethod: "modssc.inductive.methods.softmatch",
}


CAT_METHODS = [
    FixMatchMethod,
    FlexMatchMethod,
    UDAMethod,
    AdaMatchMethod,
    FreeMatchMethod,
    SoftMatchMethod,
]


def _make_spec(method_cls, bundle, **overrides):
    if method_cls is PiModelMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return PiModelSpec(**kwargs)
    if method_cls is MeanTeacherMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return MeanTeacherSpec(**kwargs)
    if method_cls is FixMatchMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return FixMatchSpec(**kwargs)
    if method_cls is FlexMatchMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return FlexMatchSpec(**kwargs)
    if method_cls is UDAMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return UDASpec(**kwargs)
    if method_cls is MixMatchMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return MixMatchSpec(**kwargs)
    if method_cls is AdaMatchMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return AdaMatchSpec(**kwargs)
    if method_cls is FreeMatchMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return FreeMatchSpec(**kwargs)
    if method_cls is SoftMatchMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return SoftMatchSpec(**kwargs)
    raise AssertionError("unknown method")


def _make_flex_data():
    data = make_torch_ssl_dataset()
    idx_u = torch.arange(int(data.X_u_w.shape[0]), device=data.X_u_w.device, dtype=torch.int64)
    return DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": idx_u, "ulb_size": int(idx_u.numel())},
    )


def _data_for_method(method_cls):
    return _make_flex_data() if method_cls is FlexMatchMethod else make_torch_ssl_dataset()


def _fit_predict(method, data):
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    proba = method.predict_proba(data.X_l)
    pred = method.predict(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])
    assert int(pred.shape[0]) == int(data.X_l.shape[0])


def _make_bundle_for(model, *, with_ema: bool = False) -> TorchModelBundle:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema_model = copy.deepcopy(model) if with_ema else None
    return TorchModelBundle(model=model, optimizer=optimizer, ema_model=ema_model)


class _BadLogits1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros((int(x.shape[0]),), device=x.device)


class _BadBatch(torch.nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.n_classes = int(n_classes)

    def forward(self, x):
        batch = max(0, int(x.shape[0]) - 1)
        return torch.zeros((batch, self.n_classes), device=x.device)


class _ConditionalClasses(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch = int(x.shape[0])
        if float(x.sum().item()) >= 0.0:
            return torch.zeros((batch, 2), device=x.device)
        return torch.zeros((batch, 3), device=x.device)


class _LogitsByBatch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch = int(x.shape[0])
        if batch <= 2:
            return torch.zeros((batch, 2), device=x.device)
        return torch.zeros((batch,), device=x.device)


def test_deep_methods_validation_errors():
    data_np = make_numpy_dataset()
    data_t = make_torch_dataset()
    data_ssl = make_torch_ssl_dataset()
    bad_labels = DummyDataset(
        X_l=data_ssl.X_l,
        y_l=data_ssl.y_l.to(torch.float32),
        X_u=data_ssl.X_u,
        X_u_w=data_ssl.X_u_w,
        X_u_s=data_ssl.X_u_s,
    )

    for method_cls in DEEP_METHODS:
        with pytest.raises(InductiveValidationError):
            method_cls().fit(None, device=DeviceSpec(device="cpu"), seed=0)
        with pytest.raises(InductiveValidationError):
            method_cls().fit(data_np, device=DeviceSpec(device="cpu"), seed=0)
        with pytest.raises(InductiveValidationError):
            method_cls().fit(data_t, device=DeviceSpec(device="cpu"), seed=0)
        with pytest.raises(InductiveValidationError):
            method_cls().fit(bad_labels, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize("method_cls", DEEP_METHODS)
def test_deep_methods_require_bundle_and_predict_errors(method_cls):
    data_ssl = make_torch_ssl_dataset()
    with pytest.raises(InductiveValidationError):
        method_cls().fit(data_ssl, device=DeviceSpec(device="cpu"), seed=0)
    with pytest.raises(RuntimeError):
        method_cls().predict_proba(data_ssl.X_l)


@pytest.mark.parametrize("method_cls", DEEP_METHODS)
def test_deep_methods_empty_inputs(method_cls):
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    spec = _make_spec(method_cls, bundle)

    empty_xl = DummyDataset(
        X_l=torch.zeros((0, data.X_l.shape[1])),
        y_l=torch.zeros((0,), dtype=torch.int64),
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
    )
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(empty_xl, device=DeviceSpec(device="cpu"), seed=0)

    empty_u = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=torch.zeros((0, data.X_l.shape[1])),
        X_u_s=torch.zeros((0, data.X_l.shape[1])),
    )
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(empty_u, device=DeviceSpec(device="cpu"), seed=0)

    mismatch = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=torch.zeros((2, data.X_l.shape[1])),
        X_u_s=torch.zeros((3, data.X_l.shape[1])),
    )
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(mismatch, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize("method_cls", DEEP_METHODS)
def test_deep_methods_invalid_batch_and_lambda(method_cls):
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()

    with pytest.raises(InductiveValidationError):
        method_cls(_make_spec(method_cls, bundle, batch_size=0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        method_cls(_make_spec(method_cls, bundle, max_epochs=0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        method_cls(_make_spec(method_cls, bundle, lambda_u=-1.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )


def test_deep_method_specific_invalid_specs():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()

    with pytest.raises(InductiveValidationError):
        FixMatchMethod(_make_spec(FixMatchMethod, bundle, p_cutoff=2.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        FixMatchMethod(_make_spec(FixMatchMethod, bundle, temperature=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )

    with pytest.raises(InductiveValidationError):
        FlexMatchMethod(_make_spec(FlexMatchMethod, bundle, p_cutoff=-1.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )

    with pytest.raises(InductiveValidationError):
        UDAMethod(_make_spec(UDAMethod, bundle, p_cutoff=-0.1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        UDAMethod(_make_spec(UDAMethod, bundle, temperature=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )

    with pytest.raises(InductiveValidationError):
        AdaMatchMethod(_make_spec(AdaMatchMethod, bundle, ema_p=1.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        AdaMatchMethod(_make_spec(AdaMatchMethod, bundle, p_cutoff=1.5)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        AdaMatchMethod(_make_spec(AdaMatchMethod, bundle, temperature=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )

    with pytest.raises(InductiveValidationError):
        FreeMatchMethod(_make_spec(FreeMatchMethod, bundle, lambda_e=-1.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        FreeMatchMethod(_make_spec(FreeMatchMethod, bundle, temperature=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        FreeMatchMethod(_make_spec(FreeMatchMethod, bundle, ema_p=1.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )

    with pytest.raises(InductiveValidationError):
        SoftMatchMethod(_make_spec(SoftMatchMethod, bundle, n_sigma=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        SoftMatchMethod(_make_spec(SoftMatchMethod, bundle, temperature=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        SoftMatchMethod(_make_spec(SoftMatchMethod, bundle, ema_p=1.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )

    with pytest.raises(InductiveValidationError):
        MixMatchMethod(_make_spec(MixMatchMethod, bundle, mixup_alpha=-0.1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        MixMatchMethod(_make_spec(MixMatchMethod, bundle, temperature=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        MixMatchMethod(_make_spec(MixMatchMethod, bundle, unsup_warm_up=-0.1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )

    with pytest.raises(InductiveValidationError):
        PiModelMethod(_make_spec(PiModelMethod, bundle, unsup_warm_up=-0.1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )

    with pytest.raises(InductiveValidationError):
        MeanTeacherMethod(_make_spec(MeanTeacherMethod, bundle, ema_decay=1.5)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        MeanTeacherMethod(_make_spec(MeanTeacherMethod, bundle, unsup_warm_up=-0.1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )

    with pytest.raises(InductiveValidationError):
        FlexMatchMethod(_make_spec(FlexMatchMethod, bundle, temperature=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )


@pytest.mark.parametrize("method_cls", DEEP_METHODS)
def test_deep_methods_predict_proba_backend_mismatch(method_cls):
    data = _make_flex_data() if method_cls is FlexMatchMethod else make_torch_ssl_dataset()
    bundle = make_model_bundle()
    spec = _make_spec(method_cls, bundle)
    method = method_cls(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    with pytest.raises(InductiveValidationError):
        method.predict_proba(make_numpy_dataset().X_l)


def test_pi_model_fit_predict_variants():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()

    method = PiModelMethod(
        PiModelSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            unsup_warm_up=0.0,
            freeze_bn=False,
            detach_target=True,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    method2 = PiModelMethod(
        PiModelSpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=1,
            unsup_warm_up=0.5,
            freeze_bn=True,
            detach_target=False,
        )
    )
    _fit_predict(method2, data)


def test_mean_teacher_fit_predict_and_errors():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    method = MeanTeacherMethod(
        MeanTeacherSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            unsup_warm_up=0.0,
            freeze_bn=True,
            detach_target=False,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    bundle2 = bundle2.__class__(model=bundle2.model, optimizer=bundle2.optimizer, ema_model=None)
    with pytest.raises(InductiveValidationError):
        MeanTeacherMethod(MeanTeacherSpec(model_bundle=bundle2)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )

    mt = MeanTeacherMethod(MeanTeacherSpec(model_bundle=make_model_bundle()))
    with pytest.raises(InductiveValidationError):
        mt._check_teacher(mt.spec.model_bundle.model, mt.spec.model_bundle.model)


def test_fixmatch_fit_predict_variants_and_sharpen():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    method = FixMatchMethod(
        FixMatchSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            use_cat=True,
            hard_label=True,
            detach_target=True,
            temperature=1.0,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    method2 = FixMatchMethod(
        FixMatchSpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=1,
            use_cat=False,
            hard_label=False,
            detach_target=False,
            temperature=0.5,
            p_cutoff=1.0,
        )
    )
    _fit_predict(method2, data)

    probs = torch.tensor([[0.4, 0.6]])
    assert torch.allclose(fixmatch._sharpen(probs, temperature=1.0), probs)
    with pytest.raises(InductiveValidationError):
        fixmatch._sharpen(probs, temperature=0.0)


def test_flexmatch_fit_predict_variants_and_meta_helpers():
    data = _make_flex_data()
    bundle = make_model_bundle()
    method = FlexMatchMethod(
        FlexMatchSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            use_cat=True,
            hard_label=True,
            detach_target=True,
            temperature=1.0,
            p_cutoff=0.0,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    method2 = FlexMatchMethod(
        FlexMatchSpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=1,
            use_cat=False,
            hard_label=False,
            detach_target=False,
            temperature=0.5,
            p_cutoff=1.0,
            thresh_warmup=False,
        )
    )
    _fit_predict(method2, data)

    probs = torch.tensor([[0.4, 0.6]])
    assert torch.allclose(flexmatch._sharpen(probs, temperature=1.0), probs)

    fm = FlexMatchMethod(FlexMatchSpec())
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(
            DummyDataset(
                X_l=data.X_l,
                y_l=data.y_l,
                X_u=data.X_u,
                X_u_w=data.X_u_w,
                X_u_s=data.X_u_s,
                meta=None,
            ),
            device=data.X_u_w.device,
            n_u=2,
        )

    bad_meta = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": torch.tensor([0, 1], dtype=torch.int32)},
    )
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(bad_meta, device=data.X_u_w.device, n_u=2)

    fm._ulb_size = int(data.X_u_w.shape[0])
    fm._init_state(n_classes=2, device=data.X_u_w.device)
    fm.spec = FlexMatchSpec(thresh_warmup=False)
    fm._update_classwise_acc()


def test_uda_fit_predict_and_tsa_threshold():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    method = UDAMethod(
        UDASpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            use_cat=True,
            detach_target=True,
            tsa_schedule="linear",
            temperature=1.0,
            p_cutoff=0.0,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    method2 = UDAMethod(
        UDASpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=1,
            use_cat=False,
            detach_target=False,
            tsa_schedule="none",
            temperature=0.5,
            p_cutoff=1.0,
        )
    )
    _fit_predict(method2, data)

    assert uda._tsa_threshold("none", step=0, total=1, n_classes=2) == 1.0
    assert uda._tsa_threshold("linear", step=0, total=0, n_classes=2) == 1.0
    assert uda._tsa_threshold("exp", step=1, total=2, n_classes=2) > 0.0
    assert uda._tsa_threshold("log", step=1, total=2, n_classes=2) > 0.0
    with pytest.raises(InductiveValidationError):
        uda._tsa_threshold("bad", step=0, total=1, n_classes=2)


def test_mixmatch_fit_predict_mixup_helpers():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    method = MixMatchMethod(
        MixMatchSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            mixup_manifold=False,
            freeze_bn=False,
            temperature=1.0,
            unsup_warm_up=0.0,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    method2 = MixMatchMethod(
        MixMatchSpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=1,
            mixup_manifold=True,
            freeze_bn=True,
            temperature=0.5,
            unsup_warm_up=0.5,
        )
    )
    _fit_predict(method2, data)

    X = torch.zeros((2, 2))
    y = torch.zeros((2, 2))
    mixmatch._mixup(X, y, alpha=0.0, generator=torch.Generator().manual_seed(0))
    with pytest.raises(InductiveValidationError):
        mixmatch._mixup(X[:0], y[:0], alpha=0.5, generator=torch.Generator())
    with pytest.raises(InductiveValidationError):
        mixmatch._mixup(
            torch.zeros((2, 2)), torch.zeros((3, 2)), alpha=0.5, generator=torch.Generator()
        )

    bundle3 = make_model_bundle()
    feat = torch.zeros((2, 2))
    out = mixmatch._forward_head(bundle3, features=feat)
    assert isinstance(out, torch.Tensor)

    class _NoOnlyFc(torch.nn.Module):
        def forward(self, x):
            return x

    bad_bundle = bundle3.__class__(model=_NoOnlyFc(), optimizer=bundle3.optimizer)
    with pytest.raises(InductiveValidationError):
        mixmatch._forward_head(bad_bundle, features=feat)


def test_mixmatch_mixup_randperm_uses_cpu(monkeypatch):
    X = torch.zeros((4, 2))
    y = torch.zeros((4, 2))
    calls = {}
    orig_randperm = torch.randperm

    def _spy_randperm(n, *args, **kwargs):
        calls["device"] = kwargs.get("device")
        calls["generator"] = kwargs.get("generator")
        return orig_randperm(n, generator=kwargs.get("generator"), device="cpu")

    monkeypatch.setattr(torch, "randperm", _spy_randperm)
    mixmatch._mixup(X, y, alpha=0.5, generator=torch.Generator().manual_seed(0))

    assert str(calls["device"]) == "cpu"
    assert calls["generator"] is not None


def test_mixmatch_mixup_moves_perm_to_device(monkeypatch):
    class _FakeCudaTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda")

    X = torch.zeros((4, 2)).as_subclass(_FakeCudaTensor)
    y = torch.zeros((4, 2))
    calls = {"to_devices": []}
    orig_ones = torch.ones
    orig_to = torch.Tensor.to

    def _spy_ones(*args, **kwargs):
        if "device" in kwargs:
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
        return orig_ones(*args, **kwargs)

    def _spy_to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args:
            device = args[0]
        if device is not None:
            calls["to_devices"].append(device)
        if "device" in kwargs:
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
        elif args:
            new_args = list(args)
            new_args[0] = torch.device("cpu")
            args = tuple(new_args)
        return orig_to(self, *args, **kwargs)

    monkeypatch.setattr(torch, "ones", _spy_ones)
    monkeypatch.setattr(torch.Tensor, "to", _spy_to, raising=False)

    mixmatch._mixup(X, y, alpha=0.0, generator=torch.Generator().manual_seed(0))

    assert any(str(device) == "cuda" for device in calls["to_devices"])


def test_adamatch_fit_predict_and_alignment():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    method = AdaMatchMethod(
        AdaMatchSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            use_cat=True,
            hard_label=True,
            detach_target=True,
            temperature=1.0,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    method2 = AdaMatchMethod(
        AdaMatchSpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=1,
            use_cat=False,
            hard_label=False,
            detach_target=False,
            temperature=0.5,
            p_cutoff=1.0,
        )
    )
    _fit_predict(method2, data)

    probs_u = torch.tensor([[0.4, 0.6]])
    probs_l = torch.tensor([[0.6, 0.4]])
    aligned = method2._update_alignment(probs_u, probs_l)
    assert aligned.shape == probs_u.shape


def test_free_match_fit_predict_and_state_helpers():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    method = FreeMatchMethod(
        FreeMatchSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            use_cat=True,
            hard_label=True,
            detach_target=True,
            temperature=1.0,
            use_quantile=True,
            clip_thresh=True,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    method2 = FreeMatchMethod(
        FreeMatchSpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=1,
            use_cat=False,
            hard_label=False,
            detach_target=False,
            temperature=0.5,
            use_quantile=False,
            clip_thresh=False,
        )
    )
    _fit_predict(method2, data)

    probs_u = torch.tensor([[0.5, 0.5]])
    max_probs = torch.tensor([0.5])
    max_idx = torch.tensor([0])
    method2._update_state(probs_u, max_probs, max_idx)
    ent = method2._entropy_loss(torch.tensor([[0.1, 0.2]]), method2._label_hist, method2._p_model)
    assert ent.ndim == 0


def test_softmatch_fit_predict_and_stats():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    method = SoftMatchMethod(
        SoftMatchSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            use_cat=True,
            hard_label=True,
            detach_target=True,
            temperature=1.0,
            per_class=False,
            dist_align=True,
            dist_uniform=True,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    method2 = SoftMatchMethod(
        SoftMatchSpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=1,
            use_cat=False,
            hard_label=False,
            detach_target=False,
            temperature=0.5,
            per_class=True,
            dist_align=False,
            dist_uniform=False,
        )
    )
    _fit_predict(method2, data)

    probs_u = torch.tensor([[0.4, 0.6]])
    probs_l = torch.tensor([[0.6, 0.4]])
    aligned = method._dist_align(probs_u, probs_l)
    assert aligned.shape == probs_u.shape

    method3 = SoftMatchMethod(
        SoftMatchSpec(
            model_bundle=make_model_bundle(),
            dist_align=True,
            dist_uniform=False,
        )
    )
    assert method3._dist_align(probs_u, probs_l).shape == probs_u.shape
    method4 = SoftMatchMethod(
        SoftMatchSpec(
            model_bundle=make_model_bundle(),
            dist_align=False,
        )
    )
    assert torch.allclose(method4._dist_align(probs_u, probs_l), probs_u)

    method2._init_stats(n_classes=2, device=probs_u.device)
    method2._update_stats(torch.tensor([0.5, 0.6]), torch.tensor([0, 1]))
    method2._update_stats(torch.tensor([0.5]), torch.tensor([0]))


@pytest.mark.parametrize(
    "module",
    [fixmatch, flexmatch, mixmatch, adamatch, free_match, softmatch],
)
def test_sharpen_errors(module):
    probs = torch.tensor([[0.4, 0.6]])
    with pytest.raises(InductiveValidationError):
        module._sharpen(probs, temperature=0.0)


@pytest.mark.parametrize("method_cls", DEEP_METHODS)
def test_deep_methods_y_l_int32_rejected(method_cls):
    data = _data_for_method(method_cls)
    bundle = make_model_bundle()
    spec = _make_spec(method_cls, bundle)
    bad = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l.to(torch.int32),
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta=getattr(data, "meta", None),
    )
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(bad, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize("method_cls", DEEP_METHODS)
def test_deep_methods_backend_guard(method_cls):
    data = _data_for_method(method_cls)
    bundle = make_model_bundle()
    spec = _make_spec(method_cls, bundle)
    method = method_cls(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    method._backend = None
    with pytest.raises(InductiveValidationError):
        method.predict_proba(make_numpy_dataset().X_l)


@pytest.mark.parametrize("method_cls", DEEP_METHODS)
def test_deep_methods_predict_proba_bad_logits(method_cls):
    data = _data_for_method(method_cls)
    bundle = make_model_bundle()
    spec = _make_spec(method_cls, bundle)
    method = method_cls(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    bad_model = _BadLogits1D()
    method._bundle = _make_bundle_for(bad_model, with_ema=method_cls is MeanTeacherMethod)
    with pytest.raises(InductiveValidationError):
        method.predict_proba(data.X_l)


@pytest.mark.parametrize("method_cls", DEEP_METHODS)
def test_deep_methods_predict_proba_eval_mode(method_cls):
    data = _data_for_method(method_cls)
    bundle = make_model_bundle()
    spec = _make_spec(method_cls, bundle)
    method = method_cls(spec)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    method._bundle.model.eval()
    if method._bundle.ema_model is not None:
        method._bundle.ema_model.eval()
    proba = method.predict_proba(data.X_l)
    assert int(proba.shape[0]) == int(data.X_l.shape[0])


@pytest.mark.parametrize("method_cls", DEEP_METHODS)
def test_deep_methods_empty_xl_hits_check(method_cls, monkeypatch):
    data = make_torch_ssl_dataset()
    empty = DummyDataset(
        X_l=torch.zeros((0, data.X_l.shape[1])),
        y_l=torch.zeros((0,), dtype=torch.int64),
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
    )
    module = DEEP_METHOD_MODULES[method_cls]
    monkeypatch.setattr(f"{module}.ensure_1d_labels_torch", lambda y, name="y_l": y)
    bundle = make_model_bundle()
    spec = _make_spec(method_cls, bundle)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(empty, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    "method_cls",
    [
        FixMatchMethod,
        FlexMatchMethod,
        UDAMethod,
        AdaMatchMethod,
        FreeMatchMethod,
        SoftMatchMethod,
        MixMatchMethod,
    ],
)
def test_deep_methods_xu_mismatch_hits_check(method_cls, monkeypatch):
    data = make_torch_ssl_dataset()
    mismatch = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=torch.zeros((2, data.X_l.shape[1])),
        X_u_s=torch.zeros((3, data.X_l.shape[1])),
    )
    module = DEEP_METHOD_MODULES[method_cls]
    monkeypatch.setattr(f"{module}.ensure_torch_data", lambda data, device: data)
    bundle = make_model_bundle()
    spec = _make_spec(method_cls, bundle)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(mismatch, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize("method_cls", CAT_METHODS)
def test_use_cat_bad_logits_ndim(method_cls):
    data = _data_for_method(method_cls)
    bundle = _make_bundle_for(_BadLogits1D())
    spec = _make_spec(method_cls, bundle, use_cat=True)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize("method_cls", CAT_METHODS)
def test_use_cat_bad_concat_batch(method_cls):
    data = _data_for_method(method_cls)
    bundle = _make_bundle_for(_BadBatch())
    spec = _make_spec(method_cls, bundle, use_cat=True)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize("method_cls", CAT_METHODS + [PiModelMethod, MeanTeacherMethod])
def test_non_use_cat_bad_logits_ndim(method_cls):
    data = _data_for_method(method_cls)
    bundle = _make_bundle_for(_BadLogits1D(), with_ema=method_cls is MeanTeacherMethod)
    spec = _make_spec(method_cls, bundle)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize("method_cls", CAT_METHODS + [PiModelMethod, MeanTeacherMethod])
def test_non_use_cat_unlabeled_shape_mismatch(method_cls):
    data = _data_for_method(method_cls)
    data_mismatch = DummyDataset(
        X_l=torch.ones_like(data.X_l),
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=torch.ones_like(data.X_u_w),
        X_u_s=-torch.ones_like(data.X_u_s),
        meta=getattr(data, "meta", None),
    )
    bundle = _make_bundle_for(_ConditionalClasses(), with_ema=method_cls is MeanTeacherMethod)
    spec = _make_spec(method_cls, bundle)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(data_mismatch, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize("method_cls", CAT_METHODS + [PiModelMethod, MeanTeacherMethod])
def test_non_use_cat_class_dim_mismatch(method_cls):
    data = _data_for_method(method_cls)
    data_mismatch = DummyDataset(
        X_l=torch.ones_like(data.X_l),
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=-torch.ones_like(data.X_u_w),
        X_u_s=-torch.ones_like(data.X_u_s),
        meta=getattr(data, "meta", None),
    )
    bundle = _make_bundle_for(_ConditionalClasses(), with_ema=method_cls is MeanTeacherMethod)
    spec = _make_spec(method_cls, bundle)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(data_mismatch, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize("method_cls", CAT_METHODS + [PiModelMethod, MeanTeacherMethod])
def test_non_use_cat_y_l_range_error(method_cls):
    data = _data_for_method(method_cls)
    bad = DummyDataset(
        X_l=data.X_l,
        y_l=torch.tensor([0, 2, 2, 0], dtype=torch.int64),
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta=getattr(data, "meta", None),
    )
    bundle = make_model_bundle()
    spec = _make_spec(method_cls, bundle)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(bad, device=DeviceSpec(device="cpu"), seed=0)


def test_mixmatch_mixup_non_torch():
    with pytest.raises(InductiveValidationError):
        mixmatch._mixup([[0.0, 1.0]], [[1.0, 0.0]], alpha=0.5, generator=torch.Generator())


def test_mixmatch_forward_head_meta():
    bundle = make_model_bundle()
    features = torch.ones((2, 2))

    head_bundle = TorchModelBundle(
        model=bundle.model,
        optimizer=bundle.optimizer,
        ema_model=bundle.ema_model,
        meta={"forward_head": lambda x: x + 1.0},
    )
    out = mixmatch._forward_head(head_bundle, features=features)
    assert torch.allclose(out, features + 1.0)

    raw_bundle = TorchModelBundle(
        model=bundle.model,
        optimizer=bundle.optimizer,
        ema_model=bundle.ema_model,
        meta=["not-a-mapping"],
    )
    out2 = mixmatch._forward_head(raw_bundle, features=features)
    assert int(out2.shape[0]) == int(features.shape[0])


def test_mixmatch_logits_errors():
    data = make_torch_ssl_dataset()
    bundle = _make_bundle_for(_ConditionalClasses())
    spec = MixMatchSpec(model_bundle=bundle, batch_size=2, max_epochs=1)
    mismatch = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=torch.ones_like(data.X_u_w),
        X_u_s=-torch.ones_like(data.X_u_s),
    )
    with pytest.raises(InductiveValidationError):
        MixMatchMethod(spec).fit(mismatch, device=DeviceSpec(device="cpu"), seed=0)

    bad_logits = _make_bundle_for(_BadLogits1D())
    spec_bad = MixMatchSpec(model_bundle=bad_logits, batch_size=2, max_epochs=1)
    with pytest.raises(InductiveValidationError):
        MixMatchMethod(spec_bad).fit(data, device=DeviceSpec(device="cpu"), seed=0)

    bad_labels = DummyDataset(
        X_l=data.X_l,
        y_l=torch.tensor([0, 2, 2, 0], dtype=torch.int64),
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
    )
    with pytest.raises(InductiveValidationError):
        MixMatchMethod(_make_spec(MixMatchMethod, make_model_bundle())).fit(
            bad_labels, device=DeviceSpec(device="cpu"), seed=0
        )

    logits_all_bad = _make_bundle_for(_LogitsByBatch())
    spec_all = MixMatchSpec(model_bundle=logits_all_bad, batch_size=2, max_epochs=1)
    with pytest.raises(InductiveValidationError):
        MixMatchMethod(spec_all).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_mean_teacher_check_teacher_mismatches():
    mt = MeanTeacherMethod(MeanTeacherSpec(model_bundle=make_model_bundle()))
    student = torch.nn.Linear(2, 2, bias=False)
    teacher = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    with pytest.raises(InductiveValidationError):
        mt._check_teacher(student, teacher)

    teacher2 = torch.nn.Linear(3, 2, bias=False)
    with pytest.raises(InductiveValidationError):
        mt._check_teacher(student, teacher2)

    teacher3 = torch.nn.Linear(2, 2, bias=False).to(device="meta")
    with pytest.raises(InductiveValidationError):
        mt._check_teacher(student, teacher3)


def test_flexmatch_meta_validation_and_state_branches():
    data = make_torch_ssl_dataset()
    fm = FlexMatchMethod(FlexMatchSpec())

    with pytest.raises(InductiveValidationError):
        fm._init_state(n_classes=2, device=data.X_l.device)
    with pytest.raises(InductiveValidationError):
        fm._update_classwise_acc()

    bad_meta = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta=["bad"],
    )
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(bad_meta, device=data.X_u_w.device, n_u=int(data.X_u_w.shape[0]))

    missing_idx = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={},
    )
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(missing_idx, device=data.X_u_w.device, n_u=int(data.X_u_w.shape[0]))

    alt_idx = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"unlabeled_idx": torch.arange(4, dtype=torch.int64)},
    )
    out_alt = fm._get_idx_u(alt_idx, device=data.X_u_w.device, n_u=4)
    assert int(out_alt.numel()) == 4

    alt_idx2 = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"unlabeled_indices": torch.arange(4, dtype=torch.int64)},
    )
    out_alt2 = fm._get_idx_u(alt_idx2, device=data.X_u_w.device, n_u=4)
    assert int(out_alt2.numel()) == 4

    not_tensor = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": [0, 1, 2, 3]},
    )
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(not_tensor, device=data.X_u_w.device, n_u=int(data.X_u_w.shape[0]))

    bad_ndim = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": torch.zeros((1, 1), dtype=torch.int64)},
    )
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(bad_ndim, device=data.X_u_w.device, n_u=int(data.X_u_w.shape[0]))

    bad_size = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": torch.tensor([0], dtype=torch.int64)},
    )
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(bad_size, device=data.X_u_w.device, n_u=int(data.X_u_w.shape[0]))

    bad_device = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="meta")},
    )
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(bad_device, device=data.X_u_w.device, n_u=int(data.X_u_w.shape[0]))

    non_contig = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": torch.tensor([1, 2, 3, 4], dtype=torch.int64)},
    )
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(non_contig, device=data.X_u_w.device, n_u=4)

    dup_idx = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": torch.tensor([0, 2, 2], dtype=torch.int64)},
    )
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(dup_idx, device=data.X_u_w.device, n_u=3)

    ok_idx = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": torch.tensor([0, 1, 2], dtype=torch.int64)},
    )
    idx_out = fm._get_idx_u(ok_idx, device=data.X_u_w.device, n_u=3)
    assert int(idx_out.numel()) == 3

    bad_ulb = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": torch.tensor([0, 1], dtype=torch.int64), "ulb_size": 1.5},
    )
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(bad_ulb, device=data.X_u_w.device, n_u=2)

    small_ulb = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": torch.tensor([0, 1], dtype=torch.int64), "ulb_size": 1},
    )
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(small_ulb, device=data.X_u_w.device, n_u=2)

    overflow_ulb = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": torch.tensor([0, 2], dtype=torch.int64), "ulb_size": 2},
    )
    with pytest.raises(InductiveValidationError):
        fm._get_idx_u(overflow_ulb, device=data.X_u_w.device, n_u=2)

    fm._ulb_size = 2
    fm._init_state(n_classes=2, device=data.X_l.device)
    fm._selected_label[0] = 0
    fm.spec = FlexMatchSpec(thresh_warmup=False)
    fm._update_classwise_acc()


def test_free_match_state_and_entropy_branches(monkeypatch):
    fm = FreeMatchMethod(FreeMatchSpec(use_quantile=False))
    probs_u = torch.zeros((0, 2))
    max_probs = torch.tensor([], dtype=torch.float32)
    max_idx = torch.tensor([], dtype=torch.int64)
    fm._update_state(probs_u, max_probs, max_idx)
    ent = fm._entropy_loss(torch.zeros((0, 2)), fm._label_hist, fm._p_model)
    assert ent.ndim == 0

    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    method = FreeMatchMethod(
        FreeMatchSpec(model_bundle=bundle, batch_size=2, max_epochs=1, clip_thresh=False)
    )

    def _force_state(self, probs_u, max_probs, max_idx):
        self._p_model = torch.ones((int(probs_u.shape[1]),), device=probs_u.device)
        self._label_hist = torch.ones((int(probs_u.shape[1]),), device=probs_u.device)
        self._time_p = torch.tensor(1.0, device=probs_u.device)

    monkeypatch.setattr(FreeMatchMethod, "_update_state", _force_state)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_softmatch_update_stats_branches():
    sm = SoftMatchMethod(SoftMatchSpec())
    with pytest.raises(InductiveValidationError):
        sm._update_stats(torch.tensor([0.5]), torch.tensor([0]))

    sm.spec = SoftMatchSpec(per_class=True)
    sm._init_stats(n_classes=2, device=torch.device("cpu"))
    max_probs = torch.tensor([0.5, 0.6, 0.7, 0.8])
    max_idx = torch.tensor([0, 0, 1, 1])
    sm._update_stats(max_probs, max_idx)


def test_softmatch_invalid_n_sigma():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    with pytest.raises(InductiveValidationError):
        SoftMatchMethod(SoftMatchSpec(model_bundle=bundle, n_sigma=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )


def test_fixmatch_empty_unlabeled_batch(monkeypatch):
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    spec = FixMatchSpec(model_bundle=bundle, batch_size=2, max_epochs=1)
    method = FixMatchMethod(spec)

    def fake_cycle_batch_indices(n, *, batch_size, generator, device, steps):
        for _ in range(int(steps)):
            yield torch.tensor([], dtype=torch.long, device=device)

    monkeypatch.setattr(
        "modssc.inductive.methods.fixmatch.cycle_batch_indices", fake_cycle_batch_indices
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
