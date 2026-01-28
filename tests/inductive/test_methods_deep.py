from __future__ import annotations

import copy
import sys
import types

import pytest
import torch

from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods import (
    adamatch,
    adsh,
    defixmatch,
    fixmatch,
    flexmatch,
    free_match,
    mixmatch,
    softmatch,
    uda,
    vat,
)
from modssc.inductive.methods.adamatch import AdaMatchMethod, AdaMatchSpec
from modssc.inductive.methods.adsh import ADSHMethod, ADSHSpec
from modssc.inductive.methods.defixmatch import DeFixMatchMethod, DeFixMatchSpec
from modssc.inductive.methods.fixmatch import FixMatchMethod, FixMatchSpec
from modssc.inductive.methods.flexmatch import FlexMatchMethod, FlexMatchSpec
from modssc.inductive.methods.free_match import FreeMatchMethod, FreeMatchSpec
from modssc.inductive.methods.mean_teacher import MeanTeacherMethod, MeanTeacherSpec
from modssc.inductive.methods.mixmatch import MixMatchMethod, MixMatchSpec
from modssc.inductive.methods.noisy_student import NoisyStudentMethod, NoisyStudentSpec
from modssc.inductive.methods.pi_model import PiModelMethod, PiModelSpec
from modssc.inductive.methods.softmatch import SoftMatchMethod, SoftMatchSpec
from modssc.inductive.methods.temporal_ensembling import (
    TemporalEnsemblingMethod,
    TemporalEnsemblingSpec,
)
from modssc.inductive.methods.uda import UDAMethod, UDASpec
from modssc.inductive.methods.vat import VATMethod, VATSpec
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
    TemporalEnsemblingMethod,
    DeFixMatchMethod,
    FixMatchMethod,
    ADSHMethod,
    FlexMatchMethod,
    UDAMethod,
    MixMatchMethod,
    AdaMatchMethod,
    FreeMatchMethod,
    SoftMatchMethod,
    VATMethod,
    NoisyStudentMethod,
]

DICT_METHODS = [method_cls for method_cls in DEEP_METHODS if method_cls is not ADSHMethod]


DEEP_METHOD_MODULES = {
    PiModelMethod: "modssc.inductive.methods.pi_model",
    MeanTeacherMethod: "modssc.inductive.methods.mean_teacher",
    TemporalEnsemblingMethod: "modssc.inductive.methods.temporal_ensembling",
    DeFixMatchMethod: "modssc.inductive.methods.defixmatch",
    FixMatchMethod: "modssc.inductive.methods.fixmatch",
    ADSHMethod: "modssc.inductive.methods.adsh",
    FlexMatchMethod: "modssc.inductive.methods.flexmatch",
    UDAMethod: "modssc.inductive.methods.uda",
    MixMatchMethod: "modssc.inductive.methods.mixmatch",
    AdaMatchMethod: "modssc.inductive.methods.adamatch",
    FreeMatchMethod: "modssc.inductive.methods.free_match",
    SoftMatchMethod: "modssc.inductive.methods.softmatch",
    VATMethod: "modssc.inductive.methods.vat",
    NoisyStudentMethod: "modssc.inductive.methods.noisy_student",
}


CAT_METHODS = [
    DeFixMatchMethod,
    FixMatchMethod,
    ADSHMethod,
    FlexMatchMethod,
    UDAMethod,
    AdaMatchMethod,
    FreeMatchMethod,
    SoftMatchMethod,
    NoisyStudentMethod,
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
    if method_cls is TemporalEnsemblingMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return TemporalEnsemblingSpec(**kwargs)
    if method_cls is DeFixMatchMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return DeFixMatchSpec(**kwargs)
    if method_cls is FixMatchMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return FixMatchSpec(**kwargs)
    if method_cls is ADSHMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return ADSHSpec(**kwargs)
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
    if method_cls is VATMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return VATSpec(**kwargs)
    if method_cls is NoisyStudentMethod:
        kwargs = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1}
        kwargs.update(overrides)
        return NoisyStudentSpec(**kwargs)
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


class _DictModel(torch.nn.Module):
    def __init__(self, in_dim: int = 2, n_classes: int = 2) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, n_classes, bias=False)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        return self.fc(x)


class _BadLogits1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros((int(x.shape[0]),), device=x.device)


class _SafeLabels(torch.Tensor):
    @staticmethod
    def __new__(cls, base):
        return torch.Tensor._make_subclass(cls, base, base.requires_grad)

    def min(self, *args, **kwargs):
        if int(self.numel()) == 0:
            return torch.tensor(0, device=self.device, dtype=self.dtype)
        return super().min(*args, **kwargs)

    def max(self, *args, **kwargs):
        if int(self.numel()) == 0:
            return torch.tensor(0, device=self.device, dtype=self.dtype)
        return super().max(*args, **kwargs)


class _BadBatch(torch.nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.n_classes = int(n_classes)

    def forward(self, x):
        batch = max(0, int(x.shape[0]) - 1)
        return torch.zeros((batch, self.n_classes), device=x.device)


class _BadBatchOnUnlabeled(torch.nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.n_classes = int(n_classes)

    def forward(self, x):
        batch = int(x.shape[0])
        if float(x.sum().item()) <= 0.0:
            return torch.zeros((batch, self.n_classes), device=x.device)
        return torch.zeros((max(0, batch - 1), self.n_classes), device=x.device)


class _BadBatchOnLabeled(torch.nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.n_classes = int(n_classes)

    def forward(self, x):
        batch = int(x.shape[0])
        if float(x.sum().item()) <= 0.0:
            return torch.zeros((max(0, batch - 1), self.n_classes), device=x.device)
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


class _GradSensitiveLogits(torch.nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.n_classes = int(n_classes)

    def forward(self, x):
        batch = int(x.shape[0])
        if x.requires_grad:
            return torch.zeros((batch,), device=x.device)
        return torch.zeros((batch, self.n_classes), device=x.device)


class _CountedLogits(torch.nn.Module):
    def __init__(self, in_dim: int = 2, n_classes: int = 2, *, bad_call: int = 3):
        super().__init__()
        self.fc = torch.nn.Linear(int(in_dim), int(n_classes), bias=False)
        self.bad_call = int(bad_call)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        logits = self.fc(x)
        if self.calls >= self.bad_call:
            return logits[:, 0]
        return logits


class _LinearLogits(torch.nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.fc = torch.nn.Linear(2, n_classes, bias=False)

    def forward(self, x):
        return self.fc(x)


class _TeacherLogits1D(_LinearLogits):
    def forward(self, x):
        logits = super().forward(x)
        return logits[:, 0]


class _TeacherLogitsTrunc(_LinearLogits):
    def forward(self, x):
        logits = super().forward(x)
        return logits[:, :1]


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

    if method_cls is VATMethod:
        empty_u = DummyDataset(
            X_l=data.X_l,
            y_l=data.y_l,
            X_u=torch.zeros((0, data.X_l.shape[1])),
            X_u_w=data.X_u_w,
            X_u_s=data.X_u_s,
        )
    else:
        empty_u = DummyDataset(
            X_l=data.X_l,
            y_l=data.y_l,
            X_u=data.X_u,
            X_u_w=torch.zeros((0, data.X_l.shape[1])),
            X_u_s=torch.zeros((0, data.X_l.shape[1])),
        )
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(empty_u, device=DeviceSpec(device="cpu"), seed=0)

    if method_cls is VATMethod:
        mismatch = DummyDataset(
            X_l=data.X_l,
            y_l=data.y_l,
            X_u=torch.zeros((0, data.X_l.shape[1])),
            X_u_w=data.X_u_w,
            X_u_s=data.X_u_s,
        )
    else:
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
        ADSHMethod(_make_spec(ADSHMethod, bundle, p_cutoff=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        ADSHMethod(_make_spec(ADSHMethod, bundle, score_warmup_epochs=-1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        DeFixMatchMethod(_make_spec(DeFixMatchMethod, bundle, p_cutoff=1.5)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        DeFixMatchMethod(_make_spec(DeFixMatchMethod, bundle, temperature=0.0)).fit(
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
        TemporalEnsemblingMethod(_make_spec(TemporalEnsemblingMethod, bundle, alpha=1.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        TemporalEnsemblingMethod(
            _make_spec(TemporalEnsemblingMethod, bundle, unsup_warm_up=-0.1)
        ).fit(data, device=DeviceSpec(device="cpu"), seed=0)

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

    with pytest.raises(InductiveValidationError):
        VATMethod(_make_spec(VATMethod, bundle, xi=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        VATMethod(_make_spec(VATMethod, bundle, eps=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        VATMethod(_make_spec(VATMethod, bundle, num_iters=0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        VATMethod(_make_spec(VATMethod, bundle, unsup_warm_up=-0.1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )

    with pytest.raises(InductiveValidationError):
        NoisyStudentMethod(_make_spec(NoisyStudentMethod, bundle, p_cutoff=-0.1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        NoisyStudentMethod(_make_spec(NoisyStudentMethod, bundle, temperature=0.0)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        NoisyStudentMethod(_make_spec(NoisyStudentMethod, bundle, teacher_epochs=-1)).fit(
            data, device=DeviceSpec(device="cpu"), seed=0
        )
    with pytest.raises(InductiveValidationError):
        NoisyStudentMethod(_make_spec(NoisyStudentMethod, bundle, unsup_warm_up=-0.1)).fit(
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


def test_temporal_ensembling_fit_predict_variants():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()

    method = TemporalEnsemblingMethod(
        TemporalEnsemblingSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            unsup_warm_up=0.0,
            alpha=0.0,
            freeze_bn=False,
            detach_target=True,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    method2 = TemporalEnsemblingMethod(
        TemporalEnsemblingSpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=2,
            unsup_warm_up=0.5,
            alpha=0.6,
            freeze_bn=True,
            detach_target=False,
        )
    )
    _fit_predict(method2, data)


def test_temporal_ensembling_unlabeled_logits_dim_error():
    data = make_torch_ssl_dataset()
    data_grad = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w.clone().requires_grad_(True),
        X_u_s=data.X_u_s.clone().requires_grad_(True),
    )
    bundle = _make_bundle_for(_GradSensitiveLogits())
    spec = TemporalEnsemblingSpec(model_bundle=bundle, batch_size=2, max_epochs=1)
    with pytest.raises(InductiveValidationError, match="Model logits must be 2D"):
        TemporalEnsemblingMethod(spec).fit(data_grad, device=DeviceSpec(device="cpu"), seed=0)


def test_temporal_ensembling_targets_not_initialized(monkeypatch):
    data = make_torch_ssl_dataset()
    bundle = _make_bundle_for(_LinearLogits())
    spec = TemporalEnsemblingSpec(model_bundle=bundle, batch_size=2, max_epochs=1)

    def _zeros_like(*_args, **_kwargs):
        return None

    monkeypatch.setattr(torch, "zeros_like", _zeros_like)
    with pytest.raises(InductiveValidationError, match="targets not initialized"):
        TemporalEnsemblingMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_temporal_ensembling_predictions_not_initialized(monkeypatch):
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    spec = TemporalEnsemblingSpec(model_bundle=bundle, batch_size=2, max_epochs=1)

    def _empty_iter(*_args, **_kwargs):
        return iter(())

    monkeypatch.setattr("modssc.inductive.methods.temporal_ensembling.cycle_batches", _empty_iter)
    monkeypatch.setattr(
        "modssc.inductive.methods.temporal_ensembling.cycle_batch_indices", _empty_iter
    )
    with pytest.raises(InductiveValidationError, match="predictions not initialized"):
        TemporalEnsemblingMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


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


def test_defixmatch_fit_predict_variants_and_views():
    data = make_torch_ssl_dataset()
    views = {"X_l_s": data.X_l + 0.01}
    ds = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        views=views,
    )
    bundle = make_model_bundle()
    method = DeFixMatchMethod(
        DeFixMatchSpec(
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
    _fit_predict(method, ds)

    bundle2 = make_model_bundle()
    method2 = DeFixMatchMethod(
        DeFixMatchSpec(
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
    assert torch.allclose(defixmatch._sharpen(probs, temperature=1.0), probs)


def test_defixmatch_predict_proba_requires_tensor():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    method = DeFixMatchMethod(DeFixMatchSpec(model_bundle=bundle, batch_size=2, max_epochs=1))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)
    with pytest.raises(InductiveValidationError, match="predict_proba requires torch.Tensor"):
        method.predict_proba(data.X_l.cpu().numpy())


@pytest.mark.parametrize(
    ("view_factory", "match"),
    [
        (
            lambda d: torch.zeros((int(d.X_l.shape[0]),), dtype=d.X_l.dtype),
            "X_l_s must be at least 2D",
        ),
        (
            lambda d: torch.zeros(
                (int(d.X_l.shape[0]) - 1, int(d.X_l.shape[1])), dtype=d.X_l.dtype
            ),
            "same number of rows",
        ),
        (
            lambda d: torch.zeros(
                (int(d.X_l.shape[0]), int(d.X_l.shape[1]) + 1), dtype=d.X_l.dtype
            ),
            "same feature dimension",
        ),
    ],
)
def test_defixmatch_labeled_strong_view_shape_errors(view_factory, match):
    data = make_torch_ssl_dataset()
    views = {"X_l_s": view_factory(data)}
    ds = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        views=views,
    )
    method = DeFixMatchMethod(
        DeFixMatchSpec(model_bundle=make_model_bundle(), batch_size=2, max_epochs=1)
    )
    with pytest.raises(InductiveValidationError, match=match):
        method.fit(ds, device=DeviceSpec(device="cpu"), seed=0)


def test_defixmatch_views_without_strong_key_uses_default():
    data = make_torch_ssl_dataset()
    views = {"unused": data.X_l + 0.01}
    ds = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        views=views,
    )
    method = DeFixMatchMethod(
        DeFixMatchSpec(model_bundle=make_model_bundle(), batch_size=2, max_epochs=1)
    )
    _fit_predict(method, ds)


def test_defixmatch_use_cat_labeled_strong_batch_mismatch():
    class _WeirdTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, base):
            return torch.Tensor._make_subclass(cls, base, base.requires_grad)

        def __getitem__(self, idx):
            out = super().__getitem__(idx)
            if int(out.shape[0]) > 0:
                return out[:-1]
            return out

    data = make_torch_ssl_dataset()
    views = {"X_l_s": _WeirdTensor(data.X_l.clone())}
    ds = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        views=views,
    )
    method = DeFixMatchMethod(
        DeFixMatchSpec(
            model_bundle=make_model_bundle(),
            batch_size=2,
            max_epochs=1,
            use_cat=True,
        )
    )
    with pytest.raises(InductiveValidationError, match="Labeled strong batch size mismatch"):
        method.fit(ds, device=DeviceSpec(device="cpu"), seed=0)


def test_defixmatch_labeled_logits_shape_mismatch():
    class _MismatchLogits(torch.nn.Module):
        def __init__(self, shapes: list[int]):
            super().__init__()
            self.shapes = shapes
            self.calls = 0
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            idx = min(self.calls, len(self.shapes) - 1)
            self.calls += 1
            return torch.zeros((int(x.shape[0]), self.shapes[idx]), device=x.device)

    data = make_torch_ssl_dataset()
    model = _MismatchLogits([2, 3, 2, 2])
    bundle = TorchModelBundle(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    method = DeFixMatchMethod(
        DeFixMatchSpec(model_bundle=bundle, batch_size=2, max_epochs=1, use_cat=False)
    )
    with pytest.raises(InductiveValidationError, match="Labeled logits shape mismatch"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_defixmatch_empty_masks_zero_losses(monkeypatch):
    def _empty_cycle_batch_indices(_n, *, batch_size, generator, device, steps):
        empty = torch.tensor([], dtype=torch.int64, device=device)
        for _ in range(int(steps)):
            yield empty

    data = make_torch_ssl_dataset()
    y_safe = _SafeLabels(data.y_l)
    data = DummyDataset(
        X_l=data.X_l,
        y_l=y_safe,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
    )
    method = DeFixMatchMethod(
        DeFixMatchSpec(model_bundle=make_model_bundle(), batch_size=2, max_epochs=1)
    )
    monkeypatch.setattr(defixmatch, "cycle_batch_indices", _empty_cycle_batch_indices)
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_adsh_fit_predict_variants_and_scores():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    method = ADSHMethod(
        ADSHSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            use_cat=True,
            detach_target=True,
            score_warmup_epochs=0,
            p_cutoff=0.95,
            majority_class=0,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    method2 = ADSHMethod(
        ADSHSpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=1,
            use_cat=False,
            detach_target=False,
            score_warmup_epochs=2,
            p_cutoff=1.0,
        )
    )
    _fit_predict(method2, data)


def test_adsh_unlabeled_batch_row_mismatch(monkeypatch):
    data = make_torch_ssl_dataset()
    bad = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s[:1],
    )
    method = ADSHMethod(ADSHSpec(model_bundle=make_model_bundle(), batch_size=2, max_epochs=1))
    monkeypatch.setattr(adsh, "ensure_torch_data", lambda _data, device: bad)
    with pytest.raises(
        InductiveValidationError, match="X_u_w and X_u_s must have the same number of rows"
    ):
        method.fit(bad, device=DeviceSpec(device="cpu"), seed=0)


def test_adsh_eval_mode_initial_logits_branch():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    bundle.model.eval()
    method = ADSHMethod(ADSHSpec(model_bundle=bundle, batch_size=2, max_epochs=1))
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_adsh_majority_class_validation():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    with pytest.raises(InductiveValidationError):
        ADSHMethod(
            ADSHSpec(
                model_bundle=bundle,
                batch_size=2,
                max_epochs=1,
                majority_class="0",  # type: ignore[arg-type]
            )
        ).fit(data, device=DeviceSpec(device="cpu"), seed=0)
    with pytest.raises(InductiveValidationError):
        ADSHMethod(
            ADSHSpec(
                model_bundle=bundle,
                batch_size=2,
                max_epochs=1,
                majority_class=5,
            )
        ).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_adsh_use_cat_bad_logits_ndim():
    data = make_torch_ssl_dataset()
    bundle = _make_bundle_for(_LogitsByBatch())
    spec = ADSHSpec(model_bundle=bundle, batch_size=2, max_epochs=1, use_cat=True)
    with pytest.raises(InductiveValidationError):
        ADSHMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_adsh_non_use_cat_bad_logits_ndim():
    data = make_torch_ssl_dataset()
    bundle = _make_bundle_for(_CountedLogits())
    spec = ADSHSpec(model_bundle=bundle, batch_size=2, max_epochs=1, use_cat=False)
    with pytest.raises(InductiveValidationError):
        ADSHMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


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


def test_vat_fit_predict_variants():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    method = VATMethod(
        VATSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            xi=1e-6,
            eps=2.5,
            num_iters=1,
            unsup_warm_up=0.0,
            freeze_bn=True,
            detach_target=True,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    method2 = VATMethod(
        VATSpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=1,
            xi=1e-4,
            eps=1.0,
            num_iters=1,
            unsup_warm_up=0.5,
            freeze_bn=False,
            detach_target=False,
        )
    )
    _fit_predict(method2, data)


def test_noisy_student_fit_predict_variants():
    data = make_torch_ssl_dataset()
    bundle = make_model_bundle()
    method = NoisyStudentMethod(
        NoisyStudentSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            teacher_epochs=1,
            use_cat=True,
            hard_label=True,
            detach_target=True,
            temperature=1.0,
            p_cutoff=0.0,
            freeze_bn=False,
        )
    )
    _fit_predict(method, data)

    bundle2 = make_model_bundle()
    bundle2 = bundle2.__class__(model=bundle2.model, optimizer=bundle2.optimizer, ema_model=None)
    method2 = NoisyStudentMethod(
        NoisyStudentSpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=1,
            teacher_epochs=0,
            use_cat=False,
            hard_label=False,
            detach_target=False,
            temperature=0.5,
            p_cutoff=1.0,
            unsup_warm_up=0.5,
            freeze_bn=False,
        )
    )
    _fit_predict(method2, data)


def test_vat_vat_loss_error_paths():
    data = make_torch_ssl_dataset()
    method = VATMethod()
    model_ok = make_model_bundle().model

    with pytest.raises(InductiveValidationError):
        method._vat_loss(
            model_ok,
            data.X_u,
            xi=1e-6,
            eps=1.0,
            num_iters=0,
            freeze_bn=True,
            detach_target=True,
            generator=torch.Generator(),
        )

    bad_model = _BadLogits1D()
    with pytest.raises(InductiveValidationError):
        method._vat_loss(
            bad_model,
            data.X_u,
            xi=1e-6,
            eps=1.0,
            num_iters=1,
            freeze_bn=True,
            detach_target=True,
            generator=torch.Generator(),
        )
    with pytest.raises(InductiveValidationError):
        method._vat_loss(
            bad_model,
            data.X_u,
            xi=1e-6,
            eps=1.0,
            num_iters=1,
            freeze_bn=True,
            detach_target=False,
            generator=torch.Generator(),
        )


def test_vat_vat_loss_bad_logits_d():
    data = make_torch_ssl_dataset()
    method = VATMethod()
    bad_model = _GradSensitiveLogits()
    with pytest.raises(InductiveValidationError):
        method._vat_loss(
            bad_model,
            data.X_u,
            xi=1e-6,
            eps=1.0,
            num_iters=1,
            freeze_bn=True,
            detach_target=True,
            generator=torch.Generator(),
        )


def test_vat_vat_loss_bad_logits_adv():
    data = make_torch_ssl_dataset()
    method = VATMethod()
    bad_model = _CountedLogits(in_dim=int(data.X_u.shape[1]), bad_call=3)
    with pytest.raises(InductiveValidationError):
        method._vat_loss(
            bad_model,
            data.X_u,
            xi=1e-6,
            eps=1.0,
            num_iters=1,
            freeze_bn=True,
            detach_target=True,
            generator=torch.Generator(),
        )


def test_vat_vat_loss_grad_none(monkeypatch):
    data = make_torch_ssl_dataset()
    method = VATMethod()
    model = _LinearLogits()

    # Mock autograd to return None for gradient
    def _grad(*args, **kwargs):
        return (None,)

    monkeypatch.setattr(torch.autograd, "grad", _grad)

    with pytest.raises(InductiveValidationError, match="VAT gradient computation failed"):
        method._vat_loss(
            model,
            data.X_u,
            xi=1e-6,
            eps=1.0,
            num_iters=1,
            freeze_bn=True,
            detach_target=True,
            generator=torch.Generator(),
        )


def test_vat_requires_unlabeled_data():
    data = make_torch_ssl_dataset()
    missing = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=None,
        X_u_w=None,
        X_u_s=None,
    )
    method = VATMethod(VATSpec(model_bundle=make_model_bundle()))
    with pytest.raises(InductiveValidationError):
        method.fit(missing, device=DeviceSpec(device="cpu"), seed=0)


def test_vat_fit_bad_logits_and_labels():
    data = make_torch_ssl_dataset()
    bundle = _make_bundle_for(_BadLogits1D())
    method = VATMethod(VATSpec(model_bundle=bundle, batch_size=2, max_epochs=1))
    with pytest.raises(InductiveValidationError):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    bad = DummyDataset(
        X_l=data.X_l,
        y_l=torch.tensor([0, 2, 2, 0], dtype=torch.int64),
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
    )
    method_ok = VATMethod(VATSpec(model_bundle=make_model_bundle(), batch_size=2, max_epochs=1))
    with pytest.raises(InductiveValidationError):
        method_ok.fit(bad, device=DeviceSpec(device="cpu"), seed=0)


def test_vat_predict_proba_backend_mismatch():
    bundle = make_model_bundle()
    method = VATMethod(VATSpec(model_bundle=bundle))
    method._bundle = bundle
    method._backend = None
    with pytest.raises(InductiveValidationError):
        method.predict_proba(make_numpy_dataset().X_l)


def test_noisy_student_train_teacher_errors():
    data = make_torch_ssl_dataset()
    method = NoisyStudentMethod()

    bad_model = _BadLogits1D()
    optimizer = torch.optim.SGD(bad_model.parameters(), lr=0.1)
    with pytest.raises(InductiveValidationError):
        method._train_teacher(
            bad_model,
            optimizer,
            data.X_l,
            data.y_l,
            batch_size=2,
            epochs=1,
            seed=0,
        )

    ok_model = _LinearLogits()
    optimizer2 = torch.optim.SGD(ok_model.parameters(), lr=0.1)
    bad_y = torch.tensor([0, 2, 2, 0], dtype=torch.int64)
    with pytest.raises(InductiveValidationError):
        method._train_teacher(
            ok_model,
            optimizer2,
            data.X_l,
            bad_y,
            batch_size=2,
            epochs=1,
            seed=0,
        )


def test_noisy_student_requires_unlabeled_data():
    data = make_torch_ssl_dataset()
    missing = DummyDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=None,
        X_u_w=None,
        X_u_s=None,
    )
    method = NoisyStudentMethod(
        NoisyStudentSpec(model_bundle=make_model_bundle(), teacher_epochs=0)
    )
    with pytest.raises(InductiveValidationError):
        method.fit(missing, device=DeviceSpec(device="cpu"), seed=0)


def test_noisy_student_teacher_logits_errors():
    data = make_torch_ssl_dataset()

    student = _LinearLogits()
    teacher = _TeacherLogits1D()
    bundle = TorchModelBundle(
        model=student,
        optimizer=torch.optim.SGD(student.parameters(), lr=0.1),
        ema_model=teacher,
    )
    method = NoisyStudentMethod(
        NoisyStudentSpec(model_bundle=bundle, batch_size=2, max_epochs=1, teacher_epochs=0)
    )
    with pytest.raises(InductiveValidationError):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    student2 = _LinearLogits()
    teacher2 = _TeacherLogitsTrunc()
    bundle2 = TorchModelBundle(
        model=student2,
        optimizer=torch.optim.SGD(student2.parameters(), lr=0.1),
        ema_model=teacher2,
    )
    method2 = NoisyStudentMethod(
        NoisyStudentSpec(model_bundle=bundle2, batch_size=2, max_epochs=1, teacher_epochs=0)
    )
    with pytest.raises(InductiveValidationError):
        method2.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_noisy_student_predict_proba_backend_mismatch():
    bundle = make_model_bundle()
    method = NoisyStudentMethod(NoisyStudentSpec(model_bundle=bundle))
    method._bundle = bundle
    method._backend = None
    with pytest.raises(InductiveValidationError):
        method.predict_proba(make_numpy_dataset().X_l)


def test_noisy_student_mask_branches():
    data = make_torch_ssl_dataset()

    model = _LinearLogits()
    with torch.no_grad():
        model.fc.weight.zero_()
    bundle = TorchModelBundle(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        ema_model=copy.deepcopy(model),
    )
    method = NoisyStudentMethod(
        NoisyStudentSpec(
            model_bundle=bundle,
            batch_size=2,
            max_epochs=1,
            teacher_epochs=0,
            hard_label=True,
            p_cutoff=0.99,
        )
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=0)

    model2 = _LinearLogits()
    with torch.no_grad():
        model2.fc.weight.zero_()
    bundle2 = TorchModelBundle(
        model=model2,
        optimizer=torch.optim.SGD(model2.parameters(), lr=0.1),
        ema_model=copy.deepcopy(model2),
    )
    method2 = NoisyStudentMethod(
        NoisyStudentSpec(
            model_bundle=bundle2,
            batch_size=2,
            max_epochs=1,
            teacher_epochs=0,
            hard_label=False,
            p_cutoff=0.0,
        )
    )
    method2.fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    "module",
    [fixmatch, flexmatch, mixmatch, adamatch, free_match, softmatch, defixmatch],
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


@pytest.mark.parametrize("method_cls", DICT_METHODS)
def test_deep_methods_predict_proba_dict_input(method_cls):
    method = method_cls()
    method._bundle = _make_bundle_for(_DictModel())
    method._backend = "torch"
    X = {"x": torch.zeros((3, 2), dtype=torch.float32)}
    proba = method.predict_proba(X)
    assert proba.shape[0] == 3


@pytest.mark.parametrize("method_cls", DICT_METHODS)
def test_deep_methods_predict_proba_empty_dict(method_cls):
    method = method_cls()
    method._bundle = _make_bundle_for(_DictModel())
    method._backend = "torch"
    X = {"x": torch.zeros((0, 2), dtype=torch.float32)}
    proba = method.predict_proba(X)
    assert proba.shape[0] == 0


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
        DeFixMatchMethod,
        FixMatchMethod,
        FlexMatchMethod,
        UDAMethod,
        AdaMatchMethod,
        FreeMatchMethod,
        SoftMatchMethod,
        MixMatchMethod,
        NoisyStudentMethod,
        TemporalEnsemblingMethod,
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
    overrides = {"use_cat": True}
    if method_cls is NoisyStudentMethod:
        overrides["teacher_epochs"] = 0
        overrides["freeze_bn"] = False
    spec = _make_spec(method_cls, bundle, **overrides)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize("method_cls", CAT_METHODS)
def test_use_cat_bad_concat_batch(method_cls):
    data = _data_for_method(method_cls)
    bundle = _make_bundle_for(_BadBatch())
    overrides = {"use_cat": True}
    if method_cls is NoisyStudentMethod:
        overrides["teacher_epochs"] = 0
        overrides["freeze_bn"] = False
    spec = _make_spec(method_cls, bundle, **overrides)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_noisy_student_unlabeled_batch_mismatch():
    x_lb = torch.zeros((2, 2))
    y_lb = torch.tensor([0, 1], dtype=torch.int64)
    x_u = torch.ones((2, 2))
    data = DummyDataset(X_l=x_lb, y_l=y_lb, X_u=x_u, X_u_w=x_u, X_u_s=x_u)
    bundle = _make_bundle_for(_BadBatchOnUnlabeled())
    spec = NoisyStudentSpec(
        model_bundle=bundle,
        batch_size=2,
        max_epochs=1,
        teacher_epochs=0,
    )
    with pytest.raises(InductiveValidationError, match="Unlabeled logits batch size"):
        NoisyStudentMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_noisy_student_labeled_batch_mismatch():
    x_lb = torch.zeros((2, 2))
    y_lb = torch.tensor([0, 1], dtype=torch.int64)
    x_u = torch.ones((2, 2))
    data = DummyDataset(X_l=x_lb, y_l=y_lb, X_u=x_u, X_u_w=x_u, X_u_s=x_u)
    bundle = _make_bundle_for(_BadBatchOnLabeled())
    spec = NoisyStudentSpec(
        model_bundle=bundle,
        batch_size=2,
        max_epochs=1,
        teacher_epochs=0,
    )
    with pytest.raises(InductiveValidationError, match="Labeled logits batch size"):
        NoisyStudentMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    "method_cls", CAT_METHODS + [PiModelMethod, MeanTeacherMethod, TemporalEnsemblingMethod]
)
def test_non_use_cat_bad_logits_ndim(method_cls):
    data = _data_for_method(method_cls)
    bundle = _make_bundle_for(_BadLogits1D(), with_ema=method_cls is MeanTeacherMethod)
    overrides = {}
    if method_cls is NoisyStudentMethod:
        overrides["teacher_epochs"] = 0
    spec = _make_spec(method_cls, bundle, **overrides)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    "method_cls", CAT_METHODS + [PiModelMethod, MeanTeacherMethod, TemporalEnsemblingMethod]
)
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
    overrides = {}
    if method_cls is NoisyStudentMethod:
        overrides["teacher_epochs"] = 0
    spec = _make_spec(method_cls, bundle, **overrides)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(data_mismatch, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    "method_cls", CAT_METHODS + [PiModelMethod, MeanTeacherMethod, TemporalEnsemblingMethod]
)
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
    overrides = {}
    if method_cls is NoisyStudentMethod:
        overrides["teacher_epochs"] = 0
    spec = _make_spec(method_cls, bundle, **overrides)
    with pytest.raises(InductiveValidationError):
        method_cls(spec).fit(data_mismatch, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    "method_cls", CAT_METHODS + [PiModelMethod, MeanTeacherMethod, TemporalEnsemblingMethod]
)
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
    overrides = {}
    if method_cls is NoisyStudentMethod:
        overrides["teacher_epochs"] = 0
    spec = _make_spec(method_cls, bundle, **overrides)
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


def test_noisy_student_check_teacher_mismatches():
    ns = NoisyStudentMethod(NoisyStudentSpec(model_bundle=make_model_bundle()))
    student = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(InductiveValidationError):
        ns._check_teacher(student, student)

    teacher = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    with pytest.raises(InductiveValidationError):
        ns._check_teacher(student, teacher)

    teacher2 = torch.nn.Linear(3, 2, bias=False)
    with pytest.raises(InductiveValidationError):
        ns._check_teacher(student, teacher2)

    teacher3 = torch.nn.Linear(2, 2, bias=False).to(device="meta")
    with pytest.raises(InductiveValidationError):
        ns._check_teacher(student, teacher3)


def test_noisy_student_init_teacher_errors():
    ns = NoisyStudentMethod(NoisyStudentSpec(model_bundle=make_model_bundle()))
    student = torch.nn.Linear(2, 2, bias=False)
    teacher = torch.nn.Linear(3, 2, bias=False)
    with pytest.raises(InductiveValidationError):
        ns._init_teacher(student, teacher)


def test_vat_l2_normalize_helpers():
    with pytest.raises(InductiveValidationError):
        vat._l2_normalize([1.0, 2.0])

    empty = torch.zeros((0, 2))
    out = vat._l2_normalize(empty)
    assert int(out.numel()) == 0


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


def test_adsh_update_scores_errors():
    data = make_torch_ssl_dataset()
    score = torch.zeros((2, 1))
    with pytest.raises(InductiveValidationError):
        adsh._update_scores(
            data.X_u_w,
            make_model_bundle().model,
            score=score,
            batch_size=2,
            p_cutoff=0.5,
            majority_class=0,
        )

    bad_device_score = torch.zeros((2,), device="meta")
    with pytest.raises(InductiveValidationError):
        adsh._update_scores(
            data.X_u_w,
            make_model_bundle().model,
            score=bad_device_score,
            batch_size=2,
            p_cutoff=0.5,
            majority_class=0,
        )

    with pytest.raises(InductiveValidationError):
        adsh._update_scores(
            data.X_u_w,
            _BadLogits1D(),
            score=torch.zeros((2,)),
            batch_size=2,
            p_cutoff=0.5,
            majority_class=0,
        )

    with pytest.raises(InductiveValidationError):
        adsh._update_scores(
            data.X_u_w,
            _LinearLogits(n_classes=3),
            score=torch.zeros((2,)),
            batch_size=2,
            p_cutoff=0.5,
            majority_class=0,
        )


def test_adsh_update_scores_logic():
    class _FixedLogits(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = logits
            self.offset = 0

        def forward(self, x):
            batch = int(x.shape[0])
            out = self.logits[self.offset : self.offset + batch]
            self.offset += batch
            return out

    X_u_w = torch.zeros((11, 2))
    logits = torch.tensor(
        [[5.0, 0.0, 0.0]] + [[0.5, 0.0, 0.0]] * 9 + [[0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    model = _FixedLogits(logits)
    score = torch.full((3,), 0.95)
    updated = adsh._update_scores(
        X_u_w,
        model,
        score=score,
        batch_size=4,
        p_cutoff=0.95,
        majority_class=0,
    )
    expected = torch.softmax(torch.tensor([0.0, 1.0, 0.0]), dim=0)[1].item()
    assert updated[0].item() == pytest.approx(0.95)
    assert updated[1].item() == pytest.approx(expected, rel=1e-3)
    assert updated[2].item() == pytest.approx(0.95)
    assert model.training


def test_adsh_update_scores_majority_missing():
    class _FixedLogits(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = logits
            self.offset = 0

        def forward(self, x):
            batch = int(x.shape[0])
            out = self.logits[self.offset : self.offset + batch]
            self.offset += batch
            return out

    X_u_w = torch.zeros((2, 2))
    logits = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    model = _FixedLogits(logits)
    model.eval()
    score = torch.full((3,), 0.9)
    updated = adsh._update_scores(
        X_u_w,
        model,
        score=score,
        batch_size=2,
        p_cutoff=0.9,
        majority_class=2,
    )
    assert not model.training
    assert updated[2].item() == pytest.approx(0.9)


def test_adsh_update_scores_count_zero():
    class _FixedLogits(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = logits
            self.offset = 0

        def forward(self, x):
            batch = int(x.shape[0])
            out = self.logits[self.offset : self.offset + batch]
            self.offset += batch
            return out

    X_u_w = torch.zeros((3, 2))
    logits = torch.tensor(
        [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    model = _FixedLogits(logits)
    score = torch.full((3,), 0.9)
    updated = adsh._update_scores(
        X_u_w,
        model,
        score=score,
        batch_size=2,
        p_cutoff=0.9,
        majority_class=0,
    )
    assert updated[0].item() == pytest.approx(0.9)
    assert updated[1].item() < 0.9


def test_adsh_update_scores_idx_clamp(monkeypatch):
    class _FixedLogits(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = logits
            self.offset = 0

        def forward(self, x):
            batch = int(x.shape[0])
            out = self.logits[self.offset : self.offset + batch]
            self.offset += batch
            return out

    X_u_w = torch.zeros((1, 2))
    logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    model = _FixedLogits(logits)
    score = torch.full((2,), 0.5)

    monkeypatch.setattr(adsh, "round", lambda _val: 10.0)
    updated = adsh._update_scores(
        X_u_w,
        model,
        score=score,
        batch_size=1,
        p_cutoff=0.5,
        majority_class=0,
    )
    assert updated[1].item() <= 0.5


def test_adsh_score_length_mismatch():
    class _SwitchingClasses(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            batch = int(x.shape[0])
            if self.calls == 1:
                return torch.zeros((batch, 3), device=x.device)
            return torch.zeros((batch, 2), device=x.device)

    data = make_torch_ssl_dataset()
    model = _SwitchingClasses()
    bundle = TorchModelBundle(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        ema_model=None,
    )
    method = ADSHMethod(ADSHSpec(model_bundle=bundle, batch_size=2, max_epochs=1))
    with pytest.raises(InductiveValidationError, match="score length"):
        method.fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_adsh_y_lb_range_error(monkeypatch):
    data = make_torch_ssl_dataset()
    spec = ADSHSpec(model_bundle=make_model_bundle(), batch_size=2, max_epochs=1)

    def fake_cycle_batches(X, y, *, batch_size, generator, steps):
        x_bad = X[:2]
        y_bad = torch.tensor([0, 3], dtype=torch.int64)
        for _ in range(int(steps)):
            yield x_bad, y_bad

    monkeypatch.setattr("modssc.inductive.methods.adsh.cycle_batches", fake_cycle_batches)
    with pytest.raises(InductiveValidationError, match="y_l labels must be within"):
        ADSHMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def _install_fake_tg_utils(monkeypatch, *, with_subgraph: bool):
    utils = types.ModuleType("torch_geometric.utils")
    if with_subgraph:

        def subgraph(idx, edge_index, relabel_nodes=True, num_nodes=None):
            return edge_index, None

        utils.subgraph = subgraph

    tg = types.ModuleType("torch_geometric")
    tg.utils = utils
    monkeypatch.setitem(sys.modules, "torch_geometric", tg)
    monkeypatch.setitem(sys.modules, "torch_geometric.utils", utils)


def test_adamatch_fit_len_zero_no_shape(monkeypatch):
    class _NoShape:
        pass

    data = DummyDataset(
        X_l=_NoShape(),
        y_l=torch.tensor([0, 1], dtype=torch.int64),
        X_u_w=_NoShape(),
        X_u_s=_NoShape(),
    )
    monkeypatch.setattr(adamatch, "detect_backend", lambda _x: "torch")
    monkeypatch.setattr(adamatch, "ensure_torch_data", lambda d, device: d)
    monkeypatch.setattr(adamatch, "ensure_float_tensor", lambda *_args, **_kwargs: None)
    spec = _make_spec(AdaMatchMethod, _make_bundle_for(_DictModel()))
    with pytest.raises(InductiveValidationError, match="X_l must be non-empty"):
        AdaMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_adamatch_fit_get_device_default(monkeypatch):
    class _ShapeOnly:
        def __init__(self):
            self.shape = (2, 2)

    data = DummyDataset(
        X_l=_ShapeOnly(),
        y_l=torch.tensor([0, 1], dtype=torch.int64),
        X_u_w=_ShapeOnly(),
        X_u_s=_ShapeOnly(),
    )
    monkeypatch.setattr(adamatch, "detect_backend", lambda _x: "torch")
    monkeypatch.setattr(adamatch, "ensure_torch_data", lambda d, device: d)
    monkeypatch.setattr(adamatch, "ensure_float_tensor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adamatch, "ensure_model_device", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        adamatch,
        "cycle_batches",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    spec = _make_spec(AdaMatchMethod, _make_bundle_for(_DictModel()))
    with pytest.raises(RuntimeError, match="stop"):
        AdaMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_adamatch_fit_slice_dict_with_subgraph(monkeypatch):
    X_l = {"x": torch.zeros((2, 2)), "edge_index": torch.tensor([[0], [1]])}
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    X_u_w = {
        "x": torch.zeros((2, 2)),
        "edge_index": torch.tensor([[0], [1]]),
        "mask": torch.tensor([1, 0]),
        "meta": "keep",
    }
    X_u_s = {
        "x": torch.zeros((2, 2)),
        "edge_index": torch.tensor([[0], [1]]),
        "mask": torch.tensor([0, 1]),
        "meta": "keep",
    }
    data = DummyDataset(X_l=X_l, y_l=y_l, X_u_w=X_u_w, X_u_s=X_u_s)

    monkeypatch.setattr(adamatch, "ensure_torch_data", lambda d, device: d)
    monkeypatch.setattr(adamatch, "cycle_batches", lambda *_args, **_kwargs: iter([(X_l, y_l)]))
    monkeypatch.setattr(
        adamatch, "cycle_batch_indices", lambda *_args, **_kwargs: iter([torch.tensor([0])])
    )
    monkeypatch.setattr(
        adamatch,
        "extract_logits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    _install_fake_tg_utils(monkeypatch, with_subgraph=True)

    spec = _make_spec(AdaMatchMethod, _make_bundle_for(_DictModel()))
    with pytest.raises(RuntimeError, match="stop"):
        AdaMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_adamatch_fit_slice_dict_without_subgraph(monkeypatch):
    X_l = {"x": torch.zeros((2, 2)), "edge_index": torch.tensor([[0], [1]])}
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    X_u_w = {
        "x": torch.zeros((2, 2)),
        "edge_index": torch.tensor([[0], [1]]),
        "mask": torch.tensor([1, 0]),
        "meta": "keep",
    }
    X_u_s = {
        "x": torch.zeros((2, 2)),
        "edge_index": torch.tensor([[0], [1]]),
        "mask": torch.tensor([0, 1]),
        "meta": "keep",
    }
    data = DummyDataset(X_l=X_l, y_l=y_l, X_u_w=X_u_w, X_u_s=X_u_s)

    monkeypatch.setattr(adamatch, "ensure_torch_data", lambda d, device: d)
    monkeypatch.setattr(adamatch, "cycle_batches", lambda *_args, **_kwargs: iter([(X_l, y_l)]))
    monkeypatch.setattr(
        adamatch, "cycle_batch_indices", lambda *_args, **_kwargs: iter([torch.tensor([0])])
    )
    monkeypatch.setattr(
        adamatch,
        "extract_logits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    _install_fake_tg_utils(monkeypatch, with_subgraph=False)

    spec = _make_spec(AdaMatchMethod, _make_bundle_for(_DictModel()))
    with pytest.raises(RuntimeError, match="stop"):
        AdaMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)


def test_adamatch_fit_slice_dict_missing_x_and_edge_index(monkeypatch):
    class _NoXDict(dict):
        @property
        def shape(self):
            return self["feat"].shape

    class _FeatModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            if isinstance(x, dict):
                x = x["feat"]
            return self.fc(x)

    X_l = torch.zeros((2, 2))
    y_l = torch.tensor([0, 1], dtype=torch.int64)
    X_u_w = _NoXDict({"feat": torch.zeros((2, 2))})
    X_u_s = _NoXDict({"feat": torch.zeros((2, 2))})
    data = DummyDataset(X_l=X_l, y_l=y_l, X_u_w=X_u_w, X_u_s=X_u_s)

    monkeypatch.setattr(adamatch, "ensure_torch_data", lambda d, device: d)
    monkeypatch.setattr(adamatch, "cycle_batches", lambda *_args, **_kwargs: iter([(X_l, y_l)]))
    monkeypatch.setattr(
        adamatch, "cycle_batch_indices", lambda *_args, **_kwargs: iter([torch.tensor([0])])
    )
    monkeypatch.setattr(
        adamatch,
        "extract_logits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )

    spec = _make_spec(AdaMatchMethod, _make_bundle_for(_FeatModel()), use_cat=False)
    with pytest.raises(RuntimeError, match="stop"):
        AdaMatchMethod(spec).fit(data, device=DeviceSpec(device="cpu"), seed=0)
