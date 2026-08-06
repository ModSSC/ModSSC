from __future__ import annotations

import copy

import pytest
import torch

from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.fixmatch import FixMatchMethod, FixMatchSpec
from modssc.inductive.methods.flexmatch import FlexMatchMethod, FlexMatchSpec
from modssc.inductive.methods.free_match import FreeMatchMethod, FreeMatchSpec
from modssc.inductive.methods.softmatch import SoftMatchMethod, SoftMatchSpec
from modssc.inductive.types import DeviceSpec, InductiveDataset

from ..conftest import make_torch_ssl_dataset

METHOD_CLASSES = (FixMatchMethod, FlexMatchMethod, FreeMatchMethod, SoftMatchMethod)


def _training_bundle() -> TorchModelBundle:
    torch.manual_seed(11)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    ema_model = copy.deepcopy(model)
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return TorchModelBundle(
        model=model,
        optimizer=optimizer,
        ema_model=ema_model,
        scheduler=scheduler,
        meta={"ema_decay": 0.5, "predict_with_ema": True},
    )


def _dataset() -> InductiveDataset:
    data = make_torch_ssl_dataset()
    idx_u = torch.arange(int(data.X_u.shape[0]), dtype=torch.int64)
    return InductiveDataset(
        X_l=data.X_l,
        y_l=data.y_l,
        X_u=data.X_u,
        X_u_w=data.X_u_w,
        X_u_s=data.X_u_s,
        meta={"idx_u": idx_u, "ulb_size": int(idx_u.numel())},
    )


def _method(method_cls: type, bundle: TorchModelBundle):
    common = {"model_bundle": bundle, "batch_size": 2, "max_epochs": 1, "mu": 1}
    if method_cls is FixMatchMethod:
        return FixMatchMethod(FixMatchSpec(**common, p_cutoff=0.0))
    if method_cls is FlexMatchMethod:
        return FlexMatchMethod(FlexMatchSpec(**common, p_cutoff=0.0))
    if method_cls is FreeMatchMethod:
        return FreeMatchMethod(FreeMatchSpec(**common))
    if method_cls is SoftMatchMethod:
        return SoftMatchMethod(SoftMatchSpec(**common))
    raise AssertionError("unsupported method class")


@pytest.mark.parametrize("method_cls", METHOD_CLASSES)
def test_standardized_ssl_method_only_advances_optimizer(method_cls: type) -> None:
    bundle = _training_bundle()
    initial_ema = {
        name: value.detach().clone() for name, value in bundle.ema_model.state_dict().items()
    }

    method = _method(method_cls, bundle)
    method.fit(_dataset(), device=DeviceSpec(device="cpu"), seed=5)

    assert bundle.scheduler.last_epoch == 0
    assert bundle.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert all(
        torch.equal(value, initial_ema[name])
        for name, value in bundle.ema_model.state_dict().items()
    )


@pytest.mark.parametrize("method_cls", METHOD_CLASSES)
def test_ssl_method_honors_exact_max_steps(method_cls: type) -> None:
    bundle = _training_bundle()
    method = _method(method_cls, bundle)
    method.spec = type(method.spec)(
        **{
            **method.spec.__dict__,
            "max_epochs": 9,
            "max_steps": 1,
        }
    )

    method.fit(_dataset(), device=DeviceSpec(device="cpu"), seed=5)

    assert bundle.scheduler.last_epoch == 0
    assert method.diagnostics_["optimization_steps"] == 1
    assert method.diagnostics_["target_steps"] == 1


@pytest.mark.parametrize("method_cls", METHOD_CLASSES)
def test_ssl_method_rejects_nonpositive_max_steps(method_cls: type) -> None:
    method = _method(method_cls, _training_bundle())
    method.spec = type(method.spec)(**{**method.spec.__dict__, "max_steps": 0})

    with pytest.raises(InductiveValidationError, match="max_steps"):
        method.fit(_dataset(), device=DeviceSpec(device="cpu"), seed=5)
