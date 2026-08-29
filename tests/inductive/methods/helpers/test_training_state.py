from __future__ import annotations

import copy

import pytest
import torch

from modssc.inductive.deep.types import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.helpers.torch_support import (
    optimizer_step_with_bundle,
    predict_proba_from_bundle,
    step_scheduler,
    update_ema_model,
)


class _StateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.bn = torch.nn.BatchNorm1d(2)
        self.register_buffer("complex_state", torch.tensor([0.0j], dtype=torch.complex64))

    def forward(self, x):
        return self.bn(self.linear(x))


def _bundle(model: torch.nn.Module, *, ema_model=None, scheduler=None, meta=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return TorchModelBundle(
        model=model,
        optimizer=optimizer,
        ema_model=ema_model,
        scheduler=scheduler,
        meta=meta,
    )


def test_step_scheduler_handles_configured_and_absent_scheduler() -> None:
    model = torch.nn.Linear(2, 2)
    bundle = _bundle(model)
    step_scheduler(bundle)

    scheduler = torch.optim.lr_scheduler.LambdaLR(bundle.optimizer, lambda step: 0.5**step)
    scheduled = TorchModelBundle(
        model=model,
        optimizer=bundle.optimizer,
        scheduler=scheduler,
    )
    bundle.optimizer.step()
    step_scheduler(scheduled)
    assert scheduler.last_epoch == 1
    assert scheduler.get_last_lr() == pytest.approx([0.05])


def test_update_ema_model_updates_float_complex_and_discrete_state() -> None:
    model = _StateModel()
    ema_model = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(2.0)
        for parameter in ema_model.parameters():
            parameter.zero_()
        model.bn.running_mean.fill_(4.0)
        ema_model.bn.running_mean.zero_()
        model.bn.num_batches_tracked.fill_(7)
        ema_model.bn.num_batches_tracked.zero_()
        model.complex_state.fill_(2.0 + 4.0j)
        ema_model.complex_state.zero_()

    bundle = _bundle(model, ema_model=ema_model, meta={"ema_decay": 0.25})
    update_ema_model(bundle)

    assert torch.allclose(ema_model.linear.weight, torch.full_like(ema_model.linear.weight, 1.5))
    assert torch.allclose(
        ema_model.bn.running_mean, torch.full_like(ema_model.bn.running_mean, 3.0)
    )
    assert ema_model.bn.num_batches_tracked.item() == 7
    assert torch.allclose(
        ema_model.complex_state,
        torch.tensor([1.5 + 3.0j], dtype=torch.complex64),
    )


def test_update_ema_model_default_decay_and_no_ema() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    ema_model = copy.deepcopy(model)
    with torch.no_grad():
        model.weight.fill_(1.0)
        ema_model.weight.zero_()
    update_ema_model(_bundle(model, ema_model=ema_model))
    assert ema_model.weight.item() == pytest.approx(0.001)

    update_ema_model(_bundle(model), decay=-1.0)


@pytest.mark.parametrize("decay", [-0.1, 1.0])
def test_update_ema_model_validates_decay(decay: float) -> None:
    model = torch.nn.Linear(1, 1)
    with pytest.raises(InductiveValidationError, match="EMA decay"):
        update_ema_model(_bundle(model, ema_model=copy.deepcopy(model)), decay=decay)


def test_update_ema_model_validates_state_keys_and_shapes() -> None:
    model = torch.nn.Linear(2, 2)
    wrong_keys = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(InductiveValidationError, match="state must match"):
        update_ema_model(_bundle(model, ema_model=wrong_keys), decay=0.5)

    wrong_shape = torch.nn.Linear(2, 3)
    with pytest.raises(InductiveValidationError, match="state shapes"):
        update_ema_model(_bundle(model, ema_model=wrong_shape), decay=0.5)


def test_optimizer_step_updates_optimizer_scheduler_and_ema_in_order() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    ema_model = copy.deepcopy(model)
    with torch.no_grad():
        model.weight.fill_(1.0)
        ema_model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5**step)
    bundle = TorchModelBundle(
        model=model,
        optimizer=optimizer,
        ema_model=ema_model,
        scheduler=scheduler,
        meta={"ema_decay": 0.0},
    )
    loss = model(torch.ones(1, 1)).sum()
    loss.backward()

    optimizer_step_with_bundle(bundle)

    assert model.weight.item() == pytest.approx(0.9)
    assert ema_model.weight.item() == pytest.approx(model.weight.item())
    assert scheduler.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)

    plain = torch.nn.Linear(1, 1, bias=False)
    plain_bundle = _bundle(plain)
    plain(torch.ones(1, 1)).sum().backward()
    before = plain.weight.detach().clone()
    optimizer_step_with_bundle(plain_bundle)
    assert not torch.equal(plain.weight, before)


def test_predict_proba_uses_student_by_default_and_can_select_ema_explicitly() -> None:
    model = torch.nn.Linear(2, 2)
    ema_model = copy.deepcopy(model)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([4.0, 0.0]))
        ema_model.weight.zero_()
        ema_model.bias.copy_(torch.tensor([0.0, 4.0]))
    ema_model.train()
    bundle = _bundle(
        model,
        ema_model=ema_model,
        meta={"predict_with_ema": True},
    )
    x = torch.zeros(3, 2)

    default_probs = predict_proba_from_bundle(
        bundle,
        fitted_backend="torch",
        X=x,
        batch_size=2,
    )
    ema_probs = predict_proba_from_bundle(
        bundle,
        fitted_backend="torch",
        X=x,
        batch_size=2,
        use_ema=True,
    )
    no_meta_probs = predict_proba_from_bundle(
        _bundle(model),
        fitted_backend="torch",
        X=x,
        batch_size=2,
    )

    assert torch.all(default_probs.argmax(dim=1) == 0)
    assert torch.all(ema_probs.argmax(dim=1) == 1)
    assert torch.all(no_meta_probs.argmax(dim=1) == 0)
    assert ema_model.training is True


def test_predict_proba_rejects_missing_requested_ema() -> None:
    model = torch.nn.Linear(2, 2)
    with pytest.raises(InductiveValidationError, match="EMA prediction requested"):
        predict_proba_from_bundle(
            _bundle(model),
            fitted_backend="torch",
            X=torch.zeros(1, 2),
            batch_size=1,
            use_ema=True,
        )
