from __future__ import annotations

import math
import os

import pytest
import torch

from modssc.inductive.deep import bundles
from modssc.inductive.deep.types import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError


def test_wide_resnet_bundle_paper_defaults() -> None:
    sample = torch.randn(2, 3, 32, 32)
    bundle = bundles._build_wide_resnet_bundle(
        sample,
        num_classes=10,
        params={},
        seed=17,
        ema=True,
    )

    assert bundle.model.depth == 28
    assert bundle.model.widen_factor == 2
    assert isinstance(bundle.optimizer, torch.optim.SGD)
    assert bundle.optimizer.defaults["lr"] == pytest.approx(0.03)
    assert bundle.optimizer.defaults["momentum"] == pytest.approx(0.9)
    assert bundle.optimizer.defaults["nesterov"] is True
    assert len(bundle.optimizer.param_groups) == 2
    assert {group["weight_decay"] for group in bundle.optimizer.param_groups} == {0.0, 5e-4}
    assert bundle.scheduler is not None
    assert bundle.ema_model is not None
    assert all(not parameter.requires_grad for parameter in bundle.ema_model.parameters())
    assert bundle.meta == {
        "contract_schema_version": 1,
        "classifier_id": "wide_resnet_cifar",
        "depth": 28,
        "widen_factor": 2,
        "in_channels": 3,
        "num_classes": 10,
        "bn_momentum": 0.001,
        "bn_eps": 0.001,
        "input_mean": None,
        "input_std": None,
        "initialization": "modssc_standardized",
        "optimizer": "sgd",
        "lr": 0.03,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0005,
        "scheduler": "cosine",
        "scheduler_step_unit": "optimizer_step",
        "max_steps": 1 << 20,
        "cosine_cycles": 7.0 / 16.0,
        "ema_decay": 0.999,
        "predict_with_ema": True,
        "decay_bias_and_norm": False,
        "reference_implementation": "standardized",
        "ema_strategy": "all_floating_state",
        "ema_reference": "modssc_standardized",
    }


def test_wide_resnet_bundle_selects_explicit_reference_implementation() -> None:
    bundle = bundles._build_wide_resnet_bundle(
        torch.randn(2, 3, 8, 8),
        num_classes=10,
        params={"reference_implementation": "torchssl"},
        seed=5,
        ema=True,
    )

    assert bundle.model.reference_implementation == "torchssl"
    assert bundle.meta["reference_implementation"] == "torchssl"
    assert bundle.meta["ema_strategy"] == "parameters_only_copy_buffers"
    assert bundle.meta["ema_reference"] == "torchssl_named_parameters"


def test_wide_resnet_paper_bundle_does_not_mutate_deterministic_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deterministic_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda *args, **kwargs: deterministic_calls.append((*args, kwargs)),
    )
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    bundles._build_wide_resnet_bundle(
        torch.randn(2, 3, 8, 8),
        num_classes=10,
        params={
            "reference_implementation": "torchssl",
            "depth": 10,
            "widen_factor": 1,
        },
        seed=5,
        ema=True,
    )

    assert deterministic_calls == []
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_wide_resnet_bundle_adamw_without_scheduler_or_ema() -> None:
    params = {
        "depth": 10,
        "widen_factor": 1,
        "optimizer": "adamw",
        "scheduler": None,
        "weight_decay": 0.0,
        "momentum": 0.0,
        "nesterov": False,
        "input_mean": [0.1, 0.2, 0.3],
        "input_std": [0.9, 0.8, 0.7],
    }
    bundle = bundles._build_wide_resnet_bundle(
        torch.randn(2, 3, 8, 8),
        num_classes=3,
        params=params,
        seed=3,
        ema=False,
    )

    assert isinstance(bundle.optimizer, torch.optim.AdamW)
    assert len(bundle.optimizer.param_groups) == 1
    assert bundle.scheduler is None
    assert bundle.ema_model is None
    assert bundle.meta["scheduler"] == "none"
    assert bundle.meta["max_steps"] is None
    assert bundle.meta["cosine_cycles"] is None
    assert bundle.meta["predict_with_ema"] is False
    assert bundle.model.input_mean.flatten().tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_weight_decay_parameter_group_options() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))

    flat_zero, global_zero = bundles._weight_decay_parameters(
        model,
        weight_decay=0.0,
        decay_bias_and_norm=False,
    )
    assert len(flat_zero) == 4
    assert global_zero == 0.0

    flat_all, global_all = bundles._weight_decay_parameters(
        model,
        weight_decay=0.1,
        decay_bias_and_norm=True,
    )
    assert len(flat_all) == 4
    assert global_all == pytest.approx(0.1)

    groups, global_grouped = bundles._weight_decay_parameters(
        model,
        weight_decay=0.1,
        decay_bias_and_norm=False,
    )
    assert global_grouped == 0.0
    assert len(groups[0]["params"]) == 1
    assert len(groups[1]["params"]) == 3


def test_fixmatch_cosine_factor_is_clamped_and_matches_official_schedule() -> None:
    cycles = 7.0 / 16.0
    assert bundles._cosine_lr_factor(-2, max_steps=8, cycles=cycles) == pytest.approx(1.0)
    assert bundles._cosine_lr_factor(4, max_steps=8, cycles=cycles) == pytest.approx(
        math.cos(7.0 * math.pi / 32.0)
    )
    expected_end = math.cos(7.0 * math.pi / 16.0)
    assert bundles._cosine_lr_factor(8, max_steps=8, cycles=cycles) == pytest.approx(expected_end)
    assert bundles._cosine_lr_factor(20, max_steps=8, cycles=cycles) == pytest.approx(expected_end)


@pytest.mark.parametrize(
    ("sample", "params", "ema", "message"),
    [
        ([1.0], {}, False, "4D torch.Tensor"),
        (torch.randn(3, 8, 8), {}, False, "4D torch.Tensor"),
        (torch.randn(2, 3, 8, 8), {"input_layout": "channels_last"}, False, "input_layout"),
        (torch.randn(2, 3, 8, 8), {"lr": 0.0}, False, "lr must"),
        (torch.randn(2, 3, 8, 8), {"weight_decay": -0.1}, False, "weight_decay"),
        (torch.randn(2, 3, 8, 8), {"momentum": -0.1}, False, "momentum"),
        (torch.randn(2, 3, 8, 8), {"momentum": 1.0}, False, "momentum"),
        (
            torch.randn(2, 3, 8, 8),
            {"momentum": 0.0, "nesterov": True},
            False,
            "nesterov requires",
        ),
        (
            torch.randn(2, 3, 8, 8),
            {"depth": 10, "widen_factor": 1, "optimizer": "rmsprop"},
            False,
            "optimizer must",
        ),
        (
            torch.randn(2, 3, 8, 8),
            {"depth": 10, "widen_factor": 1, "max_steps": 0},
            False,
            "max_steps",
        ),
        (
            torch.randn(2, 3, 8, 8),
            {"depth": 10, "widen_factor": 1, "cosine_cycles": 0.0},
            False,
            "cosine_cycles",
        ),
        (
            torch.randn(2, 3, 8, 8),
            {"depth": 10, "widen_factor": 1, "cosine_cycles": 0.6},
            False,
            "cosine_cycles",
        ),
        (
            torch.randn(2, 3, 8, 8),
            {"depth": 10, "widen_factor": 1, "scheduler": "linear"},
            False,
            "scheduler must",
        ),
        (
            torch.randn(2, 3, 8, 8),
            {"depth": 10, "widen_factor": 1, "scheduler": None, "ema_decay": -0.1},
            True,
            "ema_decay",
        ),
        (
            torch.randn(2, 3, 8, 8),
            {"depth": 10, "widen_factor": 1, "scheduler": None, "ema_decay": 1.0},
            True,
            "ema_decay",
        ),
        (
            torch.randn(2, 3, 8, 8),
            {
                "depth": 10,
                "widen_factor": 1,
                "scheduler": None,
                "predict_with_ema": True,
            },
            False,
            "requires ema=true",
        ),
    ],
)
def test_wide_resnet_bundle_validation(
    sample: object,
    params: dict[str, object],
    ema: bool,
    message: str,
) -> None:
    with pytest.raises(InductiveValidationError, match=message):
        bundles._build_wide_resnet_bundle(
            sample,
            num_classes=2,
            params=params,
            seed=0,
            ema=ema,
        )


def test_public_bundle_factory_dispatches_wide_resnet(monkeypatch) -> None:
    model = torch.nn.Linear(2, 2)
    dummy = TorchModelBundle(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    monkeypatch.setattr(bundles, "_build_wide_resnet_bundle", lambda *args, **kwargs: dummy)

    result = bundles.build_torch_bundle_from_classifier(
        classifier_id="wide_resnet_cifar",
        classifier_backend="torch",
        classifier_params={},
        sample=torch.randn(2, 3, 8, 8),
        num_classes=2,
        seed=4,
        ema=True,
    )
    assert result is dummy
