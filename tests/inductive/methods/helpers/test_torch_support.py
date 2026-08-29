from __future__ import annotations

import copy

import pytest
import torch

import modssc.inductive.methods.deep_utils as legacy_deep_utils
import modssc.inductive.methods.helpers.torch_support as torch_support
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError


def test_legacy_deep_utils_module_aliases_torch_support() -> None:
    assert legacy_deep_utils is torch_support
    assert legacy_deep_utils.ensure_model_bundle is torch_support.ensure_model_bundle
    assert legacy_deep_utils.predict_proba_from_bundle is torch_support.predict_proba_from_bundle


def test_parameter_only_ema_averages_parameters_and_copies_all_buffers() -> None:
    class ModelWithBuffers(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0]))
            self.register_buffer("running", torch.tensor([4.0]))
            self.register_buffer("updates", torch.tensor(7, dtype=torch.int64))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value * self.weight

    model = ModelWithBuffers()
    ema_model = copy.deepcopy(model)
    with torch.no_grad():
        ema_model.weight.zero_()
        ema_model.running.zero_()
        ema_model.updates.zero_()
    bundle = TorchModelBundle(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        ema_model=ema_model,
        meta={
            "ema_decay": 0.25,
            "ema_strategy": "parameters_only_copy_buffers",
        },
    )

    torch_support.update_ema_model(bundle)

    torch.testing.assert_close(ema_model.weight, torch.tensor([1.5]))
    torch.testing.assert_close(ema_model.running, torch.tensor([4.0]))
    torch.testing.assert_close(ema_model.updates, torch.tensor(7, dtype=torch.int64))


def test_parameter_only_ema_rejects_state_names_and_shape_mismatches() -> None:
    class StateModel(torch.nn.Module):
        def __init__(
            self,
            *,
            parameter_size: int = 2,
            buffer_size: int = 2,
            parameter_name: str = "weight",
        ) -> None:
            super().__init__()
            self.register_parameter(
                parameter_name,
                torch.nn.Parameter(torch.ones(parameter_size)),
            )
            self.register_buffer("running", torch.ones(buffer_size))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            parameter = next(self.parameters())
            return value * parameter[0]

    def bundle(model: torch.nn.Module, ema_model: torch.nn.Module) -> TorchModelBundle:
        return TorchModelBundle(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            ema_model=ema_model,
            meta={
                "ema_decay": 0.5,
                "ema_strategy": "parameters_only_copy_buffers",
            },
        )

    with pytest.raises(InductiveValidationError, match="state must match"):
        torch_support.update_ema_model(
            bundle(
                StateModel(parameter_name="weight"),
                StateModel(parameter_name="other"),
            )
        )

    with pytest.raises(InductiveValidationError, match="parameter shapes"):
        torch_support.update_ema_model(
            bundle(
                StateModel(parameter_size=2),
                StateModel(parameter_size=3),
            )
        )

    with pytest.raises(InductiveValidationError, match="buffer shapes"):
        torch_support.update_ema_model(
            bundle(
                StateModel(buffer_size=2),
                StateModel(buffer_size=3),
            )
        )
