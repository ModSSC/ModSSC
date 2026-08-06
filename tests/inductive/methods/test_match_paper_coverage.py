from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import modssc.inductive.methods.fixmatch as fixmatch_module
import modssc.inductive.methods.flexmatch as flexmatch_module
import modssc.inductive.methods.free_match as free_match_module
import modssc.inductive.methods.softmatch as softmatch_module
from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.fixmatch import FixMatchMethod, FixMatchSpec
from modssc.inductive.methods.flexmatch import FlexMatchMethod, FlexMatchSpec
from modssc.inductive.methods.free_match import FreeMatchMethod, FreeMatchSpec
from modssc.inductive.methods.softmatch import SoftMatchMethod, SoftMatchSpec
from modssc.inductive.types import DeviceSpec, InductiveDataset

_FIXED_STEP_CASES = (
    (
        fixmatch_module,
        FixMatchMethod,
        FixMatchSpec,
        {"use_cat": True},
    ),
    (
        flexmatch_module,
        FlexMatchMethod,
        FlexMatchSpec,
        {"use_cat": True},
    ),
    (
        free_match_module,
        FreeMatchMethod,
        FreeMatchSpec,
        {"lambda_e": 0.05, "use_quantile": False, "use_cat": True},
    ),
    (
        softmatch_module,
        SoftMatchMethod,
        SoftMatchSpec,
        {"dist_uniform": False, "use_cat": True},
    ),
)


def _bundle() -> TorchModelBundle:
    model = torch.nn.Linear(3, 2, dtype=torch.float64)
    return TorchModelBundle(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
    )


def _standard_data() -> InductiveDataset:
    return InductiveDataset(
        X_l=torch.tensor(
            [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
            dtype=torch.float64,
        ),
        y_l=torch.tensor([0, 1, 0], dtype=torch.int64),
        X_u=torch.tensor(
            [[0.2, 0.1, 0.0], [0.5, 0.4, 0.3], [0.8, 0.7, 0.6]],
            dtype=torch.float64,
        ),
        X_u_w=torch.tensor(
            [[0.2, 0.1, 0.0], [0.5, 0.4, 0.3], [0.8, 0.7, 0.6]],
            dtype=torch.float64,
        ),
        X_u_s=torch.tensor(
            [[0.21, 0.11, 0.01], [0.51, 0.41, 0.31], [0.81, 0.71, 0.61]],
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize(
    ("module", "method_cls", "spec_cls", "frozen"),
    _FIXED_STEP_CASES,
)
def test_fixed_step_fit_dispatches_to_shared_trainer(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    method_cls: Any,
    spec_cls: Any,
    frozen: dict[str, Any],
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(owner: Any, data: Any, **kwargs: Any) -> None:
        calls.append({"owner": owner, "data": data, **kwargs})

    monkeypatch.setattr(module, "run_fixed_step_match", fake_run)
    method = method_cls(spec_cls(training_mode="fixed_steps", max_steps=1, **frozen))
    data = SimpleNamespace(X_u_w=torch.ones((4, 3)))
    returned = method.fit(data, device=DeviceSpec(device="cpu"), seed=9)
    assert returned is method
    assert len(calls) == 1
    assert calls[0]["owner"] is method
    assert calls[0]["data"] is data
    assert calls[0]["seed"] == 9
    assert calls[0]["method_id"] == method.info.method_id


@pytest.mark.parametrize("spec_cls", [FixMatchSpec, FlexMatchSpec, FreeMatchSpec, SoftMatchSpec])
def test_method_specs_do_not_expose_bench_profile_identity(spec_cls: Any) -> None:
    assert not hasattr(spec_cls(), "profile")


@pytest.mark.parametrize(
    "method",
    [
        FixMatchMethod(
            FixMatchSpec(
                use_cat=False,
            )
        ),
        FlexMatchMethod(
            FlexMatchSpec(
                use_cat=False,
            )
        ),
        FreeMatchMethod(
            FreeMatchSpec(
                lambda_e=0.001,
                use_quantile=False,
                use_cat=True,
            )
        ),
        SoftMatchMethod(
            SoftMatchSpec(
                dist_uniform=False,
                use_cat=False,
            )
        ),
    ],
)
def test_method_rejects_frozen_fixed_step_contract_changes(method: Any) -> None:
    with pytest.raises(InductiveValidationError, match="frozen hyperparameter"):
        method._validate_fixed_step_contract()


def test_flexmatch_fixed_step_fit_requires_unlabeled_weak_pool() -> None:
    method = FlexMatchMethod(
        FlexMatchSpec(
            training_mode="fixed_steps",
            use_cat=True,
            max_steps=1,
        )
    )
    with pytest.raises(InductiveValidationError, match="requires X_u_w"):
        method.fit(
            SimpleNamespace(X_u_w=None),
            device=DeviceSpec(device="cpu"),
            seed=0,
        )


def test_uninitialized_paper_states_traces_and_checkpoint_guards() -> None:
    fix = FixMatchMethod(FixMatchSpec(p_cutoff=0.91))
    assert fix._paper_trace() == {"confidence_threshold": 0.91}

    flex = FlexMatchMethod(FlexMatchSpec())
    with pytest.raises(InductiveValidationError, match="state is not initialized"):
        flex._paper_state()
    with pytest.raises(InductiveValidationError, match="state not initialized"):
        flex._update_selected_labels(
            idx_u=torch.tensor([0]),
            pseudo=torch.tensor([1]),
            select=torch.tensor([True]),
        )
    assert flex._paper_trace() == {
        "classwise_acc": None,
        "selected_count": 0,
        "selected_label_count": 0,
    }
    with pytest.raises(InductiveValidationError, match="requires a model bundle"):
        flex._load_paper_state({})
    flex_with_bundle = FlexMatchMethod(FlexMatchSpec(model_bundle=_bundle()))
    with pytest.raises(InductiveValidationError, match="state is invalid"):
        flex_with_bundle._load_paper_state(
            {"selected_label": "invalid", "classwise_acc": torch.ones(2), "ulb_size": 2}
        )
    with pytest.raises(InductiveValidationError, match="pool size is invalid"):
        flex_with_bundle._load_paper_state(
            {
                "selected_label": torch.tensor([-1, -1]),
                "classwise_acc": torch.ones(2),
                "ulb_size": 3,
            }
        )

    free = FreeMatchMethod(FreeMatchSpec())
    with pytest.raises(InductiveValidationError, match="state is not initialized"):
        free._paper_state()
    assert free._paper_trace() == {
        "time_p": None,
        "self_adaptive_threshold": None,
        "p_model": None,
        "label_hist": None,
        "lambda_e": 0.001,
    }
    with pytest.raises(InductiveValidationError, match="requires a model bundle"):
        free._load_paper_state({})
    free_with_bundle = FreeMatchMethod(FreeMatchSpec(model_bundle=_bundle()))
    with pytest.raises(InductiveValidationError, match="state is invalid"):
        free_with_bundle._load_paper_state({})
    with pytest.raises(InductiveValidationError, match="state shapes are invalid"):
        free_with_bundle._load_paper_state(
            {
                "p_model": torch.ones((2, 1)),
                "label_hist": torch.ones((2, 1)),
                "time_p": torch.ones(1),
            }
        )

    soft = SoftMatchMethod(SoftMatchSpec())
    with pytest.raises(InductiveValidationError, match="state is not initialized"):
        soft._paper_state()
    assert soft._paper_trace() == {
        "p_model": None,
        "p_target": None,
        "prob_max_mu_t": None,
        "prob_max_var_t": None,
    }
    with pytest.raises(InductiveValidationError, match="requires a model bundle"):
        soft._load_paper_state({})
    soft_with_bundle = SoftMatchMethod(SoftMatchSpec(model_bundle=_bundle()))
    with pytest.raises(InductiveValidationError, match="state is invalid"):
        soft_with_bundle._load_paper_state({})


def test_freematch_zero_acceptance_uses_zero_entropy_term() -> None:
    method = FreeMatchMethod(FreeMatchSpec(lambda_e=0.05, ema_p=0.999, use_quantile=False))
    method._p_model = torch.tensor([0.5, 0.5], dtype=torch.float64)
    method._label_hist = torch.tensor([0.5, 0.5], dtype=torch.float64)
    method._time_p = torch.tensor(0.99, dtype=torch.float64)
    logits_l = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float64)
    logits_u = torch.zeros((4, 2), dtype=torch.float64)
    result = method._paper_step(
        logits_l,
        logits_u,
        logits_u,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.arange(4),
    )
    assert result.accepted == 0.0
    assert result.diagnostics["entropy_loss"] == 0.0


def test_softmatch_lazy_per_class_statistics_cover_present_and_absent_classes() -> None:
    method = SoftMatchMethod(
        SoftMatchSpec(per_class=True, dist_align=True, dist_uniform=False, ema_p=0.5)
    )
    logits_l = torch.tensor(
        [[3.0, 0.0, -1.0], [0.0, 3.0, -1.0], [2.0, 0.0, -1.0]],
        dtype=torch.float64,
    )
    logits_uw = torch.tensor(
        [[4.0, 0.0, -1.0], [3.0, 0.0, -1.0], [0.0, 4.0, -1.0]],
        dtype=torch.float64,
    )
    result = method._paper_step(
        logits_l,
        logits_uw,
        logits_uw,
        torch.tensor([0, 1, 0], dtype=torch.int64),
        torch.arange(3),
    )
    assert result.unlabeled == 3
    assert method._prob_max_mu_t is not None
    assert method._prob_max_var_t is not None
    assert tuple(method._prob_max_mu_t.shape) == (3,)


@pytest.mark.parametrize(
    ("method_cls", "spec_cls"),
    [
        (FixMatchMethod, FixMatchSpec),
        (FlexMatchMethod, FlexMatchSpec),
        (FreeMatchMethod, FreeMatchSpec),
        (SoftMatchMethod, SoftMatchSpec),
    ],
)
def test_standardized_max_steps_stops_inside_epoch_and_rejects_zero(
    method_cls: Any,
    spec_cls: Any,
) -> None:
    data = _standard_data()
    invalid = method_cls(
        spec_cls(
            model_bundle=_bundle(),
            batch_size=1,
            mu=1,
            max_steps=0,
        )
    )
    with pytest.raises(InductiveValidationError, match="max_steps must be >= 1"):
        invalid.fit(data, device=DeviceSpec(device="cpu"), seed=3)

    method = method_cls(
        spec_cls(
            model_bundle=_bundle(),
            batch_size=1,
            mu=1,
            max_steps=2,
        )
    )
    method.fit(data, device=DeviceSpec(device="cpu"), seed=3)
    assert method.diagnostics_["optimization_steps"] == 2
    assert method.diagnostics_["target_steps"] == 2
