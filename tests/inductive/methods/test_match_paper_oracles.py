from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from modssc.inductive.deep import TorchModelBundle
from modssc.inductive.deep.match_primitives import FixedSSLBatchSampler
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.fixmatch import FixMatchMethod, FixMatchSpec
from modssc.inductive.methods.flexmatch import FlexMatchMethod, FlexMatchSpec
from modssc.inductive.methods.free_match import FreeMatchMethod, FreeMatchSpec
from modssc.inductive.methods.helpers import match_trainer
from modssc.inductive.methods.helpers.match_trainer import uses_fixed_step_match
from modssc.inductive.methods.softmatch import SoftMatchMethod, SoftMatchSpec
from modssc.inductive.types import DeviceSpec


def _log_probs(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float64).log()


def _bundle() -> TorchModelBundle:
    model = torch.nn.Linear(2, 2, dtype=torch.float64)
    return TorchModelBundle(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )


@pytest.fixture
def step_inputs() -> dict[str, torch.Tensor]:
    return {
        "logits_l": _log_probs([[0.8, 0.2], [0.3, 0.7]]),
        "logits_us": _log_probs([[0.7, 0.3], [0.55, 0.45], [0.4, 0.6], [0.45, 0.55]]),
        "logits_uw_1": _log_probs([[0.96, 0.04], [0.6, 0.4], [0.2, 0.8], [0.51, 0.49]]),
        "logits_uw_2": _log_probs([[0.4, 0.6], [0.9, 0.1], [0.1, 0.9], [0.7, 0.3]]),
        "y": torch.tensor([0, 1], dtype=torch.int64),
        "idx_u": torch.arange(4, dtype=torch.int64),
    }


def test_fixed_step_mode_and_method_contracts_are_explicit() -> None:
    assert uses_fixed_step_match(FixMatchSpec(training_mode="fixed_steps"))
    assert not uses_fixed_step_match(FixMatchSpec())
    FixMatchMethod(FixMatchSpec(use_cat=True))._validate_fixed_step_contract()
    FlexMatchMethod(FlexMatchSpec(use_cat=True))._validate_fixed_step_contract()
    FreeMatchMethod(
        FreeMatchSpec(
            lambda_e=0.05,
            use_quantile=False,
            use_cat=True,
        )
    )._validate_fixed_step_contract()
    SoftMatchMethod(
        SoftMatchSpec(
            dist_uniform=False,
            use_cat=True,
        )
    )._validate_fixed_step_contract()
    with pytest.raises(InductiveValidationError, match="training_mode"):
        uses_fixed_step_match(FixMatchSpec(training_mode="unknown"))


@pytest.mark.parametrize(
    ("method_cls", "spec_cls"),
    [
        (FixMatchMethod, FixMatchSpec),
        (FlexMatchMethod, FlexMatchSpec),
        (FreeMatchMethod, FreeMatchSpec),
        (SoftMatchMethod, SoftMatchSpec),
    ],
)
def test_match_method_entrypoints_reject_unknown_training_modes(
    method_cls: type,
    spec_cls: type,
) -> None:
    method = method_cls(spec_cls(training_mode="unknown"))
    with pytest.raises(InductiveValidationError, match="training_mode"):
        method.fit(None, device=DeviceSpec(device="cpu"), seed=0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_implementation", "standardized", "reference_implementation"),
        ("sampler_mode", "unknown", "sampler_mode"),
        ("augmentation_profile", "unknown", "augmentation_profile"),
        ("reporting_policy", "unknown", "reporting_policy"),
        ("sampler_shuffle_buffer", 0, "sampler_shuffle_buffer"),
        ("evaluation_interval_steps", 0, "evaluation_interval_steps"),
        ("checkpoint_interval_steps", 0, "checkpoint_interval_steps"),
        ("reporting_window_checkpoints", 0, "reporting_window_checkpoints"),
    ],
)
def test_fixed_step_trainer_rejects_invalid_generic_contract_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    params: dict[str, object] = {
        "training_mode": "fixed_steps",
        "reference_implementation": "google_fixmatch",
        "sampler_mode": "shuffle_repeat",
        "sampler_shuffle_buffer": 8192,
        "augmentation_profile": "google_fixmatch_ra",
        "interleave_bn": True,
        "evaluation_interval_steps": 1024,
        "checkpoint_interval_steps": 1024,
        "reporting_policy": "median_last_checkpoints",
        "reporting_window_checkpoints": 20,
        "allow_short_run": True,
    }
    params[field] = value
    spec = FixMatchSpec(**params)  # type: ignore[arg-type]
    with pytest.raises(InductiveValidationError, match=message):
        match_trainer._trainer_config(spec)


def test_historical_evaluation_schedule_stays_at_configured_cadence() -> None:
    target = match_trainer.MATCH_REFERENCE_TARGET_STEPS
    scheduled = {
        step
        for step in range(target)
        if match_trainer._historical_evaluation(
            global_step=step,
            target_steps=target,
            interval=5000,
        )
    }

    assert 0 in scheduled
    assert 5000 in scheduled
    assert 835000 in scheduled
    assert 838000 not in scheduled
    assert 839000 not in scheduled
    assert 1_045_000 in scheduled
    assert 1_048_000 not in scheduled
    assert 838861 not in scheduled
    assert target - 1 not in scheduled
    assert not match_trainer._historical_evaluation(
        global_step=-1,
        target_steps=target,
        interval=5000,
    )
    assert not match_trainer._historical_evaluation(
        global_step=target,
        target_steps=target,
        interval=5000,
    )


def test_forced_continuation_requires_explicit_short_run_and_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODSSC_FORCE_CONTINUATION_STEP", "1024")
    assert (
        match_trainer._forced_continuation_step(
            allow_short_run=True,
            target_steps=4096,
        )
        == 1024
    )
    with pytest.raises(InductiveValidationError, match="allow_short_run"):
        match_trainer._forced_continuation_step(
            allow_short_run=False,
            target_steps=match_trainer.MATCH_REFERENCE_TARGET_STEPS,
        )
    monkeypatch.setenv("MODSSC_FORCE_CONTINUATION_STEP", "not-an-integer")
    with pytest.raises(InductiveValidationError, match="must be an integer"):
        match_trainer._forced_continuation_step(
            allow_short_run=True,
            target_steps=4096,
        )
    monkeypatch.setenv("MODSSC_FORCE_CONTINUATION_STEP", "4096")
    with pytest.raises(InductiveValidationError, match="inside the configured run"):
        match_trainer._forced_continuation_step(
            allow_short_run=True,
            target_steps=4096,
        )
    monkeypatch.delenv("MODSSC_FORCE_CONTINUATION_STEP")
    assert (
        match_trainer._forced_continuation_step(
            allow_short_run=True,
            target_steps=4096,
        )
        is None
    )


def test_torchssl_historical_metric_excludes_terminal_evaluation() -> None:
    metrics = match_trainer._paper_metrics(
        reporting_policy="best_historical_checkpoint",
        reporting_window_checkpoints=20,
        history=[
            {"test_accuracy": 0.70, "historical_eligible": True},
            {"test_accuracy": 0.80, "historical_eligible": True},
            {
                "test_accuracy": 0.99,
                "historical_eligible": False,
                "terminal_evaluation": True,
            },
        ],
    )

    assert metrics["historical_paper_metric"]["test_accuracy"] == pytest.approx(0.80)
    assert metrics["fixed_terminal_metric"]["test_accuracy"] == pytest.approx(0.99)
    with pytest.raises(InductiveValidationError, match="historical checkpoint"):
        match_trainer._paper_metrics(
            reporting_policy="best_historical_checkpoint",
            reporting_window_checkpoints=20,
            history=[{"test_accuracy": 0.75, "historical_eligible": False}],
        )


def test_fixmatch_two_step_mask_and_loss_oracle(
    step_inputs: dict[str, torch.Tensor],
) -> None:
    method = FixMatchMethod(FixMatchSpec(p_cutoff=0.95))
    first = method._paper_step(
        step_inputs["logits_l"],
        step_inputs["logits_uw_1"],
        step_inputs["logits_us"],
        step_inputs["y"],
        step_inputs["idx_u"],
    )
    second_weak = _log_probs([[0.94, 0.06], [0.97, 0.03], [0.04, 0.96], [0.52, 0.48]])
    second = method._paper_step(
        step_inputs["logits_l"],
        second_weak,
        step_inputs["logits_us"],
        step_inputs["y"],
        step_inputs["idx_u"],
    )

    assert (first.accepted, second.accepted) == (1.0, 2.0)
    assert (first.unlabeled, second.unlabeled) == (4, 4)
    assert first.diagnostics["mask_rate"] == pytest.approx(0.25)
    assert second.diagnostics["mask_rate"] == pytest.approx(0.5)
    assert float(first.loss.detach()) == pytest.approx(0.3790779836111542)
    assert float(second.loss.detach()) == pytest.approx(0.5670749037568739)
    assert method._paper_state() == {}
    method._load_paper_state({})
    with pytest.raises(InductiveValidationError, match="must be empty"):
        method._load_paper_state({"unexpected": 1})


def test_flexmatch_two_step_cpl_state_and_resume_oracle(
    step_inputs: dict[str, torch.Tensor],
) -> None:
    spec = FlexMatchSpec(model_bundle=_bundle(), p_cutoff=0.95, use_cat=True)
    method = FlexMatchMethod(spec)
    method._ulb_size = 6
    first = method._paper_step(
        step_inputs["logits_l"],
        step_inputs["logits_uw_1"],
        step_inputs["logits_us"],
        step_inputs["y"],
        step_inputs["idx_u"],
    )
    second_weak = _log_probs([[0.94, 0.06], [0.97, 0.03], [0.04, 0.96], [0.03, 0.97]])
    second = method._paper_step(
        step_inputs["logits_l"],
        second_weak,
        step_inputs["logits_us"],
        step_inputs["y"],
        step_inputs["idx_u"],
    )

    assert first.accepted == second.accepted == 4.0
    assert first.diagnostics["threshold_mean"] == pytest.approx(0.0)
    assert second.diagnostics["threshold_mean"] == pytest.approx(0.05277777777777778)
    state = copy.deepcopy(method._paper_state())
    torch.testing.assert_close(state["selected_label"], torch.tensor([0, 0, 1, 1, -1, -1]))
    torch.testing.assert_close(
        state["classwise_acc"],
        torch.tensor([0.2, 0.0], dtype=state["classwise_acc"].dtype),
    )
    assert state["ulb_size"] == 6

    resumed = FlexMatchMethod(spec)
    resumed._load_paper_state(state)
    third_weak = _log_probs([[0.7, 0.3], [0.45, 0.55], [0.85, 0.15], [0.2, 0.8]])
    expected = method._paper_step(
        step_inputs["logits_l"],
        third_weak,
        step_inputs["logits_us"],
        step_inputs["y"],
        step_inputs["idx_u"],
    )
    actual = resumed._paper_step(
        step_inputs["logits_l"],
        third_weak,
        step_inputs["logits_us"],
        step_inputs["y"],
        step_inputs["idx_u"],
    )
    assert float(actual.loss.detach()) == pytest.approx(float(expected.loss.detach()))
    torch.testing.assert_close(
        resumed._paper_state()["selected_label"],
        method._paper_state()["selected_label"],
    )
    torch.testing.assert_close(
        resumed._paper_state()["classwise_acc"],
        method._paper_state()["classwise_acc"],
    )


def test_flexmatch_duplicate_indices_use_last_accepted_occurrence_and_resume() -> None:
    spec = FlexMatchSpec(model_bundle=_bundle(), p_cutoff=0.95, use_cat=True)
    method = FlexMatchMethod(spec)
    method._ulb_size = 5
    logits_l = _log_probs([[0.8, 0.2], [0.3, 0.7]])
    logits_us = _log_probs([[0.7, 0.3], [0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.55, 0.45]])
    labels = torch.tensor([0, 1], dtype=torch.int64)
    duplicate_indices = torch.tensor([2, 2, 3, 2, 3], dtype=torch.int64)

    method._paper_step(
        logits_l,
        _log_probs([[0.99, 0.01], [0.01, 0.99], [0.98, 0.02], [0.02, 0.98], [0.03, 0.97]]),
        logits_us,
        labels,
        duplicate_indices,
    )
    state = copy.deepcopy(method._paper_state())
    # Pool index 2 receives 0, 1, 1 and pool index 3 receives 0, 1 in
    # occurrence order.  The explicitly registered last-occurrence rule wins.
    torch.testing.assert_close(
        state["selected_label"],
        torch.tensor([-1, -1, 1, 1, -1], dtype=torch.int64),
    )

    resumed = FlexMatchMethod(spec)
    resumed._load_paper_state(state)
    next_indices = torch.tensor([2, 2, 3, 3, 2], dtype=torch.int64)
    next_weak = _log_probs([[0.01, 0.99], [0.99, 0.01], [0.02, 0.98], [0.97, 0.03], [0.96, 0.04]])
    expected = method._paper_step(
        logits_l,
        next_weak,
        logits_us,
        labels,
        next_indices,
    )
    actual = resumed._paper_step(
        logits_l,
        next_weak,
        logits_us,
        labels,
        next_indices,
    )

    assert float(actual.loss.detach()) == pytest.approx(float(expected.loss.detach()))
    expected_state = method._paper_state()
    resumed_state = resumed._paper_state()
    torch.testing.assert_close(
        expected_state["selected_label"],
        torch.tensor([-1, -1, 0, 0, -1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        resumed_state["selected_label"],
        expected_state["selected_label"],
    )
    torch.testing.assert_close(
        resumed_state["classwise_acc"],
        expected_state["classwise_acc"],
    )


def test_freematch_two_step_sat_saf_state_and_resume_oracle(
    step_inputs: dict[str, torch.Tensor],
) -> None:
    spec = FreeMatchSpec(
        model_bundle=_bundle(),
        lambda_e=0.05,
        ema_p=0.999,
        use_quantile=False,
        use_cat=True,
    )
    method = FreeMatchMethod(spec)
    results = [
        method._paper_step(
            step_inputs["logits_l"],
            weak,
            step_inputs["logits_us"],
            step_inputs["y"],
            step_inputs["idx_u"],
        )
        for weak in (step_inputs["logits_uw_1"], step_inputs["logits_uw_2"])
    ]

    assert [result.accepted for result in results] == [4.0, 4.0]
    state = copy.deepcopy(method._paper_state())
    torch.testing.assert_close(
        state["p_model"],
        torch.tensor([0.5000924389308644, 0.4999075739308643], dtype=torch.float64),
    )
    torch.testing.assert_close(
        state["label_hist"],
        torch.tensor([0.5002497564308643, 0.4997502564308643], dtype=torch.float64),
    )
    assert float(state["time_p"]) == pytest.approx(0.5004922889308644)
    assert results[1].diagnostics["threshold_mean"] == pytest.approx(0.5003997825262574)
    assert results[1].diagnostics["entropy_loss"] == pytest.approx(-0.6944144905477425)

    resumed = FreeMatchMethod(spec)
    resumed._load_paper_state(state)
    expected = method._paper_step(
        step_inputs["logits_l"],
        step_inputs["logits_uw_1"],
        step_inputs["logits_us"],
        step_inputs["y"],
        step_inputs["idx_u"],
    )
    actual = resumed._paper_step(
        step_inputs["logits_l"],
        step_inputs["logits_uw_1"],
        step_inputs["logits_us"],
        step_inputs["y"],
        step_inputs["idx_u"],
    )
    assert float(actual.loss.detach()) == pytest.approx(float(expected.loss.detach()))
    for name in ("p_model", "label_hist", "time_p"):
        torch.testing.assert_close(resumed._paper_state()[name], method._paper_state()[name])


def test_softmatch_two_step_gaussian_state_and_resume_oracle(
    step_inputs: dict[str, torch.Tensor],
) -> None:
    spec = SoftMatchSpec(
        model_bundle=_bundle(),
        ema_p=0.999,
        n_sigma=2.0,
        dist_uniform=False,
        use_cat=True,
    )
    method = SoftMatchMethod(spec)
    method._p_model = torch.tensor([0.5, 0.5], dtype=torch.float64)
    method._p_target = torch.tensor([0.5, 0.5], dtype=torch.float64)
    method._prob_max_mu_t = torch.tensor(0.8, dtype=torch.float64)
    method._prob_max_var_t = torch.tensor(0.04, dtype=torch.float64)
    results = [
        method._paper_step(
            step_inputs["logits_l"],
            weak,
            step_inputs["logits_us"],
            step_inputs["y"],
            step_inputs["idx_u"],
        )
        for weak in (step_inputs["logits_uw_1"], step_inputs["logits_uw_2"])
    ]

    assert results[0].accepted == pytest.approx(2.1504690214539894)
    assert results[1].accepted == pytest.approx(2.742587900675272)
    assert results[0].diagnostics["mean_weight"] == pytest.approx(0.5376172553634974)
    assert results[1].diagnostics["mean_weight"] == pytest.approx(0.685646975168818)
    state = copy.deepcopy(method._paper_state())
    torch.testing.assert_close(
        state["p_model"],
        torch.tensor([0.5000924325, 0.4999075675], dtype=torch.float64),
    )
    torch.testing.assert_close(
        state["p_target"],
        torch.tensor([0.5000999500000001, 0.49990005], dtype=torch.float64),
    )
    assert float(state["prob_max_mu_t"]) == pytest.approx(0.7998925825000001)
    assert float(state["prob_max_var_t"]) == pytest.approx(0.03998332417500001)

    resumed = SoftMatchMethod(spec)
    resumed._load_paper_state(state)
    expected = method._paper_step(
        step_inputs["logits_l"],
        step_inputs["logits_uw_1"],
        step_inputs["logits_us"],
        step_inputs["y"],
        step_inputs["idx_u"],
    )
    actual = resumed._paper_step(
        step_inputs["logits_l"],
        step_inputs["logits_uw_1"],
        step_inputs["logits_us"],
        step_inputs["y"],
        step_inputs["idx_u"],
    )
    assert float(actual.loss.detach()) == pytest.approx(float(expected.loss.detach()))
    for name in ("p_model", "p_target", "prob_max_mu_t", "prob_max_var_t"):
        torch.testing.assert_close(resumed._paper_state()[name], method._paper_state()[name])


def test_checkpoint_current_is_transactional_and_sampler_resume_is_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    monkeypatch.setenv("MODSSC_CHECKPOINT_ROOT", str(checkpoint_root))
    identity = {
        "task_id": "task-7",
        "campaign_identity_sha256": "campaign-sha",
    }
    store = match_trainer._CheckpointStore(identity=identity)
    sampler = FixedSSLBatchSampler(
        5,
        9,
        labeled_batch_size=3,
        unlabeled_batch_size=4,
        seed=17,
        mode="shuffle_repeat",
        shuffle_buffer=8,
    )
    sampler.next_batch()
    payload_one = {
        "identity": identity,
        "next_step": 1,
        "sampler": copy.deepcopy(sampler.state_dict()),
        "trajectory": torch.tensor([1.0, 2.0]),
    }
    expected_next = sampler.next_batch()
    store.save(payload_one, step=1, reason="periodic")

    pointer_before = store.pointer_path.read_bytes()
    pointer = json.loads(pointer_before)
    generation_one = store.generations_root / pointer["generation"]
    assert generation_one.is_dir()
    assert store.payload_path == generation_one / "checkpoint.pt"
    assert store.metadata_path == generation_one / "checkpoint.json"

    original_atomic_json = match_trainer._atomic_json

    def fail_pointer(path: Path, payload: dict[str, object]) -> None:
        if path.name == "CURRENT.json":
            raise OSError("simulated pointer interruption")
        original_atomic_json(path, payload)

    monkeypatch.setattr(match_trainer, "_atomic_json", fail_pointer)
    with pytest.raises(OSError, match="pointer interruption"):
        store.save(
            {
                "identity": identity,
                "next_step": 2,
                "sampler": sampler.state_dict(),
                "trajectory": torch.tensor([9.0]),
            },
            step=2,
            reason="periodic",
        )
    assert store.pointer_path.read_bytes() == pointer_before

    monkeypatch.setenv("MODSSC_CHECKPOINT_RESUME", "1")
    recovered_store = match_trainer._CheckpointStore(identity=identity)
    recovered = recovered_store.load()
    assert recovered is not None
    assert recovered["next_step"] == 1
    torch.testing.assert_close(recovered["trajectory"], torch.tensor([1.0, 2.0]))
    resumed_sampler = FixedSSLBatchSampler(
        5,
        9,
        labeled_batch_size=3,
        unlabeled_batch_size=4,
        seed=17,
        mode="shuffle_repeat",
        shuffle_buffer=8,
    )
    resumed_sampler.load_state_dict(recovered["sampler"])
    actual_next = resumed_sampler.next_batch()
    np.testing.assert_array_equal(actual_next.labeled, expected_next.labeled)
    np.testing.assert_array_equal(actual_next.unlabeled, expected_next.unlabeled)
