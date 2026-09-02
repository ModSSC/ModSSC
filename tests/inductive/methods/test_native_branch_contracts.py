from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

import modssc.inductive.methods.helpers.match_trainer as match
from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods.co_training import CoTrainingMethod, CoTrainingSpec
from modssc.inductive.methods.tri_training import TriTrainingMethod


def _match_spec(**overrides):
    values = {
        "reference_implementation": "torchssl",
        "sampler_mode": "replacement",
        "augmentation_profile": "torchssl_ra",
        "reporting_policy": "best_historical_checkpoint",
        "sampler_shuffle_buffer": 8,
        "evaluation_interval_steps": 4,
        "evaluation_tail_interval_steps": 2,
        "evaluation_tail_start_fraction": 0.5,
        "checkpoint_interval_steps": 4,
        "reporting_window_checkpoints": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"evaluation_tail_interval_steps": None}, "configured together"),
        ({"evaluation_tail_interval_steps": 0}, "tail_interval_steps"),
        ({"evaluation_tail_start_fraction": 1.0}, "start_fraction"),
    ],
)
def test_match_trainer_rejects_invalid_tail_configuration(overrides, message: str) -> None:
    with pytest.raises(InductiveValidationError, match=message):
        match._trainer_config(_match_spec(**overrides))


def test_match_checkpoint_store_wraps_save_failures() -> None:
    checkpoint = match._CheckpointStore(identity={"id": 1}, context=None)
    checkpoint.store = SimpleNamespace(
        save=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError())
    )

    with pytest.raises(InductiveValidationError, match="save failed"):
        checkpoint.save({}, step=1, reason="periodic")


def test_match_checkpoint_load_normalizes_history_and_avoids_duplicate_event() -> None:
    event = {"step": 2, "reason": "periodic", "payload_sha256": "abc"}
    checkpoint = match._CheckpointStore(identity={"id": 1}, context=None)
    checkpoint.context = object()

    class Store:
        calls = 0

        def load_from_context(self, *_args, **_kwargs):
            self.calls += 1
            history = "invalid" if self.calls == 1 else [event]
            return SimpleNamespace(
                payload={"identity": {"id": 1}, "checkpoint_history": history},
                record=SimpleNamespace(**event),
            )

    checkpoint.store = Store()
    assert checkpoint.load() == {"identity": {"id": 1}}
    assert checkpoint.history == [event]
    assert checkpoint.load() == {"identity": {"id": 1}}
    assert checkpoint.history == [event]


def test_match_tensor_state_clone_supports_tuples() -> None:
    assert match._clone_tensor_state((1, {"x": [2]})) == (1, {"x": [2]})


def test_match_best_checkpoint_capture_validates_event_and_bundle(monkeypatch) -> None:
    with pytest.raises(InductiveValidationError, match="event is invalid"):
        match._capture_best_historical_checkpoint(
            object(), event={"step": 0, "test_accuracy": 0.8, "test_error_percent": 20.0}
        )
    with pytest.raises(InductiveValidationError, match="error is invalid"):
        match._capture_best_historical_checkpoint(
            object(), event={"step": 1, "test_accuracy": 0.8, "test_error_percent": 10.0}
        )

    monkeypatch.setattr(match, "_bundle_state", lambda _bundle: {"model": 1, "ema_model": None})
    with pytest.raises(InductiveValidationError, match="model state"):
        match._capture_best_historical_checkpoint(
            object(), event={"step": 1, "test_accuracy": 0.8, "test_error_percent": 20.0}
        )
    monkeypatch.setattr(match, "_bundle_state", lambda _bundle: {"model": {}, "ema_model": 1})
    with pytest.raises(InductiveValidationError, match="EMA state"):
        match._capture_best_historical_checkpoint(
            object(), event={"step": 1, "test_accuracy": 0.8, "test_error_percent": 20.0}
        )


def _history() -> list[dict[str, object]]:
    return [
        {"historical_eligible": True, "step": 1, "test_accuracy": 0.8},
        {"historical_eligible": True, "step": 2, "test_accuracy": 0.7},
    ]


def _best_raw() -> dict[str, object]:
    return {
        "schema_version": 1,
        "step": 1,
        "test_accuracy": 0.8,
        "test_error_percent": 20.0,
        "model_sha256": "model",
        "ema_model_sha256": None,
        "bundle": {"model": {}, "ema_model": None},
    }


def test_match_best_checkpoint_policy_guards() -> None:
    with pytest.raises(InductiveValidationError, match="unexpectedly"):
        match._validate_best_historical_checkpoint(
            {}, history=_history(), reporting_policy="median_last_checkpoints"
        )
    with pytest.raises(InductiveValidationError, match="no eligible"):
        match._validate_best_historical_checkpoint(
            {}, history=[], reporting_policy="best_historical_checkpoint"
        )
    with pytest.raises(InductiveValidationError, match="is missing"):
        match._validate_best_historical_checkpoint(
            None, history=_history(), reporting_policy="best_historical_checkpoint"
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema_version": 2}, "schema"),
        ({"step": object()}, "metadata"),
        ({"step": 2}, "disagrees"),
        ({"test_error_percent": 19.0}, "metric"),
        ({"bundle": 1}, "state"),
        ({"bundle": {"model": 1, "ema_model": None}}, "model is invalid"),
        ({"bundle": {"model": {}, "ema_model": 1}}, "EMA is invalid"),
        ({"model_sha256": "wrong"}, "digest"),
    ],
)
def test_match_best_checkpoint_rejects_corruption(monkeypatch, mutation, message: str) -> None:
    monkeypatch.setattr(match, "_tensor_group_sha256", lambda _state: "model")
    raw = _best_raw()
    raw.update(deepcopy(mutation))
    with pytest.raises(InductiveValidationError, match=message):
        match._validate_best_historical_checkpoint(
            raw,
            history=_history(),
            reporting_policy="best_historical_checkpoint",
        )


def test_tri_training_optional_evaluation_output_and_recorder(monkeypatch) -> None:
    method = TriTrainingMethod()
    assert method.predict_evaluation_outputs(object()) == {}
    method.record_evaluation_metrics(split="test", output="other", metrics={"accuracy": 1.0})
    assert "initial_evaluation" not in method.diagnostics_

    method._initial_clfs = [object()]
    monkeypatch.setattr(method, "predict_proba_initial", lambda _X: "scores")
    assert method.predict_evaluation_outputs(object()) == {"initial": "scores"}
    method.record_evaluation_metrics(split="test", output="initial", metrics={"accuracy": 1.0})
    assert method.diagnostics_["initial_evaluation"]["test"] == {"accuracy": 1.0}


def test_co_training_evaluation_output_protocol_branches(monkeypatch) -> None:
    method = CoTrainingMethod(
        CoTrainingSpec(
            protocol="shared_pool_exhaustive_multiset",
            classifier_id="multinomial_nb",
            classifier_backend="sklearn",
            p=1,
            n=3,
            u=75,
            k=0,
            positive_label=1,
            negative_label=0,
            selection_score="posterior_probability",
        )
    )
    monkeypatch.setattr(method, "predict_named_proba", lambda _data: {"named": "scores"})
    assert method.predict_evaluation_outputs(
        SimpleNamespace(meta={"evaluation_split": "test"})
    ) == {"named": "scores"}

    with pytest.raises(RuntimeError, match="missing view keys"):
        method.predict_evaluation_outputs(SimpleNamespace(meta={"evaluation_split": "val"}))
