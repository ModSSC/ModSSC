from __future__ import annotations

import math

import pytest

from modssc.hpo import (
    HpoError,
    PreparedTrial,
    Space,
    run_search,
    validate_space_targets,
)
from modssc.runtime import MethodExecutionOutcome, MethodNotEvaluableError


def test_native_search_owns_repeats_aggregation_and_tie_breaking() -> None:
    space = Space.from_dict({"method": {"params": {"value": [2, 1, 1]}}})
    seen: list[tuple[int, int]] = []

    result = run_search(
        space=space,
        kind="grid",
        seed=None,
        n_trials=None,
        repeats=2,
        direction="minimize",
        aggregate="mean",
        prepare_trial=lambda trial: PreparedTrial(
            value=int(trial.params["method.params.value"]),
            effective_patch=trial.patch,
        ),
        evaluate=lambda value, seed: seen.append((value, seed)) or float(value),
        repeat_seed=lambda trial, repeat: 100 * trial + repeat,
    )

    assert len(result.trials) == 3
    assert result.best.index == 1
    assert result.best.score == 1.0
    assert seen == [(2, 0), (2, 1), (1, 100), (1, 101), (1, 200), (1, 201)]


def test_native_search_marks_all_non_finite_objectives_not_evaluable() -> None:
    space = Space.from_dict({"method": {"params": {"value": [1, 2]}}})

    result = run_search(
        space=space,
        kind="grid",
        seed=None,
        n_trials=None,
        repeats=1,
        direction="maximize",
        aggregate="mean",
        prepare_trial=lambda trial: PreparedTrial(
            value=trial.index,
            effective_patch=trial.patch,
        ),
        evaluate=lambda _value, _seed: math.nan,
        repeat_seed=lambda _trial, _repeat: 0,
    )

    assert result.status == "not_evaluable"
    assert result.reason == "all_trial_objectives_non_finite"
    assert result.best is None
    assert [trial.status for trial in result.trials] == [
        "not_evaluable",
        "not_evaluable",
    ]
    assert [trial.score for trial in result.trials] == [None, None]
    assert [trial.values for trial in result.trials] == [(None,), (None,)]


def test_native_search_keeps_non_finite_trial_but_selects_finite_best() -> None:
    space = Space.from_dict({"method": {"params": {"value": [1, 2]}}})

    result = run_search(
        space=space,
        kind="grid",
        seed=None,
        n_trials=None,
        repeats=1,
        direction="maximize",
        aggregate="mean",
        prepare_trial=lambda trial: PreparedTrial(
            value=trial.index,
            effective_patch=trial.patch,
        ),
        evaluate=lambda value, _seed: math.nan if value == 0 else 0.5,
        repeat_seed=lambda _trial, _repeat: 0,
    )

    assert result.status == "success"
    assert result.best is not None
    assert result.best.index == 1
    assert result.trials[0].status == "not_evaluable"
    assert result.trials[0].score is None


def test_native_search_records_callback_failure_and_continues() -> None:
    space = Space.from_dict({"method": {"params": {"value": [1, 2]}}})

    def evaluate(value: int, _seed: int) -> float:
        if value == 0:
            raise RuntimeError("trial exploded")
        return 0.5

    result = run_search(
        space=space,
        kind="grid",
        seed=None,
        n_trials=None,
        repeats=1,
        direction="maximize",
        aggregate="mean",
        prepare_trial=lambda trial: PreparedTrial(
            value=trial.index,
            effective_patch=trial.patch,
        ),
        evaluate=evaluate,
        repeat_seed=lambda _trial, _repeat: 0,
    )

    assert result.status == "success"
    assert result.best is not None
    assert result.best.index == 1
    assert result.trials[0].status == "failed"
    assert result.trials[0].reason == "evaluation_error"
    assert result.trials[0].error_type == "RuntimeError"
    assert result.trials[0].error_message == "trial exploded"


def test_native_search_preserves_a_method_not_evaluable_outcome() -> None:
    error = MethodNotEvaluableError(
        MethodExecutionOutcome(
            status="not_evaluable",
            reason="solver did not converge",
            diagnostics={"converged": False},
        )
    )

    result = run_search(
        space=Space.from_dict({"method": {"params": {"value": [1]}}}),
        kind="grid",
        seed=None,
        n_trials=None,
        repeats=1,
        direction="maximize",
        aggregate="mean",
        prepare_trial=lambda trial: PreparedTrial(
            value=trial.index,
            effective_patch=trial.patch,
        ),
        evaluate=lambda _value, _seed: (_ for _ in ()).throw(error),
        repeat_seed=lambda _trial, _repeat: 0,
    )

    assert result.status == "not_evaluable"
    assert result.reason == "all_trials_not_evaluable"
    assert result.best is None
    assert result.trials[0].status == "not_evaluable"
    assert result.trials[0].reason == "evaluation_not_evaluable"
    assert result.trials[0].error_code == "E_METHOD_NOT_EVALUABLE"


def test_native_search_distinguishes_all_failed_from_not_evaluable() -> None:
    space = Space.from_dict({"method": {"params": {"value": [1]}}})

    result = run_search(
        space=space,
        kind="grid",
        seed=None,
        n_trials=None,
        repeats=1,
        direction="maximize",
        aggregate="mean",
        prepare_trial=lambda _trial: (_ for _ in ()).throw(ValueError("invalid trial")),
        evaluate=lambda _value, _seed: 0.5,
        repeat_seed=lambda _trial, _repeat: 0,
    )

    assert result.status == "failed"
    assert result.reason == "no_successful_trial"
    assert result.best is None
    assert result.trials[0].status == "failed"
    assert result.trials[0].reason == "prepare_trial_error"


def test_search_targets_cannot_change_runtime_contracts() -> None:
    validate_space_targets(
        {"method": {"params": {"alpha": [0.1, 0.2]}}},
        allowed_prefix=("method", "params"),
        forbidden_leaf_names=frozenset({"backend"}),
    )

    with pytest.raises(HpoError, match="runtime contract field"):
        validate_space_targets(
            {"method": {"params": {"backend": ["numpy", "torch"]}}},
            allowed_prefix=("method", "params"),
            forbidden_leaf_names=frozenset({"backend"}),
        )
