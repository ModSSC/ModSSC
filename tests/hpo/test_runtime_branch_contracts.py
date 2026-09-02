from __future__ import annotations

from collections.abc import Iterator

import pytest

import modssc.hpo.runtime as runtime
from modssc.hpo import HpoError, PreparedTrial, Space, run_search, validate_space_targets
from modssc.hpo.types import Trial


@pytest.mark.parametrize(
    ("space", "prefix", "message"),
    [
        ({"method": {"params": {"x": [1]}}}, (), "allowed_prefix"),
        ({"method": {"params": {"x": [1]}}}, ("",), "allowed_prefix"),
        ({"outside": [1]}, ("method", "params"), "outside"),
        ({"method": {"params": {"x": []}}}, ("method", "params"), "non-empty lists"),
        ({"method": {"params": {}}}, ("method", "params"), "empty mappings"),
        ({"method": {"params": {1: [2]}}}, ("method", "params"), "non-empty strings"),
        ({"method": {"params": {"x": 1}}}, ("method", "params"), "lists or distribution"),
    ],
)
def test_validate_space_targets_rejects_every_malformed_shape(space, prefix, message) -> None:
    with pytest.raises(HpoError, match=message):
        validate_space_targets(space, allowed_prefix=prefix)


def test_validate_space_targets_accepts_distribution_leaf() -> None:
    validate_space_targets(
        {"method": {"params": {"x": {"dist": "uniform", "low": 0.0, "high": 1.0}}}},
        allowed_prefix=("method", "params"),
    )


class _IteratorSpace:
    def __init__(self, trials: tuple[Trial, ...] = ()) -> None:
        self.trials = trials
        self.random_call = None

    def iter_grid(self) -> Iterator[Trial]:
        return iter(self.trials)

    def iter_random(self, *, seed: int, n_trials: int) -> Iterator[Trial]:
        self.random_call = (seed, n_trials)
        return iter(self.trials[:n_trials])


def test_trial_iterator_validates_random_contract_and_delegates() -> None:
    space = _IteratorSpace((Trial(index=0, params={}, patch={}),))
    with pytest.raises(HpoError, match="kind"):
        runtime._trial_iterator(space, kind="bayes", seed=1, n_trials=1)
    with pytest.raises(HpoError, match="requires a seed"):
        runtime._trial_iterator(space, kind="random", seed=None, n_trials=1)
    for invalid in (None, True, 0):
        with pytest.raises(HpoError, match="positive n_trials"):
            runtime._trial_iterator(space, kind="random", seed=1, n_trials=invalid)

    assert list(runtime._trial_iterator(space, kind="random", seed=7, n_trials=1)) == list(
        space.trials
    )
    assert space.random_call == (7, 1)


def test_search_internal_comparisons_and_aggregate_guards() -> None:
    candidate = runtime.TrialResult(1, {}, {}, {}, (2.0,), 2.0, "success")
    later = runtime.TrialResult(2, {}, {}, {}, (1.0,), 1.0, "success")
    no_score = runtime.TrialResult(0, {}, {}, {}, (), None, "not_evaluable")

    with pytest.raises(HpoError, match="aggregate"):
        runtime._aggregate((1.0,), "median")
    assert runtime._is_better(candidate, no_score, direction="maximize")
    assert runtime._is_better(candidate, later, direction="maximize")
    with pytest.raises(HpoError, match="direction"):
        runtime._is_better(candidate, later, direction="sideways")


def test_run_search_validates_execution_contracts_and_empty_spaces() -> None:
    space = Space.from_dict({"method": {"params": {"x": [1]}}})
    common = dict(
        space=space,
        kind="grid",
        seed=None,
        n_trials=None,
        aggregate="mean",
        prepare_trial=lambda trial: PreparedTrial(trial.index, trial.patch),
        evaluate=lambda _value, _seed: 1.0,
        repeat_seed=lambda _trial, _repeat: 0,
    )
    for invalid in (True, 0):
        with pytest.raises(HpoError, match="repeats"):
            run_search(repeats=invalid, direction="maximize", **common)
    with pytest.raises(HpoError, match="direction"):
        run_search(repeats=1, direction="sideways", **common)
    with pytest.raises(HpoError, match="PreparedTrial"):
        run_search(
            repeats=1,
            direction="maximize",
            **{**common, "prepare_trial": lambda _trial: object()},
        )
    with pytest.raises(HpoError, match="no trials"):
        run_search(
            repeats=1,
            direction="maximize",
            **{**common, "space": _IteratorSpace()},
        )


def test_callback_error_code_is_preserved() -> None:
    class CodedError(RuntimeError):
        code = "E_TEST"

    result = run_search(
        space=Space.from_dict({"method": {"params": {"x": [1]}}}),
        kind="grid",
        seed=None,
        n_trials=None,
        repeats=1,
        direction="maximize",
        aggregate="mean",
        prepare_trial=lambda _trial: (_ for _ in ()).throw(CodedError("bad")),
        evaluate=lambda _value, _seed: 1.0,
        repeat_seed=lambda _trial, _repeat: 0,
    )

    assert result.trials[0].error_code == "E_TEST"
