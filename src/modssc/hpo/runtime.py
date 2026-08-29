"""Native, runner-independent hyperparameter search execution."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import numpy as np

from .space import Space
from .types import HpoError, Trial

SearchKind = Literal["grid", "random"]
ObjectiveDirection = Literal["maximize", "minimize"]
ObjectiveAggregate = Literal["mean"]
TrialStatus = Literal["success", "failed", "not_evaluable"]
SearchStatus = Literal["success", "failed", "not_evaluable"]

PreparedT = TypeVar("PreparedT")

RUNTIME_CONTRACT_FIELDS = frozenset(
    {
        "backend",
        "classifier_backend",
        "device",
        "dtype",
        "profile",
    }
)


@dataclass(frozen=True)
class PreparedTrial(Generic[PreparedT]):
    """Caller-owned executable value and public metadata for one trial."""

    value: PreparedT
    effective_patch: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialResult:
    """One fully evaluated trial, including all repeated objective values."""

    index: int
    params: Mapping[str, Any]
    requested_patch: Mapping[str, Any]
    effective_patch: Mapping[str, Any]
    values: tuple[float | None, ...]
    score: float | None
    status: TrialStatus
    reason: str | None = None
    error_code: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """Native search result with deterministic best-trial selection."""

    trials: tuple[TrialResult, ...]
    best: TrialResult | None
    status: SearchStatus
    reason: str | None = None


def validate_space_targets(
    space: Mapping[str, Any],
    *,
    allowed_prefix: tuple[str, ...],
    forbidden_leaf_names: frozenset[str] = frozenset(),
) -> None:
    """Validate that every search leaf targets an allowed config subtree."""

    if not allowed_prefix or any(not isinstance(part, str) or not part for part in allowed_prefix):
        raise HpoError("allowed_prefix must contain non-empty path components")

    def check_path(path: tuple[str, ...]) -> None:
        if len(path) <= len(allowed_prefix) or path[: len(allowed_prefix)] != allowed_prefix:
            joined = ".".join(path) if path else "<root>"
            raise HpoError(f"search space leaf {joined!r} is outside {'.'.join(allowed_prefix)}.*")
        if path[-1] in forbidden_leaf_names:
            raise HpoError(f"search space cannot tune runtime contract field {'.'.join(path)!r}")

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, list):
            if not node:
                raise HpoError("search space leaves must be non-empty lists")
            check_path(path)
            return
        if isinstance(node, Mapping):
            if not node:
                raise HpoError("search space cannot contain empty mappings")
            if "dist" in node:
                check_path(path)
                return
            for key, value in node.items():
                if not isinstance(key, str) or not key:
                    raise HpoError("search space keys must be non-empty strings")
                walk(value, path + (key,))
            return
        raise HpoError("search space leaves must be lists or distribution specs")

    walk(space, ())


def _trial_iterator(
    space: Space,
    *,
    kind: SearchKind,
    seed: int | None,
    n_trials: int | None,
) -> Iterable[Trial]:
    if kind == "grid":
        return space.iter_grid()
    if kind != "random":
        raise HpoError("search kind must be 'grid' or 'random'")
    if seed is None:
        raise HpoError("random search requires a seed")
    if n_trials is None or isinstance(n_trials, bool) or int(n_trials) <= 0:
        raise HpoError("random search requires a positive n_trials")
    return space.iter_random(seed=int(seed), n_trials=int(n_trials))


def _aggregate(values: tuple[float, ...], aggregate: ObjectiveAggregate) -> float:
    if aggregate != "mean":
        raise HpoError(f"unknown objective aggregate: {aggregate}")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _is_better(
    candidate: TrialResult,
    best: TrialResult | None,
    *,
    direction: ObjectiveDirection,
) -> bool:
    if candidate.score is None:
        return False
    if best is None:
        return True
    if best.score is None:  # pragma: no cover - guarded by run_search
        return True
    if candidate.score == best.score:
        return candidate.index < best.index
    if direction == "maximize":
        return candidate.score > best.score
    if direction == "minimize":
        return candidate.score < best.score
    raise HpoError("objective direction must be 'maximize' or 'minimize'")


def run_search(
    *,
    space: Space,
    kind: SearchKind,
    seed: int | None,
    n_trials: int | None,
    repeats: int,
    direction: ObjectiveDirection,
    aggregate: ObjectiveAggregate,
    prepare_trial: Callable[[Trial], PreparedTrial[PreparedT]],
    evaluate: Callable[[PreparedT, int], float],
    repeat_seed: Callable[[int, int], int],
) -> SearchResult:
    """Execute search trials and select the best finite aggregate objective.

    The caller adapts configuration and method execution through callbacks;
    trial iteration, repeats, aggregation, non-finite handling, and
    tie-breaking stay native and reusable outside the benchmark YAML runner.
    """

    if isinstance(repeats, bool) or int(repeats) <= 0:
        raise HpoError("repeats must be a positive integer")
    if direction not in {"maximize", "minimize"}:
        raise HpoError("objective direction must be 'maximize' or 'minimize'")

    results: list[TrialResult] = []
    best: TrialResult | None = None
    for trial in _trial_iterator(space, kind=kind, seed=seed, n_trials=n_trials):
        try:
            prepared = prepare_trial(trial)
        except Exception as exc:
            results.append(
                TrialResult(
                    index=int(trial.index),
                    params=dict(trial.params),
                    requested_patch=dict(trial.patch),
                    effective_patch={},
                    values=(),
                    score=None,
                    status="failed",
                    reason="prepare_trial_error",
                    error_code=_exception_code(exc),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue
        if not isinstance(prepared, PreparedTrial):
            raise HpoError("prepare_trial must return PreparedTrial")
        raw_values: list[float] = []
        evaluation_error: Exception | None = None
        for repeat in range(int(repeats)):
            try:
                raw_values.append(
                    float(
                        evaluate(
                            prepared.value,
                            int(repeat_seed(trial.index, repeat)),
                        )
                    )
                )
            except Exception as exc:
                evaluation_error = exc
                break
        values: tuple[float | None, ...] = tuple(
            value if math.isfinite(value) else None for value in raw_values
        )
        if evaluation_error is not None:
            score = None
            status = _exception_trial_status(evaluation_error)
            reason = "evaluation_not_evaluable" if status == "not_evaluable" else "evaluation_error"
        elif all(value is not None for value in values):
            score = _aggregate(tuple(float(value) for value in values), aggregate)
            status = "success"
            reason = None
        else:
            score = None
            status = "not_evaluable"
            reason = "non_finite_objective"
        result = TrialResult(
            index=int(trial.index),
            params=dict(trial.params),
            requested_patch=dict(trial.patch),
            effective_patch=dict(prepared.effective_patch),
            values=values,
            score=score,
            status=status,
            reason=reason,
            error_code=(
                _exception_code(evaluation_error) if evaluation_error is not None else None
            ),
            error_type=(type(evaluation_error).__name__ if evaluation_error is not None else None),
            error_message=(str(evaluation_error) if evaluation_error is not None else None),
            metadata=dict(prepared.metadata),
        )
        results.append(result)
        if _is_better(result, best, direction=direction):
            best = result

    if not results:
        raise HpoError("search produced no trials")
    if best is None:
        if any(result.status == "failed" for result in results):
            return SearchResult(
                trials=tuple(results),
                best=None,
                status="failed",
                reason="no_successful_trial",
            )
        return SearchResult(
            trials=tuple(results),
            best=None,
            status="not_evaluable",
            reason=(
                "all_trial_objectives_non_finite"
                if all(result.reason == "non_finite_objective" for result in results)
                else "all_trials_not_evaluable"
            ),
        )
    return SearchResult(trials=tuple(results), best=best, status="success")


def _exception_code(exc: BaseException) -> str | None:
    code = getattr(exc, "code", None)
    return str(code) if isinstance(code, str) and code else None


def _exception_trial_status(exc: BaseException) -> TrialStatus:
    """Preserve a callback's explicit native non-evaluable outcome."""

    return "not_evaluable" if getattr(exc, "status", None) == "not_evaluable" else "failed"


__all__ = [
    "ObjectiveAggregate",
    "ObjectiveDirection",
    "PreparedTrial",
    "RUNTIME_CONTRACT_FIELDS",
    "SearchStatus",
    "SearchKind",
    "SearchResult",
    "TrialResult",
    "TrialStatus",
    "run_search",
    "validate_space_targets",
]
