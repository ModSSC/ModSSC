from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from modssc.hpo import (
    RUNTIME_CONTRACT_FIELDS,
    HpoError,
    PreparedTrial,
    Space,
    deep_merge,
    run_search,
    validate_space_targets,
)
from modssc.runtime.pipeline import MethodRuntimeResolution

from ..context import RunContext
from ..limits import apply_limits
from ..schema import BenchConfigError, ExperimentConfig
from . import evaluation as eval_orch
from . import method_inductive as inductive_orch
from . import method_transductive as transductive_orch

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedBenchTrial:
    cfg: ExperimentConfig
    requires_torch: bool


def _validate_space_targets(space: dict[str, Any]) -> None:
    try:
        validate_space_targets(
            space,
            allowed_prefix=("method", "params"),
            forbidden_leaf_names=RUNTIME_CONTRACT_FIELDS,
        )
    except HpoError as exc:
        raise BenchConfigError(str(exc), code="E_BENCH_HPO_SPACE") from exc


def run_hpo(
    *,
    ctx: RunContext,
    base_cfg: ExperimentConfig,
    base_cfg_dict: dict[str, Any],
    prepared_artifacts: dict[str, Any],
    method_runtime_resolver: Callable[[ExperimentConfig], MethodRuntimeResolution] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    search = base_cfg.search
    if search is None:
        raise ValueError("search config is required for HPO")

    start = perf_counter()
    _validate_space_targets(search.space)
    space = Space.from_dict(search.space)
    _LOGGER.info(
        "HPO start: kind=%s seed=%s n_trials=%s repeats=%s objective=%s",
        search.kind,
        search.seed,
        search.n_trials,
        search.repeats,
        {
            "split": search.objective.split,
            "metric": search.objective.metric,
            "direction": search.objective.direction,
            "aggregate": search.objective.aggregate,
        },
    )
    hpo_dir = ctx.run_dir / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    trials_path = hpo_dir / "trials.jsonl"

    def prepare_trial(trial: Any) -> PreparedTrial[_PreparedBenchTrial]:
        patched_dict = deep_merge(base_cfg_dict, trial.patch)
        limited_dict, changes, _ = apply_limits(
            patched_dict,
            limits=base_cfg.limits,
            strict=base_cfg.run.benchmark_mode,
        )
        if limited_dict.get("acceptance") is None:
            trial_cfg = ExperimentConfig.from_dict(limited_dict)
        else:
            trial_cfg = ExperimentConfig.from_dict(
                limited_dict,
                allow_resolved_acceptance_seed=True,
            )
        requires_torch = bool(prepared_artifacts.get("requires_torch", False))
        metadata: dict[str, Any] = {"limit_changes": list(changes)}
        if method_runtime_resolver is not None:
            method_runtime = method_runtime_resolver(trial_cfg)
            requires_torch = bool(method_runtime.requires_torch)
            metadata["method_runtime"] = method_runtime.to_dict()
        return PreparedTrial(
            value=_PreparedBenchTrial(
                cfg=trial_cfg,
                requires_torch=requires_torch,
            ),
            effective_patch={"method": {"params": dict(trial_cfg.method.params)}},
            metadata=metadata,
        )

    try:
        result = run_search(
            space=space,
            kind=search.kind,
            seed=search.seed,
            n_trials=search.n_trials,
            repeats=search.repeats,
            direction=search.objective.direction,
            aggregate=search.objective.aggregate,
            prepare_trial=prepare_trial,
            evaluate=lambda trial, trial_seed: _objective_value(
                cfg=trial.cfg,
                prepared_artifacts=prepared_artifacts,
                seed=trial_seed,
                split=search.objective.split,
                metric=search.objective.metric,
                requires_torch=trial.requires_torch,
            ),
            repeat_seed=lambda trial_index, repeat: ctx.seed_for(
                f"hpo-trial-{trial_index}-repeat-{repeat}"
            ),
        )
    except HpoError as exc:
        raise BenchConfigError(str(exc), code="E_BENCH_HPO_SPACE") from exc

    with trials_path.open("w", encoding="utf-8") as handle:
        for trial in result.trials:
            payload = {
                "index": trial.index,
                "params": dict(trial.params),
                "requested_patch": dict(trial.requested_patch),
                "effective_patch": dict(trial.effective_patch),
                "limit_changes": list(trial.metadata.get("limit_changes", [])),
                "method_runtime": trial.metadata.get("method_runtime"),
                "objective": {
                    "split": search.objective.split,
                    "metric": search.objective.metric,
                    "values": list(trial.values),
                    "value": trial.score,
                },
                "status": trial.status,
                "reason": trial.reason,
                "error_code": trial.error_code,
                "error_type": trial.error_type,
                "error_message": trial.error_message,
            }
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")

    summary: dict[str, Any] = {
        "kind": search.kind,
        "seed": search.seed,
        "n_trials": len(result.trials),
        "repeats": search.repeats,
        "status": result.status,
        "reason": result.reason,
        "objective": {
            "split": search.objective.split,
            "metric": search.objective.metric,
            "direction": search.objective.direction,
            "aggregate": search.objective.aggregate,
        },
        "trials_path": _relative_path(ctx.run_dir, trials_path),
    }
    best = result.best
    if best is None:
        summary.update(
            {
                "best_index": None,
                "best_score": None,
                "best_params": None,
                "best_patch": None,
                "best_effective_patch": None,
                "best_limit_changes": [],
            }
        )
        _LOGGER.warning(
            "HPO ended without a successful trial: status=%s reason=%s",
            result.status,
            result.reason,
        )
        return None, summary

    best_patch = dict(best.requested_patch)
    summary.update(
        {
            "best_index": best.index,
            "best_score": best.score,
            "best_params": dict(best.params),
            "best_patch": best_patch,
            "best_effective_patch": dict(best.effective_patch),
            "best_limit_changes": list(best.metadata.get("limit_changes", [])),
        }
    )
    _LOGGER.info("HPO best trial %s score=%s", best.index, best.score)
    _LOGGER.info(
        "HPO done: trials=%s duration_s=%.3f",
        len(result.trials),
        perf_counter() - start,
    )
    return best_patch, summary


def _objective_value(
    *,
    cfg: ExperimentConfig,
    prepared_artifacts: dict[str, Any],
    seed: int,
    split: str,
    metric: str,
    requires_torch: bool | None = None,
) -> float:
    routed_input = prepared_artifacts["routed_input"]
    execution_input = routed_input.execution_input
    strict = bool(prepared_artifacts.get("strict", False))
    if requires_torch is None:
        requires_torch = bool(prepared_artifacts.get("requires_torch", False))

    if cfg.method.kind == "inductive":
        method, _ = inductive_orch.run(
            execution_input,
            during_fit_splits=cfg.evaluation.during_fit_splits,
            cfg=cfg.method,
            seed=seed,
            strict=strict,
            requires_torch=requires_torch,
        )
        metrics = eval_orch.evaluate_inductive(
            method=method,
            pre=execution_input.preprocess,
            sampling=execution_input.sampling,
            report_splits=[split],
            metrics=[metric],
            views=execution_input.views,
            strict=strict,
        )
    else:
        method, data, _ = transductive_orch.run(
            execution_input,
            cfg=cfg.method,
            seed=seed,
            use_test_split=bool(prepared_artifacts["use_test"]),
            expected_labeled_count=routed_input.expected_labeled_count,
            strict=strict,
        )
        metrics = eval_orch.evaluate_transductive(
            method=method,
            data=data,
            report_splits=[split],
            metrics=[metric],
            masks=execution_input.masks,
        )

    return float(metrics[split][metric])


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
