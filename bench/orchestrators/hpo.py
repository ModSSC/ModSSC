from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from modssc.hpo import Space, deep_merge

from ..context import RunContext
from ..schema import BenchConfigError, ExperimentConfig
from . import evaluation as eval_orch
from . import method_inductive as inductive_orch
from . import method_transductive as transductive_orch

_LOGGER = logging.getLogger(__name__)


def _validate_space_targets(space: Mapping[str, Any]) -> None:
    def _check_path(path: tuple[str, ...]) -> None:
        if len(path) < 3 or path[0] != "method" or path[1] != "params":
            joined = ".".join(path) if path else "<root>"
            raise BenchConfigError(
                f"search.space can only target method.params.*; got leaf at {joined!r}."
            )

    def _walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, list):
            if not node:
                raise BenchConfigError("search.space leaves must be non-empty lists")
            _check_path(path)
            return
        if isinstance(node, Mapping):
            if not node:
                raise BenchConfigError("search.space cannot contain empty mappings")
            if "dist" in node:
                _check_path(path)
                return
            for key, value in node.items():
                if not isinstance(key, str) or not key:
                    raise BenchConfigError("search.space keys must be non-empty strings")
                _walk(value, path + (key,))
            return
        raise BenchConfigError("search.space leaves must be lists or dist specs")

    _walk(space, ())


def run_hpo(
    *,
    ctx: RunContext,
    base_cfg: ExperimentConfig,
    base_cfg_dict: dict[str, Any],
    prepared_artifacts: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    if search.kind == "grid":
        trials_iter = space.iter_grid()
    else:
        trials_iter = space.iter_random(seed=int(search.seed), n_trials=int(search.n_trials))

    hpo_dir = ctx.run_dir / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    trials_path = hpo_dir / "trials.jsonl"

    best_patch: dict[str, Any] | None = None
    best_params: dict[str, Any] | None = None
    best_score: float | None = None
    best_cmp: float | None = None
    best_index: int | None = None
    n_trials = 0

    with trials_path.open("w", encoding="utf-8") as handle:
        for trial in trials_iter:
            n_trials += 1
            trial_start = perf_counter()
            _LOGGER.debug("HPO trial %s params=%s", trial.index, trial.params)
            patched_dict = deep_merge(base_cfg_dict, trial.patch)
            trial_cfg = ExperimentConfig.from_dict(patched_dict)

            scores = []
            for repeat in range(int(search.repeats)):
                seed = ctx.seed_for(f"hpo-trial-{trial.index}-repeat-{repeat}")
                value = _objective_value(
                    cfg=trial_cfg,
                    prepared_artifacts=prepared_artifacts,
                    seed=seed,
                    split=search.objective.split,
                    metric=search.objective.metric,
                )
                scores.append(float(value))

            agg_score = _aggregate(scores, search.objective.aggregate)
            cmp_score = agg_score
            if np.isnan(agg_score):
                cmp_score = -np.inf if search.objective.direction == "maximize" else np.inf

            if _is_better(
                candidate=cmp_score,
                best=best_cmp,
                direction=search.objective.direction,
                index=trial.index,
                best_index=best_index,
            ):
                best_cmp = cmp_score
                best_score = agg_score
                best_patch = trial.patch
                best_params = trial.params
                best_index = trial.index

            payload = {
                "index": trial.index,
                "params": trial.params,
                "patch": trial.patch,
                "objective": {
                    "split": search.objective.split,
                    "metric": search.objective.metric,
                    "values": scores,
                    "value": agg_score,
                },
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            _LOGGER.debug(
                "HPO trial %s objective=%s duration_s=%.3f",
                trial.index,
                {"values": scores, "value": agg_score},
                perf_counter() - trial_start,
            )

    if best_patch is None or best_params is None or best_index is None:
        raise ValueError("HPO produced no trials")

    summary = {
        "kind": search.kind,
        "seed": search.seed,
        "n_trials": n_trials,
        "repeats": search.repeats,
        "objective": {
            "split": search.objective.split,
            "metric": search.objective.metric,
            "direction": search.objective.direction,
            "aggregate": search.objective.aggregate,
        },
        "best_index": best_index,
        "best_score": best_score,
        "best_params": best_params,
        "best_patch": best_patch,
        "trials_path": _relative_path(ctx.run_dir, trials_path),
    }
    _LOGGER.info("HPO best trial %s score=%s", best_index, best_score)
    _LOGGER.info("HPO done: trials=%s duration_s=%.3f", n_trials, perf_counter() - start)
    return best_patch, summary


def _objective_value(
    *,
    cfg: ExperimentConfig,
    prepared_artifacts: dict[str, Any],
    seed: int,
    split: str,
    metric: str,
) -> float:
    pre = prepared_artifacts["pre"]
    sampling = prepared_artifacts["sampling"]
    views = prepared_artifacts["views"]
    graph = prepared_artifacts["graph"]
    masks = prepared_artifacts["masks"]
    expected_labeled_count = prepared_artifacts.get("expected_labeled_count")
    X_u_w = prepared_artifacts["X_u_w"]
    X_u_s = prepared_artifacts["X_u_s"]
    X_u_s_1 = prepared_artifacts.get("X_u_s_1")
    use_test = prepared_artifacts["use_test"]
    strict = bool(prepared_artifacts.get("strict", False))
    requires_torch = bool(prepared_artifacts.get("requires_torch", False))

    if cfg.method.kind == "inductive":
        method, _ = inductive_orch.run(
            pre,
            sampling,
            views=views,
            X_u_w=X_u_w,
            X_u_s=X_u_s,
            X_u_s_1=X_u_s_1,
            cfg=cfg.method,
            seed=seed,
            strict=strict,
            requires_torch=requires_torch,
        )
        metrics = eval_orch.evaluate_inductive(
            method=method,
            pre=pre,
            sampling=sampling,
            report_splits=[split],
            metrics=[metric],
            method_id=cfg.method.method_id,
            views=views,
            strict=strict,
        )
    else:
        if graph is None:
            raise ValueError("Transductive HPO requires a graph")
        method, data, _ = transductive_orch.run(
            dataset=pre.dataset,
            graph=graph,
            masks=masks,
            cfg=cfg.method,
            seed=seed,
            use_test_split=use_test,
            expected_labeled_count=expected_labeled_count,
            strict=strict,
        )
        metrics = eval_orch.evaluate_transductive(
            method=method,
            data=data,
            report_splits=[split],
            metrics=[metric],
            masks=masks,
        )

    return float(metrics[split][metric])


def _aggregate(values: list[float], aggregate: str) -> float:
    if aggregate == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unknown aggregate: {aggregate}")


def _is_better(
    *,
    candidate: float,
    best: float | None,
    direction: str,
    index: int,
    best_index: int | None,
) -> bool:
    if best is None or best_index is None:
        return True
    if candidate == best:
        return index < best_index
    if direction == "maximize":
        return candidate > best
    return candidate < best


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
