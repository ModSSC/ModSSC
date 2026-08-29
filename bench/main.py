from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_augmentation import prepare_unlabeled_augmentation
from modssc.evaluation import AcceptanceSpec, assess_evaluation_metrics, list_metrics
from modssc.hpo import deep_merge
from modssc.preprocess import (
    PreprocessPlan,
    resolve_fit_indices,
    steps_require_fit_indices,
    steps_with_runtime_role,
)
from modssc.runtime.continuation import (
    PLANNED_CONTINUATION_EXIT_CODE,
    PlannedContinuation,
    continuation_signal_handler,
)
from modssc.runtime.dependencies import (
    PipelineDependencyError,
    PipelineDependencyRequest,
    PipelineDependencyResolution,
    resolve_pipeline_dependencies,
)
from modssc.runtime.execution import ExecutionContext
from modssc.runtime.input_routing import ScientificInputRequest, route_scientific_input
from modssc.runtime.logging import configure_logging, resolve_log_level
from modssc.runtime.pipeline import (
    MethodResolutionRequest,
    MethodRuntimeResolution,
    PipelineResolutionError,
    resolve_method,
)
from modssc.sampling.plan import SamplingPlan
from modssc.sampling.result import SamplingResult
from modssc.sampling.storage import save_split
from modssc.transductive.data import graph_from_dataset
from modssc.views import ViewsPlan

from .context import RunContext, next_available_run_dir
from .errors import BenchRuntimeError, extract_error_code
from .execution_contracts import (
    EXECUTION_CONTRACT_ERROR_CODE,
    persist_execution_contract_from_error,
    persist_execution_contract_from_resolution,
)
from .limits import apply_limits
from .orchestrators import dataset as ds_orch
from .orchestrators import evaluation as eval_orch
from .orchestrators import graph as graph_orch
from .orchestrators import hpo as hpo_orch
from .orchestrators import input_artifacts as input_artifact_orch
from .orchestrators import method_inductive as inductive_orch
from .orchestrators import method_transductive as transductive_orch
from .orchestrators import preprocess as prep_orch
from .orchestrators import reporting as report_orch
from .orchestrators import sampling as sampling_orch
from .orchestrators import views as views_orch
from .schema import BenchConfigError, ExperimentConfig
from .seed_sweep import apply_global_seed, sweep_run_name
from .utils.hashing import hash_any
from .utils.identity import build_resume_identity, protocol_sha256
from .utils.import_tools import check_extra_installed, distributions_for_extras
from .utils.io import atomic_write_json, load_yaml
from .utils.runtime import collect_runtime_versions

_ALLOWED_METRICS = set(list_metrics())
_ALLOWED_SPLITS = {"train", "train_labeled", "val", "test", "unlabeled"}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleRunResult:
    code: int
    run_dir: Path
    run_json_path: Path


def _positive_int(value: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if out <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return out


def _non_negative_int(value: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if out < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return out


def _check_extra(extra: str) -> None:
    try:
        missing = check_extra_installed(extra)
    except ValueError as exc:
        raise BenchConfigError(
            str(exc),
            code="E_BENCH_DEPENDENCY_UNKNOWN_EXTRA",
        ) from exc
    if missing:
        raise BenchConfigError(
            f"Missing optional dependency for extra '{extra}': {sorted(set(missing))}",
            code="E_BENCH_DEPENDENCY_MISSING",
        )


def _preflight(
    *,
    cfg: ExperimentConfig,
) -> None:
    if cfg.augmentation is not None and cfg.augmentation.enabled:
        if cfg.augmentation.mode not in {"fixed", "online"}:
            raise BenchConfigError(
                "augmentation.mode must be 'fixed' or 'online'",
                code="E_BENCH_CONFIG",
            )
        if not cfg.augmentation.weak or not cfg.augmentation.strong:
            raise BenchConfigError(
                "augmentation.weak and augmentation.strong must be provided",
                code="E_BENCH_CONFIG",
            )

    for metric in cfg.evaluation.metrics:
        if metric not in _ALLOWED_METRICS:
            raise BenchConfigError(f"Unknown metric: {metric}", code="E_BENCH_CONFIG")

    for split in cfg.evaluation.report_splits:
        if split not in _ALLOWED_SPLITS:
            raise BenchConfigError(f"Unknown split: {split}", code="E_BENCH_CONFIG")
        if split == "unlabeled" and cfg.method.kind != "transductive":
            raise BenchConfigError(
                "evaluation split 'unlabeled' is only supported for transductive methods",
                code="E_BENCH_CONFIG",
            )
    if cfg.evaluation.during_fit_splits and cfg.method.kind != "inductive":
        raise BenchConfigError(
            "evaluation.during_fit_splits is supported only for inductive methods",
            code="E_BENCH_CONFIG",
        )
    if len(set(cfg.evaluation.during_fit_splits)) != len(cfg.evaluation.during_fit_splits):
        raise BenchConfigError(
            "evaluation.during_fit_splits must not contain duplicates",
            code="E_BENCH_CONFIG",
        )
    for split in cfg.evaluation.during_fit_splits:
        if split not in _ALLOWED_SPLITS - {"unlabeled"}:
            raise BenchConfigError(
                f"Unknown during-fit evaluation split: {split}",
                code="E_BENCH_CONFIG",
            )


_PIPELINE_BENCH_ERROR_CODES = {
    "method_lookup": "E_BENCH_METHOD_INTROSPECTION",
    "method_introspection": "E_BENCH_METHOD_INTROSPECTION",
    "method_spec": "E_BENCH_METHOD_SPEC",
    "auto_device": "E_BENCH_AUTO_FORBIDDEN",
    "auto_backend": "E_BENCH_AUTO_FORBIDDEN",
    "backend_required": "E_BENCH_BACKEND_REQUIRED",
    "torch_preprocess": "E_BENCH_PREPROCESS_TO_TORCH_REQUIRED",
    "capability": "E_BENCH_CAPABILITY",
}


def _bench_pipeline_error(exc: PipelineResolutionError) -> BenchRuntimeError:
    return BenchRuntimeError(_PIPELINE_BENCH_ERROR_CODES[exc.kind], str(exc))


def _preprocess_logged_artifacts(pre: Any) -> dict[str, Any]:
    """Collect step-owned runtime metadata without naming preprocessing bricks."""

    out: dict[str, Any] = {}
    by_split: dict[str, dict[str, Any]] = {}
    for split, store in (("train", pre.train_artifacts), ("test", pre.test_artifacts)):
        if store is None:
            continue
        split_out: dict[str, Any] = {}
        for key in store:
            if not key.endswith(".info"):
                continue
            value = store.get(key)
            if value is None:
                continue
            split_out[key] = value
            if split == "train":
                out[key] = value
        if split_out:
            by_split[split] = split_out
    if by_split:
        out["by_split"] = by_split
    return out


def _write_error_traceback(ctx: RunContext, tb: str) -> None:
    try:
        (ctx.run_dir / "error.txt").write_text(tb, encoding="utf-8")
    except OSError:
        _LOGGER.exception("Failed to write error.txt")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _persist_sampling_replay(
    ctx: RunContext,
    sampling: SamplingResult,
) -> dict[str, Any]:
    """Persist the exact sampled partition in a portable, reloadable form."""

    relative_path = Path("sampling_split")
    replay_dir = ctx.run_dir / relative_path
    manifest_name = "MANIFEST.json"
    save_split(sampling, replay_dir, overwrite=False)
    files = {
        name: {"sha256": _file_sha256(replay_dir / name)} for name in ("split.json", "arrays.npz")
    }
    atomic_write_json(
        replay_dir / manifest_name,
        {
            "schema_version": 1,
            "format": "modssc.sampling.storage.v1",
            "dataset_fingerprint": sampling.dataset_fingerprint,
            "split_fingerprint": sampling.split_fingerprint,
            "files": files,
        },
    )
    return {
        "format": "modssc.sampling.storage.v1",
        "path": relative_path.as_posix(),
        "manifest": manifest_name,
        "manifest_sha256": _file_sha256(replay_dir / manifest_name),
    }


def _scan_auto_entries(node: Any, *, path: str) -> list[str]:
    paths: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            if (
                isinstance(value, str)
                and value.strip().lower() == "auto"
                and key
                in {
                    "device",
                    "backend",
                    "classifier_backend",
                    "profile",
                }
            ):
                paths.append(child)
            paths.extend(_scan_auto_entries(value, path=child))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            paths.extend(_scan_auto_entries(value, path=f"{path}[{i}]"))
    return paths


def _benchmark_contract_preflight(
    *,
    cfg: ExperimentConfig,
    raw: Mapping[str, Any],
    preprocess_steps: list[str],
    view_preprocess_steps: list[str],
) -> None:
    if not cfg.run.benchmark_mode:
        return

    if cfg.dataset.download:
        raise BenchConfigError(
            "dataset.download must be false when run.benchmark_mode=true",
            code="E_BENCH_DOWNLOAD_FORBIDDEN",
        )
    if not cfg.dataset.cache_dir:
        raise BenchConfigError(
            "dataset.cache_dir must be explicitly set when run.benchmark_mode=true",
            code="E_BENCH_CACHE_DIR_REQUIRED",
        )
    if cfg.method.device.device == "auto":
        raise BenchConfigError(
            "method.device.device='auto' is forbidden when run.benchmark_mode=true",
            code="E_BENCH_AUTO_FORBIDDEN",
        )
    if cfg.limits is not None and cfg.limits.profile == "auto":
        raise BenchConfigError(
            "limits.profile='auto' is forbidden when run.benchmark_mode=true",
            code="E_BENCH_AUTO_FORBIDDEN",
        )
    if cfg.graph is not None and cfg.graph.enabled:
        graph_backend = cfg.graph.spec.get("backend")
        if isinstance(graph_backend, str) and graph_backend.lower() == "auto":
            raise BenchConfigError(
                "graph.spec.backend='auto' is forbidden when run.benchmark_mode=true",
                code="E_BENCH_AUTO_FORBIDDEN",
            )

    if cfg.method.model is not None and cfg.method.model.classifier_backend == "auto":
        raise BenchConfigError(
            "method.model.classifier_backend='auto' is forbidden when run.benchmark_mode=true",
            code="E_BENCH_AUTO_FORBIDDEN",
        )

    evaluation_raw = raw.get("evaluation")
    selection_split_declared = (
        isinstance(evaluation_raw, Mapping) and "split_for_model_selection" in evaluation_raw
    )
    selection_split = cfg.evaluation.split_for_model_selection
    search_enabled = cfg.search is not None and cfg.search.enabled
    if selection_split is None and (
        not selection_split_declared or search_enabled or bool(cfg.evaluation.during_fit_splits)
    ):
        raise BenchConfigError(
            "evaluation.split_for_model_selection must be set when model selection is performed "
            "in benchmark_mode; declare it explicitly as null for a fixed terminal protocol",
            code="E_BENCH_SPLIT_MODEL_SELECTION_REQUIRED",
        )
    if selection_split is not None and selection_split not in cfg.evaluation.report_splits:
        raise BenchConfigError(
            "evaluation.split_for_model_selection must be included in evaluation.report_splits",
            code="E_BENCH_SPLIT_MODEL_SELECTION_INVALID",
        )
    test_selection_policy = cfg.evaluation.test_selection_policy
    if test_selection_policy == "paper_protocol" and selection_split != "test":
        raise BenchConfigError(
            "evaluation.test_selection_policy='paper_protocol' is valid only when "
            "evaluation.split_for_model_selection='test'",
            code="E_BENCH_TEST_SELECTION_POLICY_INVALID",
        )
    if selection_split == "test" and test_selection_policy != "paper_protocol":
        raise BenchConfigError(
            "evaluation.split_for_model_selection='test' is forbidden in benchmark_mode "
            "unless evaluation.test_selection_policy='paper_protocol' is explicitly declared",
            code="E_BENCH_TEST_SELECTION_FORBIDDEN",
        )
    if search_enabled and cfg.search is not None and cfg.search.objective.split != selection_split:
        raise BenchConfigError(
            "search.objective.split must match evaluation.split_for_model_selection in benchmark_mode",
            code="E_BENCH_SPLIT_MODEL_SELECTION_CONFLICT",
        )

    auto_paths = _scan_auto_entries(raw, path="")
    if auto_paths:
        raise BenchConfigError(
            "auto is forbidden in benchmark_mode at: " + ", ".join(sorted(set(auto_paths))),
            code="E_BENCH_AUTO_FORBIDDEN",
        )

    if steps_require_fit_indices(preprocess_steps) and cfg.preprocess.fit_on is None:
        raise BenchConfigError(
            "preprocess.fit_on must be set when preprocess includes fittable steps",
            code="E_BENCH_FIT_ON_REQUIRED",
        )
    if steps_require_fit_indices(view_preprocess_steps) and cfg.preprocess.fit_on is None:
        raise BenchConfigError(
            "preprocess.fit_on must be set when views preprocess includes fittable steps",
            code="E_BENCH_FIT_ON_REQUIRED",
        )


def _resolve_log_level_for_run(config_path: Path, cli_log_level: str | None) -> str:
    raw = load_yaml(config_path)
    cfg = ExperimentConfig.from_dict(raw)
    if cli_log_level is not None and str(cli_log_level).strip():
        resolved = resolve_log_level(cli_log_level)
    else:
        resolved = resolve_log_level(cfg.run.log_level)
    if cfg.run.benchmark_mode and resolved == "none":
        return "basic"
    return resolved


def _collect_code_runtime_versions(
    *,
    required_distributions: tuple[str, ...] = (),
    require_complete_manifest: bool = False,
) -> dict[str, Any]:
    """Collect provenance from the repository containing the benchmark code."""

    return collect_runtime_versions(
        repo_root=Path(__file__).resolve().parent,
        required_distributions=required_distributions,
        require_complete_manifest=require_complete_manifest,
    )


def _resolve_method_runtime(
    cfg: ExperimentConfig,
    *,
    preprocess_steps: list[str],
) -> MethodRuntimeResolution:
    """Adapt a parsed method block to the native runtime resolver."""

    try:
        return resolve_method(
            MethodResolutionRequest(
                regime=cfg.method.kind,
                method_id=cfg.method.method_id,
                params=cfg.method.params,
                requested_device=cfg.method.device.device,
                dtype=cfg.method.device.dtype,
                strict=cfg.run.benchmark_mode,
                preprocess_step_ids=tuple(preprocess_steps),
                model_classifier_id=(
                    cfg.method.model.classifier_id if cfg.method.model is not None else None
                ),
                model_classifier_backend=(
                    cfg.method.model.classifier_backend if cfg.method.model is not None else None
                ),
                model_configured=cfg.method.model is not None,
            )
        )
    except PipelineResolutionError as exc:
        raise _bench_pipeline_error(exc) from exc


def _required_software_distributions(
    *,
    cfg: ExperimentConfig,
    preprocess_plan: PreprocessPlan,
    views_plan: ViewsPlan | None,
    method: MethodRuntimeResolution,
) -> tuple[tuple[str, ...], PipelineDependencyResolution]:
    """Expand native component declarations through local package metadata."""

    step_ids = list(preprocess_plan.enabled_step_ids())
    if views_plan is not None:
        step_ids.extend(views_plan.preprocess_step_ids())

    try:
        resolution = resolve_pipeline_dependencies(
            PipelineDependencyRequest(
                dataset_id=cfg.dataset.id,
                preprocess_step_ids=tuple(step_ids),
                method_required_extra=method.required_extra,
                method_required_extras=method.required_extras,
                classifier_id=(
                    cfg.method.model.classifier_id
                    if cfg.method.model is not None and cfg.method.model.factory is None
                    else None
                ),
                classifier_backend=(
                    cfg.method.model.classifier_backend
                    if cfg.method.model is not None and cfg.method.model.factory is None
                    else None
                ),
                graph_spec=(
                    dict(cfg.graph.spec) if cfg.graph is not None and cfg.graph.enabled else None
                ),
            )
        )
        for extra in resolution.extras:
            _check_extra(extra)
        return (
            distributions_for_extras(
                resolution.extras,
                explicit=tuple(cfg.run.software_dependencies),
            ),
            resolution,
        )
    except BenchConfigError:
        raise
    except (PipelineDependencyError, ValueError) as exc:
        raise BenchConfigError(
            f"Unable to resolve software dependency manifest: {exc}",
            code="E_BENCH_SOFTWARE_PROVENANCE",
        ) from exc


def _sync_ctx_run_identity(ctx: RunContext, *, run_id: str) -> None:
    if ctx.run_id == run_id:
        return

    old_run_id = ctx.run_id
    old_run_dir = ctx.run_dir
    prefix = f"{ctx.name}-{old_run_id}-"
    if old_run_dir.name.startswith(prefix):
        suffix = old_run_dir.name[len(prefix) :]
    else:
        suffix = old_run_dir.name.rsplit("-", 1)[-1]
    desired_dir = ctx.output_dir / f"{ctx.name}-{run_id}-{suffix}"
    new_run_dir = next_available_run_dir(desired_dir)
    if new_run_dir != desired_dir:
        _LOGGER.warning(
            "Run directory collision while updating run_id; using alternate path: %s",
            new_run_dir,
        )

    old_run_dir.rename(new_run_dir)
    ctx.run_id = str(run_id)
    ctx.run_dir = new_run_dir
    _LOGGER.info("Run identity updated after config mutation: %s -> %s", old_run_id, run_id)


def _resolve_method_seed(ctx: RunContext, cfg: ExperimentConfig) -> int:
    """Resolve the method RNG, honoring an explicitly configured seed."""

    return ctx.seed_for("method", cfg.run.model_seed)


def _hpo_not_evaluable_message(summary: Mapping[str, Any]) -> str:
    reason = summary.get("reason")
    return "HPO produced no evaluable trial" + (f": {reason}" if reason else "")


def _run_experiment_single(
    config_path: Path, *, raw: dict[str, Any], cfg: ExperimentConfig
) -> SingleRunResult:
    resource_measurement = report_orch.begin_run_resource_measurement()
    requested_raw = deep_merge({}, raw)
    config_hash = hash_any(requested_raw)

    raw, limit_changes, resolved_limits = apply_limits(
        raw, limits=cfg.limits, strict=cfg.run.benchmark_mode
    )
    if limit_changes:
        profile = resolved_limits.profile if resolved_limits is not None else None
        profile_label = profile or "custom"
        _LOGGER.info(
            "Applied memory limits: profile=%s changes=%s", profile_label, len(limit_changes)
        )
        _LOGGER.debug("Limit adjustments: %s", limit_changes)
        cfg = ExperimentConfig.from_dict(raw, allow_resolved_acceptance_seed=True)

    preprocess_plan = PreprocessPlan.from_dict(cfg.preprocess.plan)
    views_plan = ViewsPlan.from_dict(cfg.views.plan) if cfg.views is not None else None
    preprocess_steps = list(preprocess_plan.enabled_step_ids())
    view_preprocess_steps = list(views_plan.preprocess_step_ids()) if views_plan is not None else []
    method_runtime = _resolve_method_runtime(cfg, preprocess_steps=preprocess_steps)
    required_distributions, dependency_resolution = _required_software_distributions(
        cfg=cfg,
        preprocess_plan=preprocess_plan,
        views_plan=views_plan,
        method=method_runtime,
    )

    effective_config_hash = hash_any(raw)
    versions = _collect_code_runtime_versions(
        required_distributions=required_distributions,
        require_complete_manifest=cfg.run.resume_policy != "never",
    )
    resume_identity = build_resume_identity(
        raw,
        seed=int(cfg.run.seed),
        runtime_versions=versions,
    )
    report_config = raw
    report_versions = versions
    run_id = resume_identity.short_id

    ctx = RunContext.from_run_config(
        name=cfg.run.name,
        seed=cfg.run.seed,
        run_id=run_id,
        output_dir=cfg.run.output_dir,
        config_path=config_path,
        fail_fast=cfg.run.fail_fast,
    )
    ctx.ensure_dirs()
    ctx.write_config_copy(raw)

    _LOGGER.info("Run started: %s", cfg.run.name)
    _LOGGER.info("Config: %s", config_path)
    _LOGGER.info("Run dir: %s", ctx.run_dir)

    artifacts: dict[str, Any] = {}
    metrics: dict[str, Any] | None = None
    hpo_summary: dict[str, Any] | None = None
    status = "success"
    error: str | None = None
    error_code: str | None = None
    fallback_events: list[dict[str, Any]] = []
    verified_input_artifacts: input_artifact_orch.InputArtifactPreflight | None = None

    normalization_requested = list(
        steps_with_runtime_role(preprocess_plan.enabled_step_ids(), role="normalization")
    )
    if views_plan is not None:
        normalization_requested.extend(
            steps_with_runtime_role(
                views_plan.preprocess_step_ids(),
                role="normalization",
            )
        )

    resolution: dict[str, Any] = {
        "device": {"requested": cfg.method.device.device, "resolved": None},
        "backend": {
            "requested": {
                "method": cfg.method.params.get("backend"),
                "classifier": (
                    cfg.method.model.classifier_backend if cfg.method.model is not None else None
                ),
                "graph": cfg.graph.spec.get("backend") if cfg.graph is not None else None,
            },
            "resolved": {
                **(
                    {"graph": dependency_resolution.resolved_graph_backend}
                    if dependency_resolution.resolved_graph_backend is not None
                    else {}
                )
            },
        },
        "dtype": {
            "requested": {"method_device_dtype": cfg.method.device.dtype},
            "resolved": {},
        },
        "normalization": {
            "requested": {"preprocess_steps": sorted(set(normalization_requested))},
            "resolved": {},
        },
        "splits": {
            "requested": list(cfg.evaluation.report_splits),
            "resolved": {},
        },
        "limits": {
            "requested": asdict(cfg.limits) if cfg.limits is not None else None,
            "resolved": asdict(resolved_limits) if resolved_limits is not None else None,
            "changes": list(limit_changes),
        },
    }
    protocol: dict[str, Any] = {
        "kind": cfg.method.kind,
        "use_test_split": None,
        "report_splits": list(cfg.evaluation.report_splits),
        "split_for_model_selection": cfg.evaluation.split_for_model_selection,
        "test_selection_policy": cfg.evaluation.test_selection_policy,
    }

    try:
        if cfg.run.input_artifacts:
            if cfg.run.artifact_root is None:  # guarded by the schema
                raise BenchConfigError(
                    "run.artifact_root is required when run.input_artifacts is not empty"
                )
            _LOGGER.info("Verifying %s declared input artifact(s)", len(cfg.run.input_artifacts))
            verified_input_artifacts = input_artifact_orch.preflight(
                cfg.run.input_artifacts,
                artifact_root=cfg.run.artifact_root,
                config_path=config_path,
            )
            artifacts["input_artifacts"] = verified_input_artifacts.report_payload()

        _benchmark_contract_preflight(
            cfg=cfg,
            raw=raw,
            preprocess_steps=preprocess_steps,
            view_preprocess_steps=view_preprocess_steps,
        )

        _LOGGER.info("Loading dataset: %s", cfg.dataset.id)
        dataset, dataset_info = ds_orch.load(cfg.dataset)
        ds_orch.verify_integrity(dataset, cfg.dataset)
        source_dataset = dataset
        dataset = sampling_orch.prepare_dataset(dataset, plan_dict=cfg.sampling.plan)
        dataset_has_graph = dataset.has_graph

        _LOGGER.info("Preflight checks")
        if steps_require_fit_indices(preprocess_steps) and cfg.preprocess.fit_on is None:
            raise BenchConfigError(
                "preprocess.fit_on must be set when the plan includes fittable steps",
                code="E_BENCH_FIT_ON_REQUIRED",
            )
        if steps_require_fit_indices(view_preprocess_steps) and cfg.preprocess.fit_on is None:
            raise BenchConfigError(
                "preprocess.fit_on must be set when views include fittable preprocess steps",
                code="E_BENCH_FIT_ON_REQUIRED",
            )

        _preflight(cfg=cfg)

        requires_torch = method_runtime.requires_torch
        resolved_device = method_runtime.resolved_device
        resolution["device"]["resolved"] = resolved_device
        artifacts["method"] = {
            "id": cfg.method.method_id,
            "kind": cfg.method.kind,
            "profile": cfg.method.profile,
            "device": {
                "requested": cfg.method.device.device,
                "resolved": resolved_device,
                "dtype": cfg.method.device.dtype,
            },
        }

        artifacts["dataset"] = {
            "id": cfg.dataset.id,
            "info": dataset_info,
            "fingerprint": dataset.meta.get("dataset_fingerprint"),
            "content_sha256": dataset.meta.get("dataset_content_sha256"),
            "content_manifest_sha256": dataset.meta.get("dataset_content_manifest_sha256"),
        }

        _LOGGER.info("Sampling splits")
        sampling_seed = ctx.seed_for("sampling", cfg.sampling.seed)
        sampling_component_seeds = SamplingPlan.from_dict(
            cfg.sampling.plan
        ).component_seeds.resolve(sampling_seed)
        sampling = sampling_orch.run(
            dataset,
            plan_dict=cfg.sampling.plan,
            seed=sampling_seed,
            dataset_id=cfg.dataset.id,
            resource_root=config_path.parent,
        )
        sampling_replay = _persist_sampling_replay(ctx, sampling)
        use_test = sampling.uses_test_split()
        protocol["use_test_split"] = bool(use_test)
        resolution["splits"]["resolved"] = {
            "report_splits": list(cfg.evaluation.report_splits),
            "use_test_split": bool(use_test),
            "refs": dict(sampling.refs),
        }
        artifacts["sampling"] = {
            "seed": sampling_seed,
            "component_seeds": sampling_component_seeds,
            "plan": sampling.plan,
            "split_fingerprint": sampling.split_fingerprint,
            "stats": sampling.stats,
            "replay": sampling_replay,
        }

        _LOGGER.info("Preprocess")
        fit_indices = resolve_fit_indices(
            dataset=dataset, sampling=sampling, fit_on=cfg.preprocess.fit_on
        )
        preprocess_seed = ctx.seed_for("preprocess", cfg.preprocess.seed)
        pre = prep_orch.run(
            dataset,
            plan_dict=cfg.preprocess.plan,
            seed=preprocess_seed,
            fit_indices=fit_indices,
            cache=cfg.preprocess.cache,
            cache_dir=cfg.preprocess.cache_dir,
        )
        artifacts["preprocess"] = {
            "seed": preprocess_seed,
            "preprocess_fingerprint": pre.preprocess_fingerprint,
            "plan_fingerprint": pre.plan.fingerprint,
            "fit_fingerprint": pre.dataset.meta.get("preprocess_fit_fingerprint"),
            "cache_dir": pre.cache_dir,
        }
        logged_preprocess_artifacts = _preprocess_logged_artifacts(pre)
        if logged_preprocess_artifacts:
            artifacts["preprocess"]["logged_artifacts"] = logged_preprocess_artifacts

        views = None
        if cfg.views is not None:
            _LOGGER.info("Views")
            views_seed = ctx.seed_for("views", cfg.views.seed)
            views = views_orch.run(
                pre.dataset,
                plan_dict=cfg.views.plan,
                seed=views_seed,
                fit_indices=fit_indices,
                cache=cfg.preprocess.cache,
            )
            artifacts["views"] = {
                "seed": views_seed,
                "n_views": len(views.views),
                "meta": views.meta,
            }

        graph = None
        if cfg.graph is not None and cfg.graph.enabled:
            _LOGGER.info("Graph")
            if not cfg.graph.spec:
                raise BenchConfigError(
                    "graph.spec must be provided when graph.enabled=true",
                    code="E_BENCH_GRAPH_SPEC_REQUIRED",
                )
            graph_seed = ctx.seed_for("graph", cfg.graph.seed)
            ds_fp = pre.dataset.meta.get("dataset_fingerprint")
            graph = graph_orch.build(
                pre,
                spec_dict=cfg.graph.spec,
                seed=graph_seed,
                dataset_fingerprint=ds_fp,
                cache=cfg.graph.cache,
                require_cache_hit=cfg.graph.require_cache_hit,
                cache_dir=cfg.graph.cache_dir,
                include_test=use_test,
                expected_fingerprint=cfg.graph.expected_fingerprint,
                expected_preprocess_fingerprint=cfg.graph.expected_preprocess_fingerprint,
                resource_root=config_path.parent,
            )
            graph_info = graph_orch.summarize_graph(graph, cfg.graph.spec)
            artifacts["graph"] = {
                "seed": graph_seed,
                "fingerprint": graph.meta.get("fingerprint"),
                "spec": cfg.graph.spec,
                "info": graph_info,
            }
            resolution["backend"]["resolved"]["graph"] = (
                dependency_resolution.resolved_graph_backend
            )
        elif dataset_has_graph:
            _LOGGER.info("Graph (dataset-provided)")
            if pre.dataset.train.edges is None:
                raise BenchConfigError(
                    "Graph dataset is missing train.edges",
                    code="E_BENCH_GRAPH_MISSING_EDGES",
                )
            n_nodes = int(pre.dataset.train.y.shape[0])
            graph = graph_from_dataset(pre.dataset, n_nodes=n_nodes)
            graph_info = graph_orch.summarize_graph(graph, None)
            artifacts["graph"] = {
                "seed": None,
                "fingerprint": graph.meta.get("fingerprint"),
                "spec": None,
                "source": "dataset",
                "info": graph_info,
            }
            resolution["backend"]["resolved"]["graph"] = "dataset"

        augmentation_result = None
        if (
            cfg.augmentation is not None
            and cfg.augmentation.enabled
            and cfg.method.kind == "inductive"
        ):
            _LOGGER.info("Augmentation")
            aug_seed = ctx.seed_for("augmentation", cfg.augmentation.seed)
            augmentation_result = prepare_unlabeled_augmentation(
                pre.dataset.train.X,
                unlabeled_indices=np.asarray(sampling.unlabeled_idx, dtype=np.int64),
                weak_plan=cfg.augmentation.weak,
                strong_plan=cfg.augmentation.strong,
                seed=aug_seed,
                mode=cfg.augmentation.mode,
                modality=cfg.augmentation.modality,
                strong_views=cfg.augmentation.strong_views,
                online_augmenter_id=cfg.augmentation.online_augmenter_id,
                online_augmenter_params=cfg.augmentation.online_augmenter_params,
            )
            artifacts["augmentation"] = {"seed": aug_seed, "mode": cfg.augmentation.mode}

        routed = route_scientific_input(
            ScientificInputRequest(
                regime=cfg.method.kind,
                preprocess=pre,
                sampling=sampling,
                graph=graph,
                views=views,
                augmentation=augmentation_result,
                augmentation_configured=bool(
                    cfg.augmentation is not None and cfg.augmentation.enabled
                ),
                inductive_graph_policy=cfg.sampling.inductive_graph_policy,
                use_test_split=use_test,
            )
        )
        execution_sampling = routed.sampling
        artifacts["input_routing"] = routed.to_dict()
        artifacts["method"]["runtime_resolution"] = method_runtime.to_dict()

        prepared_artifacts = {
            "routed_input": routed,
            "use_test": use_test,
            "strict": bool(cfg.run.benchmark_mode),
            "requires_torch": bool(requires_torch),
        }

        if cfg.search is not None and cfg.search.enabled:
            _LOGGER.info("HPO: %s search", cfg.search.kind)
            best_patch, hpo_summary = hpo_orch.run_hpo(
                ctx=ctx,
                base_cfg=cfg,
                base_cfg_dict=raw,
                prepared_artifacts=prepared_artifacts,
                method_runtime_resolver=lambda trial_cfg: _resolve_method_runtime(
                    trial_cfg,
                    preprocess_steps=preprocess_steps,
                ),
            )
            if best_patch is None:
                if hpo_summary.get("status") == "not_evaluable":
                    raise BenchRuntimeError(
                        "E_BENCH_HPO_NOT_EVALUABLE",
                        _hpo_not_evaluable_message(hpo_summary),
                    )
                raise BenchRuntimeError(
                    "E_BENCH_HPO_TRIALS_FAILED",
                    "HPO produced no successful trial",
                )
            patched_raw = deep_merge(raw, best_patch)
            patched_raw, hpo_limit_changes, _ = apply_limits(
                patched_raw,
                limits=cfg.limits,
                strict=cfg.run.benchmark_mode,
            )
            if hpo_limit_changes:
                _LOGGER.info("Applied memory limits after HPO: changes=%s", len(hpo_limit_changes))
                _LOGGER.debug("Limit adjustments: %s", hpo_limit_changes)
            raw = patched_raw
            cfg = ExperimentConfig.from_dict(raw, allow_resolved_acceptance_seed=True)
            method_runtime = _resolve_method_runtime(
                cfg,
                preprocess_steps=preprocess_steps,
            )
            required_distributions, dependency_resolution = _required_software_distributions(
                cfg=cfg,
                preprocess_plan=preprocess_plan,
                views_plan=views_plan,
                method=method_runtime,
            )
            candidate_versions = _collect_code_runtime_versions(
                required_distributions=required_distributions,
                require_complete_manifest=cfg.run.resume_policy != "never",
            )
            requires_torch = method_runtime.requires_torch
            resolution["device"]["resolved"] = method_runtime.resolved_device
            if dependency_resolution.resolved_graph_backend is not None:
                resolution["backend"]["resolved"]["graph"] = (
                    dependency_resolution.resolved_graph_backend
                )
            artifacts["method"]["runtime_resolution"] = method_runtime.to_dict()
            artifacts["method"]["device"]["resolved"] = method_runtime.resolved_device
            candidate_effective_config_hash = hash_any(raw)
            candidate_resume_identity = build_resume_identity(
                raw,
                seed=int(cfg.run.seed),
                runtime_versions=candidate_versions,
            )
            effective_config_hash, resume_identity, versions = (
                candidate_effective_config_hash,
                candidate_resume_identity,
                candidate_versions,
            )
            report_config = raw
            report_versions = versions
            patched_run_id = resume_identity.short_id
            _sync_ctx_run_identity(ctx, run_id=patched_run_id)
            ctx.write_config_copy(raw)
            resolution["backend"]["requested"]["method"] = cfg.method.params.get("backend")

        execution_context = ExecutionContext(
            identity=resume_identity,
            output_dir=ctx.output_dir,
            resume_policy=cfg.run.resume_policy,
            checkpoint_root=(
                None
                if cfg.run.checkpoint_dir is None
                else Path(cfg.run.checkpoint_dir).expanduser().resolve()
            ),
        )
        final_routed = route_scientific_input(
            ScientificInputRequest(
                regime=cfg.method.kind,
                preprocess=pre,
                sampling=sampling,
                graph=graph,
                views=views,
                augmentation=augmentation_result,
                augmentation_configured=bool(
                    cfg.augmentation is not None and cfg.augmentation.enabled
                ),
                inductive_graph_policy=cfg.sampling.inductive_graph_policy,
                use_test_split=use_test,
                execution_context=execution_context,
            )
        )
        execution_sampling = final_routed.sampling

        if cfg.method.kind == "inductive":
            _LOGGER.info("Method: %s", cfg.method.method_id)
            method_seed = _resolve_method_seed(ctx, cfg)
            method, method_resolution = inductive_orch.run(
                final_routed.execution_input,
                during_fit_splits=cfg.evaluation.during_fit_splits,
                cfg=cfg.method,
                seed=method_seed,
                strict=cfg.run.benchmark_mode,
                requires_torch=requires_torch,
            )
            persist_execution_contract_from_resolution(
                method_resolution,
                artifacts=artifacts,
                resolution=resolution,
            )
            metrics = eval_orch.evaluate_inductive(
                method=method,
                pre=pre,
                sampling=execution_sampling,
                report_splits=cfg.evaluation.report_splits,
                metrics=cfg.evaluation.metrics,
                views=views,
                strict=cfg.run.benchmark_mode,
            )
        else:
            _LOGGER.info("Method: %s", cfg.method.method_id)
            method_seed = _resolve_method_seed(ctx, cfg)
            method, data, method_resolution = transductive_orch.run(
                final_routed.execution_input,
                cfg=cfg.method,
                seed=method_seed,
                use_test_split=use_test,
                expected_labeled_count=final_routed.expected_labeled_count,
                strict=cfg.run.benchmark_mode,
            )
            persist_execution_contract_from_resolution(
                method_resolution,
                artifacts=artifacts,
                resolution=resolution,
            )
            metrics = eval_orch.evaluate_transductive(
                method=method,
                data=data,
                report_splits=cfg.evaluation.report_splits,
                metrics=cfg.evaluation.metrics,
                masks=final_routed.masks,
            )

        evaluation_outcome = assess_evaluation_metrics(metrics)
        metrics = evaluation_outcome.metrics
        if evaluation_outcome.status == "not_evaluable":
            status = evaluation_outcome.status
            error_code = evaluation_outcome.code
            error = f"{evaluation_outcome.reason}: " + ", ".join(
                evaluation_outcome.non_finite_paths
            )

        resolution["backend"]["resolved"]["method"] = method_resolution.get("backend")
        resolution["backend"]["resolved"]["classifier"] = method_resolution.get(
            "classifier_backend"
        )
        resolution["dtype"]["resolved"] = method_resolution.get("dtypes", {})
        resolution["normalization"]["resolved"] = method_resolution.get("normalization", {})
        method_diagnostics = method_resolution.get("diagnostics")
        if isinstance(method_diagnostics, Mapping):
            artifacts["method"]["diagnostics"] = dict(method_diagnostics)
        pipeline_capabilities = method_resolution.get("pipeline_capabilities")
        if isinstance(pipeline_capabilities, Mapping):
            artifacts["method"]["pipeline_capabilities"] = dict(pipeline_capabilities)

        if verified_input_artifacts is not None:
            verified_input_artifacts = input_artifact_orch.revalidate(verified_input_artifacts)
            artifacts["input_artifacts"] = verified_input_artifacts.report_payload()

        dataset_revalidation = ds_orch.revalidate_integrity(source_dataset, cfg.dataset)
        if dataset_revalidation is not None:
            artifacts["dataset"]["revalidated_before_result"] = True
            artifacts["dataset"]["final_content_state_sha256"] = dataset_revalidation[
                "cache_state_sha256"
            ]

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        error_code = extract_error_code(exc)
        if persist_execution_contract_from_error(
            exc,
            artifacts=artifacts,
            resolution=resolution,
        ):
            error_code = EXECUTION_CONTRACT_ERROR_CODE
        exception_diagnostics = getattr(exc, "diagnostics", None)
        if isinstance(exception_diagnostics, Mapping):
            method_artifacts = artifacts.get("method")
            if not isinstance(method_artifacts, dict):
                method_artifacts = {}
                artifacts["method"] = method_artifacts
            method_artifacts["diagnostics"] = dict(exception_diagnostics)
        status = (
            "not_evaluable"
            if error_code
            in {
                "E_BENCH_HPO_NOT_EVALUABLE",
                "E_EVALUATION_NOT_EVALUABLE",
                "E_METHOD_NOT_EVALUABLE",
            }
            else "failed"
        )
        _write_error_traceback(ctx, traceback.format_exc())
        _LOGGER.exception("Run failed")
        report_orch.write_run_summary(
            ctx=ctx,
            cfg=cfg,
            artifacts=artifacts,
            metrics=metrics,
            hpo=hpo_summary,
            status=status,
            hashes={
                "config_hash": config_hash,
                "effective_config_hash": effective_config_hash,
                "protocol_sha256": resume_identity.config_sha256,
                "software_sha256": resume_identity.code_sha256,
                "execution_identity_sha256": resume_identity.sha256,
            },
            execution_identity=resume_identity.to_dict(),
            resolution=resolution,
            protocol=protocol,
            versions=report_versions,
            effective_config=report_config,
            fallback_events=fallback_events,
            resource_measurement=resource_measurement,
            error=error,
            error_code=error_code,
        )
        if cfg.run.benchmark_mode or ctx.fail_fast:
            raise
        return SingleRunResult(code=1, run_dir=ctx.run_dir, run_json_path=ctx.run_dir / "run.json")

    if metrics is not None:
        _LOGGER.info("Metrics: %s", metrics)
    _LOGGER.info("Run finished: %s", status)
    report_orch.write_run_summary(
        ctx=ctx,
        cfg=cfg,
        artifacts=artifacts,
        metrics=metrics,
        hpo=hpo_summary,
        status=status,
        hashes={
            "config_hash": config_hash,
            "effective_config_hash": effective_config_hash,
            "protocol_sha256": resume_identity.config_sha256,
            "software_sha256": resume_identity.code_sha256,
            "execution_identity_sha256": resume_identity.sha256,
        },
        execution_identity=resume_identity.to_dict(),
        resolution=resolution,
        protocol=protocol,
        versions=report_versions,
        effective_config=report_config,
        fallback_events=fallback_events,
        resource_measurement=resource_measurement,
        error=error,
        error_code=error_code,
    )
    return SingleRunResult(
        code=0 if status == "success" else 1,
        run_dir=ctx.run_dir,
        run_json_path=ctx.run_dir / "run.json",
    )


def run_experiment_single(
    config_path: Path,
    *,
    raw: dict[str, Any] | None = None,
    cfg: ExperimentConfig | None = None,
) -> SingleRunResult:
    """Execute one resolved configuration and one seed.

    This is the stable primitive used by the CLI as well as scheduler wrappers.
    It deliberately has no knowledge of scheduler policies, articles, or compute sites.
    """

    if raw is None:
        raw = load_yaml(config_path)
    if cfg is None:
        cfg = ExperimentConfig.from_dict(raw, allow_resolved_acceptance_seed=True)
    if cfg.run.seeds is not None:
        raise ValueError("run_experiment_single requires run.seeds to be absent")
    return _run_experiment_single(config_path, raw=raw, cfg=cfg)


def _expected_report_hashes(
    raw: dict[str, Any],
    *,
    cfg: ExperimentConfig,
) -> tuple[str, str]:
    effective_raw, _changes, _resolved = apply_limits(
        raw,
        limits=cfg.limits,
        strict=cfg.run.benchmark_mode,
    )
    return hash_any(raw), protocol_sha256(effective_raw)


def _write_seed_sweep_aggregate(
    *,
    sweep_root: Path,
    config_path: Path,
    base_name: str,
    requested_seeds: list[int],
    run_results: list[SingleRunResult],
    expected_config_hashes: Mapping[int, str],
    expected_protocol_hashes: Mapping[int, str] | None,
    acceptance: AcceptanceSpec | None = None,
) -> Path:
    run_json_paths = {
        result.run_json_path for result in run_results if result.run_json_path.is_file()
    }
    run_json_paths.update(path for path in sweep_root.rglob("run.json") if path.is_file())
    return report_orch.write_seed_sweep_summary(
        output_dir=sweep_root,
        config_path=config_path,
        base_name=base_name,
        requested_seeds=requested_seeds,
        run_json_paths=sorted(run_json_paths, key=str),
        expected_config_hashes=expected_config_hashes,
        expected_protocol_hashes=expected_protocol_hashes,
        acceptance=acceptance,
    )


def _validate_run_selection(
    *,
    num_runs: int | None,
    seed: int | None,
    seed_index: int | None,
) -> None:
    """Validate mutually exclusive CLI/API controls before reading a card."""

    if num_runs is not None and num_runs <= 0:
        raise ValueError("num_runs must be > 0")
    if seed is not None and seed < 0:
        raise ValueError("seed must be >= 0")
    if seed_index is not None and seed_index < 0:
        raise ValueError("seed_index must be >= 0")
    selected_controls = sum(value is not None for value in (seed, seed_index, num_runs))
    if selected_controls > 1:
        raise ValueError("seed, seed_index and num_runs are mutually exclusive")


def _run_experiment_body(
    config_path: Path,
    *,
    raw: dict[str, Any],
    cfg: ExperimentConfig,
    num_runs: int | None = None,
    seed: int | None = None,
    seed_index: int | None = None,
) -> int:
    if seed_index is not None:
        if not cfg.run.seeds:
            raise ValueError("seed_index requires a non-empty run.seeds list in the YAML")
        if seed_index >= len(cfg.run.seeds):
            raise ValueError(
                f"seed_index {seed_index} is outside run.seeds (size={len(cfg.run.seeds)})"
            )
        seed = int(cfg.run.seeds[seed_index])

    if seed is not None:
        run_name = sweep_run_name(cfg.run.name, seed=seed, index=0, total=1)
        single_raw = apply_global_seed(
            raw,
            seed=seed,
            run_name=run_name,
            seeded_sections=cfg.run.seeded_sections,
        )
        single_cfg = ExperimentConfig.from_dict(
            single_raw,
            allow_resolved_acceptance_seed=True,
        )
        _LOGGER.info("Single-seed run: name=%s seed=%s", run_name, seed)
        return _run_experiment_single(config_path, raw=single_raw, cfg=single_cfg).code

    if num_runs is not None:
        seeds = [int(cfg.run.seed) + i for i in range(num_runs)]
        _LOGGER.info(
            "Run-count sweep start: name=%s num_runs=%s base_seed=%s seeds=%s",
            cfg.run.name,
            num_runs,
            cfg.run.seed,
            seeds,
        )
    elif cfg.run.seeds:
        seeds = [int(s) for s in cfg.run.seeds]
        _LOGGER.info("Seed sweep start: name=%s seeds=%s", cfg.run.name, seeds)
    else:
        return _run_experiment_single(config_path, raw=raw, cfg=cfg).code

    sweep_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sweep_root = next_available_run_dir(
        Path(cfg.run.output_dir).expanduser().resolve() / f"{cfg.run.name}-sweep-{sweep_timestamp}"
    )
    _LOGGER.info("Seed sweep output dir: %s", sweep_root)

    seed_jobs: list[tuple[int, int, str, dict[str, Any], ExperimentConfig]] = []
    expected_config_hashes: dict[int, str] = {}
    expected_protocol_hashes: dict[int, str] | None = (
        None if cfg.search is not None and cfg.search.enabled else {}
    )
    for i, seed in enumerate(seeds, start=1):
        run_name = sweep_run_name(cfg.run.name, seed=seed, index=i - 1, total=len(seeds))
        sweep_raw = apply_global_seed(
            raw,
            seed=seed,
            run_name=run_name,
            seeded_sections=cfg.run.seeded_sections,
        )
        sweep_run = sweep_raw.get("run")
        if not isinstance(sweep_run, dict):
            sweep_run = {}
            sweep_raw["run"] = sweep_run
        sweep_run["output_dir"] = str(sweep_root)
        sweep_cfg = ExperimentConfig.from_dict(
            sweep_raw,
            allow_resolved_acceptance_seed=True,
        )
        config_hash, protocol_hash = _expected_report_hashes(sweep_raw, cfg=sweep_cfg)
        expected_config_hashes[seed] = config_hash
        if expected_protocol_hashes is not None:
            expected_protocol_hashes[seed] = protocol_hash
        seed_jobs.append((i, seed, run_name, sweep_raw, sweep_cfg))

    failures = 0
    run_results: list[SingleRunResult] = []
    for i, seed, run_name, sweep_raw, sweep_cfg in seed_jobs:
        _LOGGER.info("Seed sweep run %s/%s: seed=%s name=%s", i, len(seeds), seed, run_name)
        try:
            result = _run_experiment_single(config_path, raw=sweep_raw, cfg=sweep_cfg)
        except Exception:
            if sweep_cfg.run.benchmark_mode or sweep_cfg.run.fail_fast:
                _write_seed_sweep_aggregate(
                    sweep_root=sweep_root,
                    config_path=config_path,
                    base_name=cfg.run.name,
                    requested_seeds=seeds,
                    run_results=run_results,
                    expected_config_hashes=expected_config_hashes,
                    expected_protocol_hashes=expected_protocol_hashes,
                    acceptance=cfg.acceptance,
                )
                raise
            failures += 1
            continue
        run_results.append(result)
        if result.code != 0:
            failures += 1
            if sweep_cfg.run.benchmark_mode or sweep_cfg.run.fail_fast:
                _write_seed_sweep_aggregate(
                    sweep_root=sweep_root,
                    config_path=config_path,
                    base_name=cfg.run.name,
                    requested_seeds=seeds,
                    run_results=run_results,
                    expected_config_hashes=expected_config_hashes,
                    expected_protocol_hashes=expected_protocol_hashes,
                    acceptance=cfg.acceptance,
                )
                raise BenchRuntimeError(
                    "E_BENCH_SWEEP_FAILED",
                    "seed sweep aborted due to failed run with fail_fast=true",
                )

    aggregate_path = _write_seed_sweep_aggregate(
        sweep_root=sweep_root,
        config_path=config_path,
        base_name=cfg.run.name,
        requested_seeds=seeds,
        run_results=run_results,
        expected_config_hashes=expected_config_hashes,
        expected_protocol_hashes=expected_protocol_hashes,
        acceptance=cfg.acceptance,
    )

    if failures:
        _LOGGER.warning("Seed sweep finished with failures: %s/%s", failures, len(seeds))
        return 1
    if cfg.acceptance is not None:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        acceptance_status = aggregate["acceptance"]["assessment_status"]
        if acceptance_status != "passed":
            _LOGGER.warning("Seed sweep acceptance did not pass: status=%s", acceptance_status)
            return 1
    _LOGGER.info("Seed sweep finished successfully: %s runs", len(seeds))
    return 0


def run_experiment(
    config_path: Path,
    *,
    num_runs: int | None = None,
    seed: int | None = None,
    seed_index: int | None = None,
) -> int:
    """Run a card and expose checkpointed continuation as ``EX_TEMPFAIL`` (75)."""

    _validate_run_selection(num_runs=num_runs, seed=seed, seed_index=seed_index)
    raw = load_yaml(config_path)
    cfg = ExperimentConfig.from_dict(raw)
    continuation_enabled = cfg.run.resume_policy != "never"
    with continuation_signal_handler(enabled=continuation_enabled):
        if not continuation_enabled:
            return _run_experiment_body(
                config_path,
                raw=raw,
                cfg=cfg,
                num_runs=num_runs,
                seed=seed,
                seed_index=seed_index,
            )
        try:
            return _run_experiment_body(
                config_path,
                raw=raw,
                cfg=cfg,
                num_runs=num_runs,
                seed=seed,
                seed_index=seed_index,
            )
        except PlannedContinuation as exc:
            _LOGGER.info(
                "Durable checkpoint committed after signal %s; exiting with retryable status "
                "%s (EX_TEMPFAIL)",
                exc.signum,
                PLANNED_CONTINUATION_EXIT_CODE,
            )
            return PLANNED_CONTINUATION_EXIT_CODE


def reconcile_seed_runs(
    config_path: Path,
    *,
    runs_root: Path,
    output_dir: Path | None = None,
    require_execution_identity: bool = True,
) -> int:
    """Reconcile independently produced ``run.json`` reports for one YAML card."""

    raw = load_yaml(config_path)
    cfg = ExperimentConfig.from_dict(raw)
    if not cfg.run.seeds:
        raise ValueError("reconciliation requires a non-empty run.seeds list in the YAML")

    expected_config_hashes: dict[int, str] = {}
    expected_protocol_hashes: dict[int, str] | None = (
        None if cfg.search is not None and cfg.search.enabled else {}
    )
    for seed in cfg.run.seeds:
        seeded_raw = apply_global_seed(
            raw,
            seed=int(seed),
            run_name=sweep_run_name(
                cfg.run.name,
                seed=int(seed),
                index=0,
                total=1,
            ),
            seeded_sections=cfg.run.seeded_sections,
        )
        seeded_cfg = ExperimentConfig.from_dict(
            seeded_raw,
            allow_resolved_acceptance_seed=True,
        )
        config_hash, protocol_hash = _expected_report_hashes(seeded_raw, cfg=seeded_cfg)
        expected_config_hashes[int(seed)] = config_hash
        if expected_protocol_hashes is not None:
            expected_protocol_hashes[int(seed)] = protocol_hash

    resolved_root = runs_root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"runs_root must be an existing directory: {resolved_root}")
    run_json_paths = sorted(
        (path for path in resolved_root.rglob("run.json") if path.is_file()),
        key=lambda path: str(path.relative_to(resolved_root)),
    )
    resolved_output = resolved_root if output_dir is None else output_dir.expanduser().resolve()
    aggregate_path = report_orch.write_seed_sweep_summary(
        output_dir=resolved_output,
        config_path=config_path,
        base_name=cfg.run.name,
        requested_seeds=[int(seed) for seed in cfg.run.seeds],
        run_json_paths=run_json_paths,
        expected_config_hashes=expected_config_hashes,
        expected_protocol_hashes=expected_protocol_hashes,
        acceptance=cfg.acceptance,
        require_execution_identity=require_execution_identity,
    )
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    sweep_passed = payload["sweep"]["certifiable"] is True
    acceptance_passed = (
        cfg.acceptance is None or payload["acceptance"]["assessment_status"] == "passed"
    )
    return 0 if sweep_passed and acceptance_passed else 1


def _main_reconcile(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="modssc-bench reconcile",
        description="Reconcile independently produced seed run reports",
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--runs-root",
        required=True,
        help="Root searched recursively for run.json reports",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for aggregate.json (defaults to --runs-root)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level: none, basic, detailed (aliases: quiet, full).",
    )
    parser.add_argument(
        "--allow-legacy-run-identity",
        action="store_true",
        help=(
            "Explicitly allow reports written before portable execution identity; "
            "never use for new scientific campaigns"
        ),
    )
    args = parser.parse_args(argv)
    try:
        resolved = _resolve_log_level_for_run(Path(args.config), args.log_level)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    configure_logging(resolved)
    return reconcile_seed_runs(
        Path(args.config),
        runs_root=Path(args.runs_root),
        output_dir=None if args.output_dir is None else Path(args.output_dir),
        require_execution_identity=not args.allow_legacy_run_identity,
    )


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "reconcile":
        return _main_reconcile(sys.argv[2:])

    parser = argparse.ArgumentParser(
        description="Run ModSSC benchmark orchestration",
        epilog=(
            "Exit status 75 (EX_TEMPFAIL) means a resumable execution committed a durable "
            "checkpoint and requests a later continuation."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level: none, basic, detailed (aliases: quiet, full).",
    )
    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument(
        "--seed",
        type=_non_negative_int,
        default=None,
        help="Run exactly one seed, overriding run.seed and run.seeds from YAML.",
    )
    sweep_group.add_argument(
        "--seed-index",
        type=_non_negative_int,
        default=None,
        help=(
            "Run one entry from run.seeds by zero-based index. This is suitable for "
            "scheduler arrays and keeps the seed list in the YAML as the source of truth."
        ),
    )
    sweep_group.add_argument(
        "--num-runs",
        type=_positive_int,
        default=None,
        help=(
            "Run count sweep from run.seed (equivalent to run.seeds=[seed, seed+1, ...]). "
            "Overrides run.seeds when provided."
        ),
    )
    args = parser.parse_args()
    try:
        resolved = _resolve_log_level_for_run(Path(args.config), args.log_level)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    configure_logging(resolved)
    return run_experiment(
        Path(args.config),
        num_runs=args.num_runs,
        seed=args.seed,
        seed_index=args.seed_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
