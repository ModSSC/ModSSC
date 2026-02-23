from __future__ import annotations

import argparse
import logging
import traceback
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from modssc.device import resolve_device_name
from modssc.evaluation import list_metrics
from modssc.hpo import deep_merge
from modssc.inductive.registry import get_method_class as get_inductive_method_class
from modssc.inductive.registry import get_method_info as get_inductive_method_info
from modssc.logging import configure_logging, resolve_log_level
from modssc.preprocess import step_info
from modssc.sampling.result import SamplingResult
from modssc.transductive.registry import get_method_class as get_transductive_method_class
from modssc.transductive.registry import get_method_info as get_transductive_method_info

from .context import RunContext
from .errors import BenchRuntimeError, extract_error_code
from .limits import apply_limits
from .orchestrators import augmentation as aug_orch
from .orchestrators import dataset as ds_orch
from .orchestrators import evaluation as eval_orch
from .orchestrators import graph as graph_orch
from .orchestrators import hpo as hpo_orch
from .orchestrators import method_inductive as inductive_orch
from .orchestrators import method_transductive as transductive_orch
from .orchestrators import preprocess as prep_orch
from .orchestrators import reporting as report_orch
from .orchestrators import sampling as sampling_orch
from .orchestrators import views as views_orch
from .orchestrators.slicing import select_rows
from .schema import BenchConfigError, ExperimentConfig
from .seed_sweep import apply_global_seed, sweep_run_name
from .utils.hashing import hash_any
from .utils.import_tools import check_extra_installed
from .utils.io import load_yaml
from .utils.runtime import collect_runtime_versions

_ALLOWED_METRICS = set(list_metrics())
_ALLOWED_SPLITS = {"train", "val", "test"}
_LOGGER = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if out <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return out


def _dataset_has_graph(dataset: Any) -> bool:
    return (
        getattr(dataset.train, "edges", None) is not None
        or getattr(dataset.train, "masks", None) is not None
    )


def _graph_sampling_to_inductive(sampling: SamplingResult) -> SamplingResult:
    masks = sampling.masks
    indices = {
        "train": np.where(np.asarray(masks["train"], dtype=bool))[0].astype(np.int64),
        "val": np.where(np.asarray(masks["val"], dtype=bool))[0].astype(np.int64),
        "test": np.where(np.asarray(masks["test"], dtype=bool))[0].astype(np.int64),
        "train_labeled": np.where(np.asarray(masks["labeled"], dtype=bool))[0].astype(np.int64),
        "train_unlabeled": np.where(np.asarray(masks["unlabeled"], dtype=bool))[0].astype(np.int64),
    }
    refs = {k: "train" for k in indices}
    return SamplingResult(
        schema_version=sampling.schema_version,
        created_at=sampling.created_at,
        dataset_fingerprint=sampling.dataset_fingerprint,
        split_fingerprint=sampling.split_fingerprint,
        plan=sampling.plan,
        indices=indices,
        refs=refs,
        masks={},
        stats=dict(sampling.stats),
    )


def _check_extra(extra: str) -> None:
    missing = check_extra_installed(extra)
    if missing:
        raise BenchConfigError(
            f"Missing optional dependency for extra '{extra}': {sorted(set(missing))}",
            code="E_BENCH_DEPENDENCY_MISSING",
        )


def _method_requires_torch(method_id: str, *, strict: bool) -> bool:
    cls = get_inductive_method_class(method_id)
    try:
        inst = cls()
    except (TypeError, ValueError, RuntimeError, ImportError, ModuleNotFoundError) as exc:
        if strict:
            raise BenchConfigError(
                f"failed to introspect inductive method '{method_id}' for benchmark preflight: {exc}",
                code="E_BENCH_METHOD_INTROSPECTION",
            ) from exc
        return False
    spec = getattr(inst, "spec", None)
    bundle_fields = (
        "model_bundle",
        "teacher_bundle",
        "student_bundle",
        "model_bundle_1",
        "model_bundle_2",
        "pretrain_bundle",
        "finetune_bundle",
        "shared_bundle",
        "head_bundles",
    )
    return any(hasattr(spec, field) for field in bundle_fields)


def _resolve_method_device(requested: str, *, supports_gpu: bool, strict: bool) -> str:
    if strict and requested == "auto":
        raise BenchConfigError(
            "method.device.device='auto' is forbidden when run.benchmark_mode=true",
            code="E_BENCH_AUTO_FORBIDDEN",
        )
    if requested != "auto":
        return requested
    if not supports_gpu:
        return "cpu"
    return resolve_device_name(requested)


def _record_resolved_device(
    raw: dict[str, Any],
    *,
    cfg: ExperimentConfig,
    resolved_device: str,
) -> None:
    method = raw.get("method")
    if not isinstance(method, dict):
        method = {}
        raw["method"] = method
    device = method.get("device")
    if not isinstance(device, dict):
        device = {}
        method["device"] = device
    device.setdefault("device", cfg.method.device.device)
    device.setdefault("dtype", cfg.method.device.dtype)
    device["resolved_device"] = resolved_device


def _preflight(
    *,
    cfg: ExperimentConfig,
    dataset_info: dict[str, Any],
    preprocess_steps: list[str],
    dataset_has_graph: bool,
) -> Any:
    extra = dataset_info.get("required_extra")
    if extra:
        _check_extra(str(extra))

    for step_id in preprocess_steps:
        info = step_info(step_id)
        extra = info.get("required_extra")
        if extra:
            _check_extra(str(extra))

    if cfg.method.kind == "inductive":
        method_info = get_inductive_method_info(cfg.method.method_id)
    else:
        method_info = get_transductive_method_info(cfg.method.method_id)

    if method_info.required_extra:
        _check_extra(str(method_info.required_extra))

    if (
        cfg.method.kind == "inductive"
        and cfg.method.method_id == "co_training"
        and cfg.views is None
    ):
        raise BenchConfigError("co_training requires views to be configured", code="E_BENCH_CONFIG")

    if cfg.method.kind == "transductive":
        graph_enabled = bool(cfg.graph and cfg.graph.enabled)
        if not dataset_has_graph and not graph_enabled:
            raise BenchConfigError(
                "Transductive methods require a graph; set graph.enabled=true or use a graph dataset",
                code="E_BENCH_GRAPH_REQUIRED",
            )

    if cfg.augmentation is not None and cfg.augmentation.enabled:
        if cfg.augmentation.mode != "fixed":
            raise BenchConfigError(
                "Only augmentation.mode='fixed' is supported",
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
    return method_info


def _preprocess_step_ids(plan_dict: dict[str, Any]) -> list[str]:
    steps_raw = plan_dict.get("steps", [])
    if not isinstance(steps_raw, list):
        raise BenchConfigError("preprocess.plan.steps must be a list", code="E_BENCH_CONFIG")
    ids: list[str] = []
    for item in steps_raw:
        if not isinstance(item, dict):
            raise BenchConfigError("Each preprocess step must be a mapping", code="E_BENCH_CONFIG")
        step_id = str(item.get("id") or item.get("step_id") or "")
        if not step_id:
            raise BenchConfigError(
                "Each preprocess step must define 'id'",
                code="E_BENCH_CONFIG",
            )
        ids.append(step_id)
    return ids


def _views_preprocess_step_ids(views_plan: dict[str, Any]) -> list[str]:
    steps: list[str] = []
    views = views_plan.get("views", [])
    if not isinstance(views, list):
        return steps
    for view in views:
        if not isinstance(view, dict):
            continue
        pre = view.get("preprocess")
        if isinstance(pre, dict):
            steps.extend(_preprocess_step_ids(pre))
    return steps


def _requires_fit_indices(step_ids: list[str]) -> bool:
    for step_id in step_ids:
        info = step_info(step_id)
        if info.get("kind") == "fittable":
            return True
    return False


def _use_test_split(sampling: Any) -> bool:
    return sampling.refs.get("test", "train") == "test"


def _mask_map_from_sampling(
    sampling: Any, n_train: int, n_test: int | None
) -> dict[str, np.ndarray]:
    if sampling.is_graph():
        m = sampling.masks
        return {
            "train_mask": np.asarray(m["train"], dtype=bool),
            "val_mask": np.asarray(m["val"], dtype=bool),
            "test_mask": np.asarray(m["test"], dtype=bool),
            "unlabeled_mask": np.asarray(m["unlabeled"], dtype=bool),
            "labeled_mask": np.asarray(m["labeled"], dtype=bool),
        }
    return transductive_orch._build_masks_from_indices(
        n_train=n_train,
        n_test=n_test,
        indices=sampling.indices,
        refs=sampling.refs,
    )


def _expected_labeled_count(stats: Mapping[str, Any] | None) -> int | None:
    if not stats:
        return None
    train_labeled = stats.get("train_labeled")
    if isinstance(train_labeled, Mapping):
        n = train_labeled.get("n")
        if isinstance(n, (int, np.integer)):
            return int(n)
    labeled = stats.get("labeled")
    if isinstance(labeled, (int, np.integer)):
        return int(labeled)
    labeled_dist = stats.get("labeled_class_dist")
    if isinstance(labeled_dist, Mapping):
        n = labeled_dist.get("n")
        if isinstance(n, (int, np.integer)):
            return int(n)
    return None


def _write_error_traceback(ctx: RunContext, tb: str) -> None:
    try:
        (ctx.run_dir / "error.txt").write_text(tb, encoding="utf-8")
    except OSError:
        _LOGGER.exception("Failed to write error.txt")


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


def _collect_requested_normalization(plan_dict: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    steps = plan_dict.get("steps", [])
    if not isinstance(steps, list):
        return out
    for item in steps:
        if not isinstance(item, Mapping):
            continue
        step_id = str(item.get("id") or item.get("step_id") or "")
        if not step_id:
            continue
        if "normalize" in step_id or step_id in {
            "core.cast_dtype",
            "core.cast_fp16",
            "core.to_torch",
            "core.to_numpy",
        }:
            out.append(step_id)
    return out


def _method_has_backend_param(kind: str, method_id: str, *, strict: bool) -> bool:
    if kind == "inductive":
        cls = get_inductive_method_class(method_id)
    else:
        cls = get_transductive_method_class(method_id)
    try:
        inst = cls()
    except (TypeError, ValueError, RuntimeError, ImportError, ModuleNotFoundError) as exc:
        if strict:
            raise BenchConfigError(
                f"failed to introspect method '{method_id}' backend policy: {exc}",
                code="E_BENCH_METHOD_INTROSPECTION",
            ) from exc
        return False
    spec = getattr(inst, "spec", None)
    return bool(hasattr(spec, "backend"))


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

    if cfg.evaluation.split_for_model_selection is None:
        raise BenchConfigError(
            "evaluation.split_for_model_selection must be set when run.benchmark_mode=true",
            code="E_BENCH_SPLIT_MODEL_SELECTION_REQUIRED",
        )
    if cfg.evaluation.split_for_model_selection not in cfg.evaluation.report_splits:
        raise BenchConfigError(
            "evaluation.split_for_model_selection must be included in evaluation.report_splits",
            code="E_BENCH_SPLIT_MODEL_SELECTION_INVALID",
        )
    if (
        cfg.search is not None
        and cfg.search.enabled
        and cfg.search.objective.split != cfg.evaluation.split_for_model_selection
    ):
        raise BenchConfigError(
            "search.objective.split must match evaluation.split_for_model_selection in benchmark_mode",
            code="E_BENCH_SPLIT_MODEL_SELECTION_CONFLICT",
        )

    if _method_has_backend_param(cfg.method.kind, cfg.method.method_id, strict=True):
        backend = cfg.method.params.get("backend")
        if backend is None:
            raise BenchConfigError(
                f"method.params.backend must be explicitly set for method '{cfg.method.method_id}' in benchmark_mode",
                code="E_BENCH_BACKEND_REQUIRED",
            )
        if str(backend).lower() == "auto":
            raise BenchConfigError(
                "method.params.backend='auto' is forbidden when run.benchmark_mode=true",
                code="E_BENCH_AUTO_FORBIDDEN",
            )

    auto_paths = _scan_auto_entries(raw, path="")
    if auto_paths:
        raise BenchConfigError(
            "auto is forbidden in benchmark_mode at: " + ", ".join(sorted(set(auto_paths))),
            code="E_BENCH_AUTO_FORBIDDEN",
        )

    if _requires_fit_indices(preprocess_steps) and cfg.preprocess.fit_on is None:
        raise BenchConfigError(
            "preprocess.fit_on must be set when preprocess includes fittable steps",
            code="E_BENCH_FIT_ON_REQUIRED",
        )
    if _requires_fit_indices(view_preprocess_steps) and cfg.preprocess.fit_on is None:
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


def _derive_run_id(*, effective_config_hash: str, seed: int, versions: Mapping[str, Any]) -> str:
    return hash_any(
        {
            "effective_config_hash": effective_config_hash,
            "seed": int(seed),
            "versions": dict(versions),
            "git_sha": versions.get("git_sha"),
        }
    )[:20]


def _sync_ctx_run_identity(ctx: RunContext, *, run_id: str) -> None:
    if ctx.run_id == run_id:
        return

    old_run_id = ctx.run_id
    old_run_dir = ctx.run_dir
    timestamp = old_run_dir.name.rsplit("-", 1)[-1]
    new_run_dir = ctx.output_dir / f"{ctx.name}-{run_id}-{timestamp}"
    if new_run_dir.exists():
        raise BenchRuntimeError(
            "E_BENCH_RUN_DIR_COLLISION",
            f"cannot update run directory for recalculated run_id; target already exists: {new_run_dir}",
        )

    old_run_dir.rename(new_run_dir)
    ctx.run_id = str(run_id)
    ctx.run_dir = new_run_dir
    _LOGGER.info("Run identity updated after config mutation: %s -> %s", old_run_id, run_id)


def _run_experiment_single(config_path: Path, *, raw: dict[str, Any], cfg: ExperimentConfig) -> int:
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
        cfg = ExperimentConfig.from_dict(raw)

    effective_config_hash = hash_any(raw)
    versions = collect_runtime_versions(repo_root=config_path.parent)
    run_id = _derive_run_id(
        effective_config_hash=effective_config_hash,
        seed=int(cfg.run.seed),
        versions=versions,
    )

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

    preprocess_steps = _preprocess_step_ids(cfg.preprocess.plan)
    view_preprocess_steps = (
        _views_preprocess_step_ids(cfg.views.plan) if cfg.views is not None else []
    )
    normalization_requested = _collect_requested_normalization(cfg.preprocess.plan)
    for pre_key in (cfg.views.plan if cfg.views is not None else {}).get("views", []):
        if isinstance(pre_key, Mapping) and isinstance(pre_key.get("preprocess"), Mapping):
            normalization_requested.extend(_collect_requested_normalization(pre_key["preprocess"]))

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
            "resolved": {},
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
    }

    try:
        _benchmark_contract_preflight(
            cfg=cfg,
            raw=raw,
            preprocess_steps=preprocess_steps,
            view_preprocess_steps=view_preprocess_steps,
        )

        _LOGGER.info("Loading dataset: %s", cfg.dataset.id)
        dataset, dataset_info = ds_orch.load(cfg.dataset)
        dataset_has_graph = _dataset_has_graph(dataset)

        all_preprocess_steps = preprocess_steps + view_preprocess_steps

        _LOGGER.info("Preflight checks")
        if _requires_fit_indices(preprocess_steps) and cfg.preprocess.fit_on is None:
            raise BenchConfigError(
                "preprocess.fit_on must be set when the plan includes fittable steps",
                code="E_BENCH_FIT_ON_REQUIRED",
            )
        if _requires_fit_indices(view_preprocess_steps) and cfg.preprocess.fit_on is None:
            raise BenchConfigError(
                "preprocess.fit_on must be set when views include fittable preprocess steps",
                code="E_BENCH_FIT_ON_REQUIRED",
            )

        method_info = _preflight(
            cfg=cfg,
            dataset_info=dataset_info,
            preprocess_steps=all_preprocess_steps,
            dataset_has_graph=dataset_has_graph,
        )

        requires_torch = (
            _method_requires_torch(cfg.method.method_id, strict=cfg.run.benchmark_mode)
            if cfg.method.kind == "inductive"
            else False
        )
        resolved_device = _resolve_method_device(
            cfg.method.device.device,
            supports_gpu=method_info.supports_gpu,
            strict=cfg.run.benchmark_mode,
        )
        _record_resolved_device(raw, cfg=cfg, resolved_device=resolved_device)
        ctx.write_config_copy(raw)

        resolution["device"]["resolved"] = resolved_device
        artifacts["method"] = {
            "id": cfg.method.method_id,
            "kind": cfg.method.kind,
            "device": {
                "requested": cfg.method.device.device,
                "resolved": resolved_device,
                "dtype": cfg.method.device.dtype,
            },
        }

        if (
            cfg.run.benchmark_mode
            and cfg.method.kind == "inductive"
            and requires_torch
            and "core.to_torch" not in preprocess_steps
        ):
            raise BenchConfigError(
                "Torch inductive methods require preprocess step 'core.to_torch' in benchmark_mode",
                code="E_BENCH_PREPROCESS_TO_TORCH_REQUIRED",
            )

        artifacts["dataset"] = {
            "id": cfg.dataset.id,
            "info": dataset_info,
            "fingerprint": dataset.meta.get("dataset_fingerprint"),
        }

        _LOGGER.info("Sampling splits")
        sampling_seed = ctx.seed_for("sampling", cfg.sampling.seed)
        sampling = sampling_orch.run(
            dataset,
            plan_dict=cfg.sampling.plan,
            seed=sampling_seed,
            dataset_id=cfg.dataset.id,
        )
        use_test = _use_test_split(sampling)
        protocol["use_test_split"] = bool(use_test)
        resolution["splits"]["resolved"] = {
            "report_splits": list(cfg.evaluation.report_splits),
            "use_test_split": bool(use_test),
            "refs": dict(sampling.refs),
        }
        artifacts["sampling"] = {
            "seed": sampling_seed,
            "plan": sampling.plan,
            "split_fingerprint": sampling.split_fingerprint,
            "stats": sampling.stats,
        }

        if cfg.method.kind == "inductive" and sampling.is_graph():
            sampling = _graph_sampling_to_inductive(sampling)
            fallback_events.append(
                {
                    "code": "E_BENCH_GRAPH_SAMPLING_CONVERTED",
                    "message": "Converted graph masks to inductive indices for inductive method",
                }
            )

        _LOGGER.info("Preprocess")
        fit_indices = prep_orch.resolve_fit_indices(
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
                cache_dir=cfg.graph.cache_dir,
                include_test=use_test,
            )
            artifacts["graph"] = {
                "seed": graph_seed,
                "fingerprint": graph.meta.get("fingerprint"),
                "spec": cfg.graph.spec,
            }
            resolution["backend"]["resolved"]["graph"] = cfg.graph.spec.get("backend")
        elif dataset_has_graph:
            _LOGGER.info("Graph (dataset-provided)")
            if pre.dataset.train.edges is None:
                raise BenchConfigError(
                    "Graph dataset is missing train.edges",
                    code="E_BENCH_GRAPH_MISSING_EDGES",
                )
            n_nodes = int(pre.dataset.train.y.shape[0])
            graph = transductive_orch.graph_from_dataset(pre.dataset, n_nodes)
            artifacts["graph"] = {
                "seed": None,
                "fingerprint": graph.meta.get("fingerprint"),
                "spec": None,
                "source": "dataset",
            }
            resolution["backend"]["resolved"]["graph"] = "dataset"

        X_u_w = None
        X_u_s = None
        X_u_s_1 = None
        if (
            cfg.augmentation is not None
            and cfg.augmentation.enabled
            and cfg.method.kind == "inductive"
        ):
            _LOGGER.info("Augmentation")
            idx_u = np.asarray(sampling.indices["train_unlabeled"], dtype=np.int64)
            X_u = select_rows(pre.dataset.train.X, idx_u, context="main.augmentation")

            X_u_aug_input = X_u
            if isinstance(X_u, dict) and "x" in X_u:
                X_u_aug_input = X_u["x"]

            aug_seed = ctx.seed_for("augmentation", cfg.augmentation.seed)
            strong_views = 2 if cfg.method.method_id == "comatch" else 1
            X_u_w, X_u_s, X_u_s_1 = aug_orch.run(
                X_u_aug_input,
                weak_plan=cfg.augmentation.weak,
                strong_plan=cfg.augmentation.strong,
                seed=aug_seed,
                mode=cfg.augmentation.mode,
                modality=cfg.augmentation.modality,
                sample_ids=idx_u,
                strong_views=strong_views,
            )

            if isinstance(X_u, dict) and "x" in X_u:

                def _wrapg(aug_x: Any, ref: Mapping[str, Any]) -> Any:
                    if aug_x is None:
                        return None
                    d = dict(ref)
                    d["x"] = aug_x
                    return d

                X_u_w = _wrapg(X_u_w, X_u)
                X_u_s = _wrapg(X_u_s, X_u)
                X_u_s_1 = _wrapg(X_u_s_1, X_u)

            artifacts["augmentation"] = {"seed": aug_seed, "mode": cfg.augmentation.mode}

        masks = None
        expected_labeled_count = None
        if cfg.method.kind == "transductive":
            if graph is None:
                raise BenchConfigError(
                    "Transductive methods require a graph",
                    code="E_BENCH_GRAPH_REQUIRED",
                )
            n_train = int(pre.dataset.train.y.shape[0])
            n_test = (
                int(pre.dataset.test.y.shape[0])
                if use_test and pre.dataset.test is not None
                else None
            )
            masks = _mask_map_from_sampling(sampling, n_train=n_train, n_test=n_test)
            expected_labeled_count = _expected_labeled_count(sampling.stats)

        prepared_artifacts = {
            "pre": pre,
            "sampling": sampling,
            "views": views,
            "graph": graph,
            "X_u_w": X_u_w,
            "X_u_s": X_u_s,
            "X_u_s_1": X_u_s_1,
            "use_test": use_test,
            "masks": masks,
            "expected_labeled_count": expected_labeled_count,
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
            cfg = ExperimentConfig.from_dict(raw)
            effective_config_hash = hash_any(raw)
            patched_run_id = _derive_run_id(
                effective_config_hash=effective_config_hash,
                seed=int(cfg.run.seed),
                versions=versions,
            )
            _sync_ctx_run_identity(ctx, run_id=patched_run_id)
            ctx.write_config_copy(raw)
            resolution["backend"]["requested"]["method"] = cfg.method.params.get("backend")

        if cfg.method.kind == "inductive":
            _LOGGER.info("Method: %s", cfg.method.method_id)
            method_seed = ctx.seed_for("method", None)
            method, method_resolution = inductive_orch.run(
                pre,
                sampling,
                views=views,
                X_u_w=X_u_w,
                X_u_s=X_u_s,
                X_u_s_1=X_u_s_1,
                cfg=cfg.method,
                seed=method_seed,
                strict=cfg.run.benchmark_mode,
                requires_torch=requires_torch,
            )
            metrics = eval_orch.evaluate_inductive(
                method=method,
                pre=pre,
                sampling=sampling,
                report_splits=cfg.evaluation.report_splits,
                metrics=cfg.evaluation.metrics,
                method_id=cfg.method.method_id,
                views=views,
                strict=cfg.run.benchmark_mode,
            )
        else:
            _LOGGER.info("Method: %s", cfg.method.method_id)
            method_seed = ctx.seed_for("method", None)
            method, data, method_resolution = transductive_orch.run(
                dataset=pre.dataset,
                graph=graph,
                masks=masks,
                cfg=cfg.method,
                seed=method_seed,
                use_test_split=use_test,
                expected_labeled_count=expected_labeled_count,
                strict=cfg.run.benchmark_mode,
            )
            metrics = eval_orch.evaluate_transductive(
                method=method,
                data=data,
                report_splits=cfg.evaluation.report_splits,
                metrics=cfg.evaluation.metrics,
                masks=masks,
            )

        resolution["backend"]["resolved"]["method"] = method_resolution.get("backend")
        resolution["backend"]["resolved"]["classifier"] = method_resolution.get(
            "classifier_backend"
        )
        resolution["dtype"]["resolved"] = method_resolution.get("dtypes", {})
        resolution["normalization"]["resolved"] = method_resolution.get("normalization", {})

    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        error_code = extract_error_code(exc)
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
            },
            resolution=resolution,
            protocol=protocol,
            versions=versions,
            fallback_events=fallback_events,
            error=error,
            error_code=error_code,
        )
        if cfg.run.benchmark_mode or ctx.fail_fast:
            raise
        return 1

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
        },
        resolution=resolution,
        protocol=protocol,
        versions=versions,
        fallback_events=fallback_events,
        error=error,
        error_code=error_code,
    )
    return 0


def run_experiment(config_path: Path, *, num_runs: int | None = None) -> int:
    if num_runs is not None and num_runs <= 0:
        raise ValueError("num_runs must be > 0")

    raw = load_yaml(config_path)
    cfg = ExperimentConfig.from_dict(raw)

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
        return _run_experiment_single(config_path, raw=raw, cfg=cfg)

    failures = 0
    for i, seed in enumerate(seeds, start=1):
        run_name = sweep_run_name(cfg.run.name, seed=seed, index=i - 1, total=len(seeds))
        sweep_raw = apply_global_seed(raw, seed=seed, run_name=run_name)
        sweep_cfg = ExperimentConfig.from_dict(sweep_raw)
        _LOGGER.info("Seed sweep run %s/%s: seed=%s name=%s", i, len(seeds), seed, run_name)
        try:
            code = _run_experiment_single(config_path, raw=sweep_raw, cfg=sweep_cfg)
        except Exception:
            if cfg.run.benchmark_mode:
                raise
            failures += 1
            continue
        if code != 0:
            failures += 1
            if cfg.run.benchmark_mode:
                raise BenchRuntimeError(
                    "E_BENCH_SWEEP_FAILED",
                    "seed sweep aborted due to failed run in benchmark_mode",
                )

    if failures:
        _LOGGER.warning("Seed sweep finished with failures: %s/%s", failures, len(seeds))
        return 1
    _LOGGER.info("Seed sweep finished successfully: %s runs", len(seeds))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ModSSC benchmark orchestration")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level: none, basic, detailed (aliases: quiet, full).",
    )
    parser.add_argument(
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
    return run_experiment(Path(args.config), num_runs=args.num_runs)


if __name__ == "__main__":
    raise SystemExit(main())
