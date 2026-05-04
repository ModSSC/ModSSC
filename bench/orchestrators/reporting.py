from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev
from time import perf_counter
from typing import Any

from ..context import RunContext
from ..report_schema import validate_run_payload
from ..schema import ExperimentConfig
from ..utils.io import write_json

_LOGGER = logging.getLogger(__name__)


def _detect_gpu_device() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "Unknown"

    try:
        if not torch.cuda.is_available():
            return "CPU"
        device_count = int(torch.cuda.device_count())
        if device_count > 1:
            names = [str(torch.cuda.get_device_name(index)) for index in range(device_count)]
            return ", ".join(names)
        return str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception:
        return "Unknown"


def _hardware_mismatch_reason(
    *,
    gpu_device: str,
    hardware_profile: str | None,
    resolved_device: str | None,
) -> str | None:
    gpu_lower = str(gpu_device or "").lower()
    resolved_lower = str(resolved_device or "").lower()

    if resolved_lower.startswith("cuda") and gpu_lower == "cpu":
        return "resolved device is cuda but detected GPU is CPU"

    profile = str(hardware_profile or "").strip().lower()
    if profile and profile not in {"auto", "none", "default", "unknown", "cpu", "cuda"}:
        if not gpu_lower or gpu_lower == "unknown":
            return "specific hardware profile requested but GPU name is unavailable"
        if profile not in gpu_lower:
            return f"profile {hardware_profile} not found in detected GPU {gpu_device}"

    return None


def _build_run_info(
    *,
    ctx: RunContext,
    cfg: ExperimentConfig,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    method_block = artifacts.get("method", {})
    method_device = method_block.get("device", {}) if isinstance(method_block, Mapping) else {}
    resolved_device = method_device.get("resolved") if isinstance(method_device, Mapping) else None
    hardware_profile = cfg.limits.profile if cfg.limits is not None else None
    gpu_device = _detect_gpu_device()
    mismatch_reason = _hardware_mismatch_reason(
        gpu_device=gpu_device,
        hardware_profile=hardware_profile,
        resolved_device=resolved_device,
    )
    try:
        started_at = datetime.fromisoformat(str(ctx.started_at))
        elapsed_seconds = (datetime.now(UTC) - started_at).total_seconds()
    except Exception:
        elapsed_seconds = 0.0
    return {
        "run_time_seconds": float(max(0.0, elapsed_seconds)),
        "device_requested": (
            method_device.get("requested") if isinstance(method_device, Mapping) else None
        ),
        "device_resolved": resolved_device,
        "gpu_device": gpu_device,
        "limits_profile": hardware_profile,
        "hardware_profile": hardware_profile,
        "hardware_mismatch": mismatch_reason is not None,
        "hardware_mismatch_reason": mismatch_reason,
    }


def _class_counts(stats: Any, split: str) -> dict[str, int] | None:
    if not isinstance(stats, Mapping):
        return None
    block = stats.get(split)
    if not isinstance(block, Mapping):
        return None
    classes = block.get("classes")
    if not isinstance(classes, Mapping):
        return None
    out: dict[str, int] = {}
    for key, value in classes.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            out[str(key)] = int(value)
    return out


def _effective_n_classes(stats: Any) -> int | None:
    classes: set[str] = set()
    for split in ("train", "val", "test", "train_labeled"):
        counts = _class_counts(stats, split)
        if counts:
            classes.update(counts.keys())
    return len(classes) if classes else None


def _uniform_count(counts: dict[str, int] | None) -> int | None:
    if not counts:
        return None
    values = {int(v) for v in counts.values()}
    if len(values) != 1:
        return None
    return int(next(iter(values)))


def _build_task_info(*, cfg: ExperimentConfig, artifacts: dict[str, Any]) -> dict[str, Any]:
    sampling = artifacts.get("sampling")
    stats = sampling.get("stats") if isinstance(sampling, Mapping) else None
    train_labeled_counts = _class_counts(stats, "train_labeled")
    test_counts = _class_counts(stats, "test")
    train_counts = _class_counts(stats, "train")
    n_classes = _effective_n_classes(stats)
    return {
        "dataset_id": cfg.dataset.id,
        "method_id": cfg.method.method_id,
        "method_kind": cfg.method.kind,
        "n_classes": n_classes,
        "effective_n_classes": n_classes,
        "class_filter": cfg.dataset.options.get("class_filter"),
        "train_labeled_per_class": _uniform_count(train_labeled_counts),
        "class_counts_train_labeled": train_labeled_counts,
        "class_counts_train": train_counts,
        "class_counts_test": test_counts,
    }


def _build_graph_info(*, artifacts: dict[str, Any]) -> dict[str, Any] | None:
    graph = artifacts.get("graph")
    if not isinstance(graph, Mapping):
        return None
    info = graph.get("info")
    if not isinstance(info, Mapping):
        return None
    return dict(info)


def write_run_summary(
    *,
    ctx: RunContext,
    cfg: ExperimentConfig,
    artifacts: dict[str, Any],
    metrics: dict[str, Any] | None,
    hpo: dict[str, Any] | None,
    status: str,
    hashes: dict[str, Any],
    resolution: dict[str, Any],
    protocol: dict[str, Any],
    versions: dict[str, Any],
    fallback_events: list[dict[str, Any]],
    error: str | None = None,
    error_code: str | None = None,
) -> None:
    start = perf_counter()
    finished_at = datetime.now(UTC).isoformat()
    run_info = _build_run_info(ctx=ctx, cfg=cfg, artifacts=artifacts)
    task_info = _build_task_info(cfg=cfg, artifacts=artifacts)
    graph_info = _build_graph_info(artifacts=artifacts)
    payload = {
        "run": {
            "name": ctx.name,
            "seed": ctx.seed,
            "run_id": ctx.run_id,
            "started_at": ctx.started_at,
            "finished_at": finished_at,
            "status": status,
            "benchmark_mode": bool(cfg.run.benchmark_mode),
            "config_path": str(ctx.config_path) if ctx.config_path else None,
            "error_code": error_code,
        },
        "hashes": hashes,
        "resolution": resolution,
        "protocol": protocol,
        "versions": versions,
        "run_info": run_info,
        "task_info": task_info,
        "graph_info": graph_info,
        "config": {
            "run": asdict(cfg.run),
            "dataset": asdict(cfg.dataset),
            "sampling": asdict(cfg.sampling),
            "preprocess": asdict(cfg.preprocess),
            "method": asdict(cfg.method),
            "evaluation": asdict(cfg.evaluation),
            "graph": asdict(cfg.graph) if cfg.graph else None,
            "views": asdict(cfg.views) if cfg.views else None,
            "augmentation": asdict(cfg.augmentation) if cfg.augmentation else None,
            "search": asdict(cfg.search) if cfg.search else None,
            "limits": asdict(cfg.limits) if cfg.limits else None,
        },
        "artifacts": artifacts,
        "metrics": metrics,
        "hpo": hpo,
        "fallback_events": fallback_events,
        "error": error,
    }
    validate_run_payload(payload)
    ctx.write_json("run.json", payload)
    _LOGGER.info(
        "Run summary written: %s status=%s duration_s=%.3f",
        str(ctx.run_dir / "run.json"),
        status,
        perf_counter() - start,
    )


def _iter_numeric_metric_leaves(
    obj: Any, *, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], float]]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, int | float):
        return [(path, float(obj))]
    if isinstance(obj, dict):
        out: list[tuple[tuple[str, ...], float]] = []
        for key, value in obj.items():
            out.extend(_iter_numeric_metric_leaves(value, path=path + (str(key),)))
        return out
    return []


def _set_nested(mapping: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if not path:
        raise ValueError("path must be non-empty")
    cur = mapping
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def write_seed_sweep_summary(
    *,
    output_dir: Path,
    config_path: Path,
    base_name: str,
    requested_seeds: list[int],
    run_json_paths: list[Path],
) -> Path:
    start = perf_counter()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in run_json_paths]

    run_entries: list[dict[str, Any]] = []
    successful_reports: list[dict[str, Any]] = []
    metric_values: dict[tuple[str, ...], list[float]] = {}

    for report, run_json_path in zip(reports, run_json_paths, strict=True):
        run_block = report.get("run", {})
        status = str(run_block.get("status", "unknown"))
        metrics = report.get("metrics")
        run_entries.append(
            {
                "seed": run_block.get("seed"),
                "name": run_block.get("name"),
                "run_id": run_block.get("run_id"),
                "status": status,
                "run_dir": str(run_json_path.parent),
                "run_json": str(run_json_path),
                "error_code": run_block.get("error_code"),
                "error": report.get("error"),
                "run_info": report.get("run_info"),
                "task_info": report.get("task_info"),
                "graph_info": report.get("graph_info"),
                "metrics": metrics,
            }
        )
        if status != "success" or not isinstance(metrics, dict):
            continue
        successful_reports.append(report)
        for metric_path, value in _iter_numeric_metric_leaves(metrics):
            metric_values.setdefault(metric_path, []).append(float(value))

    aggregated_metrics: dict[str, Any] = {}
    for metric_path, values in sorted(metric_values.items()):
        _set_nested(
            aggregated_metrics,
            metric_path,
            {
                "count": len(values),
                "mean": float(fmean(values)),
                "std": float(pstdev(values)) if len(values) > 1 else 0.0,
                "min": float(min(values)),
                "max": float(max(values)),
                "values": [float(v) for v in values],
            },
        )

    failed_count = len(run_entries) - len(successful_reports)
    if failed_count == 0:
        status = "success"
    elif successful_reports:
        status = "partial_failure"
    else:
        status = "failed"

    runtime_values = [
        float(report["run_info"]["run_time_seconds"])
        for report in reports
        if isinstance(report.get("run_info"), Mapping)
        and isinstance(report["run_info"].get("run_time_seconds"), int | float)
    ]
    gpu_devices = sorted(
        {
            str(report["run_info"].get("gpu_device"))
            for report in reports
            if isinstance(report.get("run_info"), Mapping) and report["run_info"].get("gpu_device")
        }
    )
    hardware_mismatch_count = sum(
        1
        for report in reports
        if isinstance(report.get("run_info"), Mapping)
        and report["run_info"].get("hardware_mismatch") is True
    )
    run_info: dict[str, Any] = {
        "gpu_devices": gpu_devices,
        "gpu_device": (
            gpu_devices[0] if len(gpu_devices) == 1 else ("Mixed" if gpu_devices else None)
        ),
        "hardware_mismatch": hardware_mismatch_count > 0,
        "hardware_mismatch_count": hardware_mismatch_count,
    }
    if runtime_values:
        run_info["run_time_seconds"] = {
            "count": len(runtime_values),
            "mean": float(fmean(runtime_values)),
            "std": float(pstdev(runtime_values)) if len(runtime_values) > 1 else 0.0,
            "min": float(min(runtime_values)),
            "max": float(max(runtime_values)),
            "values": [float(v) for v in runtime_values],
        }

    payload = {
        "sweep": {
            "base_name": str(base_name),
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "requested_seeds": [int(seed) for seed in requested_seeds],
            "requested_run_count": len(requested_seeds),
            "completed_run_count": len(run_entries),
            "successful_run_count": len(successful_reports),
            "failed_run_count": failed_count,
            "status": status,
            "aggregated_at": datetime.now(UTC).isoformat(),
        },
        "metrics": aggregated_metrics,
        "run_info": run_info,
        "runs": run_entries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "aggregate.json"
    write_json(out_path, payload)
    _LOGGER.info(
        "Seed sweep aggregate written: %s status=%s duration_s=%.3f",
        str(out_path),
        status,
        perf_counter() - start,
    )
    return out_path
