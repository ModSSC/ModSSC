from __future__ import annotations

import json
import logging
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
