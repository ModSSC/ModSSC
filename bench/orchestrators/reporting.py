from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from ..context import RunContext
from ..report_schema import validate_run_payload
from ..schema import ExperimentConfig

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
    finished_at = datetime.now(timezone.utc).isoformat()
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
