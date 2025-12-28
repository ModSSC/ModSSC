from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from ..context import RunContext
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
    error: str | None = None,
) -> None:
    start = perf_counter()
    finished_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "run": {
            "name": ctx.name,
            "seed": ctx.seed,
            "started_at": ctx.started_at,
            "finished_at": finished_at,
            "status": status,
            "config_path": str(ctx.config_path) if ctx.config_path else None,
        },
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
        },
        "artifacts": artifacts,
        "metrics": metrics,
        "hpo": hpo,
        "error": error,
    }
    ctx.write_json("run.json", payload)
    _LOGGER.info(
        "Run summary written: %s status=%s duration_s=%.3f",
        str(ctx.run_dir / "run.json"),
        status,
        perf_counter() - start,
    )
