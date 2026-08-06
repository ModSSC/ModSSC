from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .errors import CampaignError
from .models import CampaignTask


def _numeric_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    raw = payload.get("metrics")
    if not isinstance(raw, Mapping):
        raise CampaignError("E_CAMPAIGN_AGGREGATE", "run.json metrics must be a mapping")
    out: dict[str, float] = {}
    for split, metrics in raw.items():
        if not isinstance(metrics, Mapping):
            continue
        for metric, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                out[f"{split}.{metric}"] = numeric
    if not out:
        raise CampaignError("E_CAMPAIGN_AGGREGATE", "run.json contains no finite metric")
    return out


def _critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df=n - 1))
    except ImportError:  # pragma: no cover - scipy is a core ModSSC dependency
        return 1.96


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_successes(
    *,
    tasks: list[CampaignTask],
    states: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Aggregate only complete, seed-paired manifest cells.

    A standardized cell is valid only with its five manifest seeds. Paper cells
    require every repetition declared by their protocol card. Incomplete cells
    are recorded separately and never enter ``aggregates.csv`` or Parquet.
    """

    state_by_id = {str(state["task_id"]): state for state in states}
    cells: dict[tuple[str, str], list[CampaignTask]] = defaultdict(list)
    for task in tasks:
        cell_id = task.protocol_id or task.config_path
        cells[(task.track, cell_id)].append(task)

    aggregate_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    for (track, cell_id), cell_tasks in sorted(cells.items()):
        resource_assignments = {(task.assigned_site, task.resource_profile) for task in cell_tasks}
        if len(resource_assignments) != 1:
            raise CampaignError(
                "E_CAMPAIGN_AGGREGATE",
                f"cell {cell_id} mixes site/resource profiles: "
                f"{sorted(resource_assignments)}; reprofile every seed in the cell together",
            )
        assigned_site, resource_profile = next(iter(resource_assignments))
        required_seed_counts = {task.required_seed_count for task in cell_tasks}
        if len(required_seed_counts) != 1:
            raise CampaignError(
                "E_CAMPAIGN_AGGREGATE",
                f"cell {cell_id} mixes required seed counts: {sorted(required_seed_counts)}",
            )
        required_seed_count = next(iter(required_seed_counts))
        expected_seeds = {task.seed for task in cell_tasks}
        if len(expected_seeds) != len(cell_tasks):
            raise CampaignError(
                "E_CAMPAIGN_AGGREGATE", f"duplicate manifest seed in cell {cell_id}"
            )
        invalid_expected_count = len(expected_seeds) != required_seed_count

        successes: list[tuple[CampaignTask, Path]] = []
        for task in cell_tasks:
            state = state_by_id[task.task_id]
            paths = state.get("run_json_paths", [])
            if state.get("status") == "success" and isinstance(paths, list) and len(paths) == 1:
                successes.append((task, Path(paths[0])))
        success_seeds = {task.seed for task, _ in successes}
        complete = (
            not invalid_expected_count
            and success_seeds == expected_seeds
            and len(successes) == len(cell_tasks)
        )
        if not complete:
            incomplete_rows.append(
                {
                    "track": track,
                    "cell_id": cell_id,
                    "method_id": cell_tasks[0].method_id,
                    "dataset_id": cell_tasks[0].dataset_id,
                    "n_expected": len(expected_seeds),
                    "n_success": len(successes),
                    "missing_seeds": ",".join(
                        str(seed) for seed in sorted(expected_seeds - success_seeds)
                    ),
                    "reason": (
                        f"manifest_requires_{required_seed_count}_seeds"
                        if invalid_expected_count
                        else "missing_successes"
                    ),
                }
            )
            continue

        by_metric: dict[str, list[float]] = defaultdict(list)
        reference_keys: set[str] | None = None
        for task, run_path in sorted(successes, key=lambda item: item[0].seed):
            payload = json.loads(run_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise CampaignError("E_CAMPAIGN_AGGREGATE", f"invalid run.json: {run_path}")
            metrics = _numeric_metrics(payload)
            keys = set(metrics)
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                raise CampaignError(
                    "E_CAMPAIGN_AGGREGATE", f"metric schema differs within cell {cell_id}"
                )
            for metric_key, value in sorted(metrics.items()):
                by_metric[metric_key].append(value)
                split, metric = metric_key.split(".", 1)
                raw_rows.append(
                    {
                        "track": track,
                        "cell_id": cell_id,
                        "task_id": task.task_id,
                        "method_id": task.method_id,
                        "method_profile": task.method_profile,
                        "dataset_id": task.dataset_id,
                        "label_budget": task.label_budget,
                        "assigned_site": assigned_site,
                        "resource_profile": resource_profile,
                        "seed": task.seed,
                        "split": split,
                        "metric": metric,
                        "value": value,
                    }
                )

        for metric_key, values in sorted(by_metric.items()):
            arr = np.asarray(values, dtype=np.float64)
            n = int(arr.size)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            population_std = float(arr.std(ddof=0))
            half_width = _critical_95(n) * std / math.sqrt(n) if n > 1 else float("nan")
            split, metric = metric_key.split(".", 1)
            first = cell_tasks[0]
            aggregate_rows.append(
                {
                    "track": track,
                    "cell_id": cell_id,
                    "method_id": first.method_id,
                    "method_profile": first.method_profile,
                    "dataset_id": first.dataset_id,
                    "label_budget": first.label_budget,
                    "assigned_site": assigned_site,
                    "resource_profile": resource_profile,
                    "split": split,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "std_ddof": 1,
                    "population_std": population_std,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                }
            )

    aggregate_fields = [
        "track",
        "cell_id",
        "method_id",
        "method_profile",
        "dataset_id",
        "label_budget",
        "assigned_site",
        "resource_profile",
        "split",
        "metric",
        "n",
        "mean",
        "std",
        "std_ddof",
        "population_std",
        "ci95_low",
        "ci95_high",
    ]
    raw_fields = [
        "track",
        "cell_id",
        "task_id",
        "method_id",
        "method_profile",
        "dataset_id",
        "label_budget",
        "assigned_site",
        "resource_profile",
        "seed",
        "split",
        "metric",
        "value",
    ]
    incomplete_fields = [
        "track",
        "cell_id",
        "method_id",
        "dataset_id",
        "n_expected",
        "n_success",
        "missing_seeds",
        "reason",
    ]
    _write_csv(output_dir / "aggregates.csv", aggregate_rows, aggregate_fields)
    _write_csv(output_dir / "paired-runs.csv", raw_rows, raw_fields)
    _write_csv(output_dir / "incomplete-cells.csv", incomplete_rows, incomplete_fields)

    parquet_paths: list[str] = []
    try:
        import pandas as pd

        for filename, rows, columns in (
            ("aggregates.parquet", aggregate_rows, aggregate_fields),
            ("paired-runs.parquet", raw_rows, raw_fields),
        ):
            path = output_dir / filename
            pd.DataFrame(rows, columns=columns).to_parquet(path, index=False)
            parquet_paths.append(str(path))
    except (ImportError, ModuleNotFoundError):
        pass

    return {
        "complete_cells": len(cells) - len(incomplete_rows),
        "incomplete_cells": len(incomplete_rows),
        "aggregate_rows": len(aggregate_rows),
        "paired_rows": len(raw_rows),
        "aggregates_csv": str(output_dir / "aggregates.csv"),
        "paired_runs_csv": str(output_dir / "paired-runs.csv"),
        "incomplete_cells_csv": str(output_dir / "incomplete-cells.csv"),
        "parquet_paths": parquet_paths,
    }
