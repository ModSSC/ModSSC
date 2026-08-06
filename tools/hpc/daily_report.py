from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from bench.campaign.errors import CampaignError
from bench.campaign.governance import load_resource_catalog
from bench.campaign.manifest import load_manifest, write_text_atomic
from bench.campaign.models import CampaignTask
from bench.campaign.reconcile import materialize_reconcile_paths
from bench.utils.io import atomic_write_json

from .preflight import load_allocation_snapshot
from .resources import format_duration

_FAILED_TASK_STATUSES = {
    "blocked",
    "conflict",
    "corrupt",
    "failed",
    "resource_blocked",
    "stale",
}
_RETRY_QUEUE_STATUSES = {"failed", "missing", "stale"}

_RAM_BYTES_PATHS = (
    ("peak_ram_bytes",),
    ("max_rss_bytes",),
    ("resource_usage", "peak_ram_bytes"),
    ("resource_usage", "max_rss_bytes"),
)
_RAM_MIB_PATHS = (
    ("peak_ram_mib",),
    ("max_rss_mib",),
    ("resource_usage", "peak_ram_mib"),
    ("resource_usage", "max_rss_mib"),
)
_VRAM_BYTES_PATHS = (
    ("peak_vram_bytes",),
    ("max_gpu_memory_bytes",),
    ("max_gpu_memory_reserved_bytes",),
    ("max_gpu_memory_allocated_bytes",),
    ("resource_usage", "peak_vram_bytes"),
    ("resource_usage", "max_gpu_memory_bytes"),
    ("resource_usage", "max_gpu_memory_reserved_bytes"),
    ("resource_usage", "max_gpu_memory_allocated_bytes"),
)
_VRAM_MIB_PATHS = (
    ("peak_vram_mib",),
    ("max_gpu_memory_mib",),
    ("resource_usage", "peak_vram_mib"),
    ("resource_usage", "max_gpu_memory_mib"),
)


@dataclass(frozen=True)
class DailyUsageReport:
    campaign_id: str
    task_count: int
    status: str
    json_path: str
    summary_csv_path: str
    runs_csv_path: str


def _read_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(code, f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise CampaignError(code, f"JSON root must be an object: {path}")
    return value


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None


def _nested_number(payload: Mapping[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return _finite_nonnegative(value)


def _first_number(
    payload: Mapping[str, Any], paths: Iterable[tuple[str, ...]]
) -> tuple[float | None, str | None]:
    for path in paths:
        value = _nested_number(payload, path)
        if value is not None:
            return value, ".".join(path)
    return None, None


def _timestamp_runtime(run: Mapping[str, Any]) -> float | None:
    try:
        started = datetime.fromisoformat(str(run["started_at"]))
        finished = datetime.fromisoformat(str(run["finished_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=UTC)
    elapsed = (finished.astimezone(UTC) - started.astimezone(UTC)).total_seconds()
    return elapsed if elapsed >= 0.0 else None


def _run_measurements(path: Path) -> dict[str, Any]:
    payload = _read_object(path, code="E_CAMPAIGN_DAILY_RUN_JSON")
    run_info = payload.get("run_info")
    info = run_info if isinstance(run_info, Mapping) else {}

    runtime = _finite_nonnegative(info.get("run_time_seconds"))
    runtime_source = "run_info.run_time_seconds" if runtime is not None else None
    if runtime is None:
        run = payload.get("run")
        runtime = _timestamp_runtime(run) if isinstance(run, Mapping) else None
        runtime_source = "run.started_at/finished_at" if runtime is not None else None

    ram_bytes, ram_source = _first_number(info, _RAM_BYTES_PATHS)
    if ram_bytes is None:
        ram_mib, ram_source = _first_number(info, _RAM_MIB_PATHS)
        ram_bytes = None if ram_mib is None else ram_mib * 1024.0 * 1024.0
    vram_bytes, vram_source = _first_number(info, _VRAM_BYTES_PATHS)
    if vram_bytes is None:
        vram_mib, vram_source = _first_number(info, _VRAM_MIB_PATHS)
        vram_bytes = None if vram_mib is None else vram_mib * 1024.0 * 1024.0

    return {
        "runtime_seconds": runtime,
        "runtime_source": runtime_source,
        "peak_ram_bytes": ram_bytes,
        "peak_ram_source": ram_source,
        "peak_vram_bytes": vram_bytes,
        "peak_vram_source": vram_source,
    }


def _load_reconcile_states(
    reconcile_path: Path,
    *,
    campaign_id: str,
    manifest_sha256: str,
    tasks: list[CampaignTask],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = materialize_reconcile_paths(
        reconcile_path,
        _read_object(reconcile_path, code="E_CAMPAIGN_DAILY_RECONCILE"),
    )
    if payload.get("campaign_id") != campaign_id:
        raise CampaignError(
            "E_CAMPAIGN_DAILY_RECONCILE", "reconcile campaign_id differs from manifest"
        )
    if payload.get("manifest_sha256") != manifest_sha256:
        raise CampaignError(
            "E_CAMPAIGN_DAILY_RECONCILE", "reconcile manifest digest differs from manifest"
        )
    raw_states = payload.get("tasks")
    if not isinstance(raw_states, list):
        raise CampaignError("E_CAMPAIGN_DAILY_RECONCILE", "reconcile tasks must be a list")

    expected = {task.task_id for task in tasks}
    states: dict[str, dict[str, Any]] = {}
    for raw in raw_states:
        if not isinstance(raw, dict):
            raise CampaignError(
                "E_CAMPAIGN_DAILY_RECONCILE", "each reconcile task must be an object"
            )
        task_id = raw.get("task_id")
        if not isinstance(task_id, str) or task_id not in expected:
            raise CampaignError(
                "E_CAMPAIGN_DAILY_RECONCILE", f"unknown reconcile task_id: {task_id!r}"
            )
        if task_id in states:
            raise CampaignError(
                "E_CAMPAIGN_DAILY_RECONCILE", f"duplicate reconcile task_id: {task_id}"
            )
        if not isinstance(raw.get("status"), str):
            raise CampaignError(
                "E_CAMPAIGN_DAILY_RECONCILE", f"status is missing for task {task_id}"
            )
        states[task_id] = raw
    missing = sorted(expected - states.keys())
    if missing:
        raise CampaignError(
            "E_CAMPAIGN_DAILY_RECONCILE",
            f"reconcile is missing {len(missing)} manifest task(s)",
        )
    return payload, states


def _attempt_records(task: CampaignTask, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        return []
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_path in attempts:
        if not isinstance(raw_path, str):
            continue
        attempt_dir = Path(raw_path)
        attempt_path = attempt_dir / "attempt.json"
        try:
            attempt = _read_object(attempt_path, code="E_CAMPAIGN_DAILY_ATTEMPT")
        except CampaignError:
            records.append(
                {
                    "task_id": task.task_id,
                    "attempt_id": None,
                    "attempt_kind": "failed",
                    "run_json_path": None,
                    "runtime_seconds": None,
                    "runtime_source": None,
                    "peak_ram_bytes": None,
                    "peak_ram_source": None,
                    "peak_vram_bytes": None,
                    "peak_vram_source": None,
                    "measurement_error": f"unreadable attempt metadata: {attempt_path}",
                }
            )
            continue
        attempt_id = attempt.get("attempt_id")
        deduplication_key = str(attempt_id) if attempt_id else str(attempt_dir.resolve())
        if deduplication_key in seen_ids:
            continue
        seen_ids.add(deduplication_key)
        run_path = attempt_dir / "run" / "run.json"
        measurement: dict[str, Any]
        if run_path.is_file():
            try:
                measurement = _run_measurements(run_path)
                measurement_error = None
            except CampaignError as exc:
                measurement = {
                    "runtime_seconds": None,
                    "runtime_source": None,
                    "peak_ram_bytes": None,
                    "peak_ram_source": None,
                    "peak_vram_bytes": None,
                    "peak_vram_source": None,
                }
                measurement_error = str(exc)
        else:
            measurement = {
                "runtime_seconds": None,
                "runtime_source": None,
                "peak_ram_bytes": None,
                "peak_ram_source": None,
                "peak_vram_bytes": None,
                "peak_vram_source": None,
            }
            measurement_error = None
        records.append(
            {
                "task_id": task.task_id,
                "attempt_id": str(attempt_id) if attempt_id is not None else None,
                "attempt_kind": "failed",
                "run_json_path": str(run_path) if run_path.is_file() else None,
                **measurement,
                "measurement_error": measurement_error,
            }
        )
    return records


def _measurement_records(
    tasks: list[CampaignTask], states: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        state = states[task.task_id]
        records.extend(_attempt_records(task, state))
        if state.get("status") != "success":
            continue
        raw_paths = state.get("run_json_paths")
        run_path = Path(raw_paths[0]) if isinstance(raw_paths, list) and raw_paths else None
        if run_path is None or not run_path.is_file():
            records.append(
                {
                    "task_id": task.task_id,
                    "attempt_id": None,
                    "attempt_kind": "success",
                    "run_json_path": str(run_path) if run_path is not None else None,
                    "runtime_seconds": None,
                    "runtime_source": None,
                    "peak_ram_bytes": None,
                    "peak_ram_source": None,
                    "peak_vram_bytes": None,
                    "peak_vram_source": None,
                    "measurement_error": "successful task has no readable run.json",
                }
            )
            continue
        try:
            measurement = _run_measurements(run_path)
            error = None
        except CampaignError as exc:
            measurement = {
                "runtime_seconds": None,
                "runtime_source": None,
                "peak_ram_bytes": None,
                "peak_ram_source": None,
                "peak_vram_bytes": None,
                "peak_vram_source": None,
            }
            error = str(exc)
        records.append(
            {
                "task_id": task.task_id,
                "attempt_id": None,
                "attempt_kind": "success",
                "run_json_path": str(run_path),
                **measurement,
                "measurement_error": error,
            }
        )
    return records


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _summarize(
    tasks: list[CampaignTask],
    states: Mapping[str, Mapping[str, Any]],
    records_by_task: Mapping[str, list[Mapping[str, Any]]],
    resources: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(str(states[task.task_id]["status"]) for task in tasks)
    task_count = len(tasks)
    success_count = statuses.get("success", 0)
    failure_count = sum(statuses.get(status, 0) for status in _FAILED_TASK_STATUSES)
    retry_queue_ids = [
        task.task_id
        for task in tasks
        if str(states[task.task_id]["status"]) in _RETRY_QUEUE_STATUSES
    ]
    retried_ids = [
        task.task_id
        for task in tasks
        if any(record.get("attempt_kind") == "failed" for record in records_by_task[task.task_id])
    ]
    records = [record for task in tasks for record in records_by_task[task.task_id]]
    runtimes = [
        float(record["runtime_seconds"])
        for record in records
        if _finite_nonnegative(record.get("runtime_seconds")) is not None
    ]
    success_runtimes = [
        float(record["runtime_seconds"])
        for record in records
        if record.get("attempt_kind") == "success"
        and _finite_nonnegative(record.get("runtime_seconds")) is not None
    ]
    metadata_missing = sorted(
        {
            f"{task.assigned_site}.{task.resource_profile}"
            for task in tasks
            if (task.assigned_site, task.resource_profile) not in resources
        }
    )
    observed_gpu_seconds = 0.0
    for task in tasks:
        resource = resources.get((task.assigned_site, task.resource_profile))
        if resource is None:
            continue
        accelerators = int(resource.get("accelerators_per_task", 0))
        observed_gpu_seconds += accelerators * sum(
            float(record["runtime_seconds"])
            for record in records_by_task[task.task_id]
            if _finite_nonnegative(record.get("runtime_seconds")) is not None
        )

    projected_total_gpu_seconds = 0.0
    projected_remaining_gpu_seconds = 0.0
    projection_missing: list[str] = []
    grouped_tasks: dict[tuple[str, str], list[CampaignTask]] = defaultdict(list)
    for task in tasks:
        grouped_tasks[(task.assigned_site, task.resource_profile)].append(task)
    for key, profile_tasks in grouped_tasks.items():
        resource = resources.get(key)
        if resource is None:
            projection_missing.append(f"{key[0]}.{key[1]}")
            continue
        accelerators = int(resource.get("accelerators_per_task", 0))
        if accelerators == 0:
            continue
        profile_success_runtimes = [
            float(record["runtime_seconds"])
            for task in profile_tasks
            for record in records_by_task[task.task_id]
            if record.get("attempt_kind") == "success"
            and _finite_nonnegative(record.get("runtime_seconds")) is not None
        ]
        if not profile_success_runtimes:
            projection_missing.append(f"{key[0]}.{key[1]}")
            continue
        mean_runtime = fmean(profile_success_runtimes)
        profile_successes = sum(
            states[task.task_id]["status"] == "success" for task in profile_tasks
        )
        projected_total_gpu_seconds += mean_runtime * len(profile_tasks) * accelerators
        projected_remaining_gpu_seconds += (
            mean_runtime * (len(profile_tasks) - profile_successes) * accelerators
        )
    ram_values = [
        float(record["peak_ram_bytes"])
        for record in records
        if _finite_nonnegative(record.get("peak_ram_bytes")) is not None
    ]
    vram_values = [
        float(record["peak_vram_bytes"])
        for record in records
        if _finite_nonnegative(record.get("peak_vram_bytes")) is not None
    ]
    if success_runtimes:
        projected_mean = fmean(success_runtimes)
        projected_total_hours = projected_mean * task_count / 3600.0
        projected_remaining_hours = projected_mean * (task_count - success_count) / 3600.0
    else:
        projected_total_hours = None
        projected_remaining_hours = None
    return {
        "task_count": task_count,
        "status_counts": dict(sorted(statuses.items())),
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / task_count if task_count else 0.0,
        "failure_rate": failure_count / task_count if task_count else 0.0,
        "retry_queue_task_count": len(retry_queue_ids),
        "retry_queue_task_ids": retry_queue_ids,
        "tasks_with_failed_attempts_count": len(retried_ids),
        "tasks_with_failed_attempts_ids": retried_ids,
        "failed_attempt_count": sum(record.get("attempt_kind") == "failed" for record in records),
        "observed_runtime_count": len(runtimes),
        "missing_runtime_count": len(records) - len(runtimes),
        "observed_task_hours": sum(runtimes) / 3600.0,
        "runtime_p50_seconds": _percentile(runtimes, 0.50),
        "runtime_p95_seconds": _percentile(runtimes, 0.95),
        "success_runtime_p95_seconds": _percentile(success_runtimes, 0.95),
        "peak_ram_observation_count": len(ram_values),
        "peak_ram_bytes": max(ram_values) if ram_values else None,
        "peak_vram_observation_count": len(vram_values),
        "peak_vram_bytes": max(vram_values) if vram_values else None,
        "projection_success_sample_count": len(success_runtimes),
        "projected_total_task_hours": projected_total_hours,
        "projected_remaining_task_hours": projected_remaining_hours,
        "resource_metadata_missing": metadata_missing,
        "observed_gpu_hours": (None if metadata_missing else observed_gpu_seconds / 3600.0),
        "gpu_projection_missing_profiles": sorted(set(projection_missing)),
        "projected_total_gpu_hours": (
            None if projection_missing else projected_total_gpu_seconds / 3600.0
        ),
        "projected_remaining_gpu_hours": (
            None if projection_missing else projected_remaining_gpu_seconds / 3600.0
        ),
    }


def _annotated_summary(
    *, group_kind: str, site: str | None, resource: str | None, summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "group_kind": group_kind,
        "site": site,
        "resource_profile": resource,
        **summary,
    }


def _resource_policy(
    *,
    tasks: list[CampaignTask],
    states: Mapping[str, Mapping[str, Any]],
    summary: Mapping[str, Any],
    resource: Mapping[str, Any] | None,
    explained_oom_task_ids: set[str],
) -> dict[str, Any]:
    if resource is None:
        return {
            "architecture": None,
            "accelerators_per_task": None,
            "recommended_concurrency": None,
            "promotion_eligible": False,
            "promotion_blockers": ["resource metadata is missing"],
            "recommended_walltime_seconds": None,
            "recommended_walltime": None,
            "walltime_status": "unknown",
            "projected_completion_hours": None,
        }
    architecture = str(resource["architecture"])
    accelerators = int(resource["accelerators_per_task"])
    initial = int(resource["initial_concurrency"])
    promoted = int(resource["promoted_concurrency"])
    minimum_successes = int(resource["promotion_min_successes"])
    maximum_failure_rate = float(resource["promotion_max_failure_rate"])
    success_count = int(summary["success_count"])
    failure_count = int(summary["failure_count"])
    completed_count = success_count + failure_count
    failure_rate = failure_count / completed_count if completed_count else 0.0
    unexplained_oom = sorted(
        task.task_id
        for task in tasks
        if states[task.task_id].get("latest_failure_class") == "resource_oom"
        and task.task_id not in explained_oom_task_ids
    )
    blockers: list[str] = []
    if success_count < minimum_successes:
        blockers.append(f"{success_count} successes < {minimum_successes}")
    if failure_rate >= maximum_failure_rate:
        blockers.append(f"failure rate {failure_rate:.4f} is not below {maximum_failure_rate:.4f}")
    if unexplained_oom:
        blockers.append(f"{len(unexplained_oom)} unexplained OOM task(s)")
    promotion_eligible = accelerators > 0 and not blockers
    recommended_concurrency = promoted if promotion_eligible else initial

    p95 = _finite_nonnegative(summary.get("success_runtime_p95_seconds"))
    if p95 is None:
        recommended_walltime = int(resource["configured_walltime_seconds"])
        walltime_status = "awaiting_observations"
    else:
        raw_walltime = max(60, math.ceil(1.25 * p95 / 60.0) * 60)
        maximum_walltime = int(resource["max_walltime_seconds"])
        # Preserve the measured requirement even when the current profile
        # cannot satisfy it.  Silently clamping to the scheduler cap would
        # turn a required reprofile/block decision into a likely timeout.
        recommended_walltime = raw_walltime
        walltime_status = "exceeds_cap" if raw_walltime > maximum_walltime else "calibrated"

    remaining = len(tasks) - success_count
    projected_total = _finite_nonnegative(summary.get("projected_total_task_hours"))
    mean_runtime_hours = (
        projected_total / len(tasks) if projected_total is not None and tasks else None
    )
    projected_completion = (
        None
        if mean_runtime_hours is None
        else math.ceil(remaining / recommended_concurrency) * mean_runtime_hours
    )
    return {
        "architecture": architecture,
        "accelerators_per_task": accelerators,
        "initial_concurrency": initial,
        "promoted_concurrency": promoted,
        "recommended_concurrency": recommended_concurrency,
        "promotion_min_successes": minimum_successes,
        "promotion_max_failure_rate": maximum_failure_rate,
        "promotion_observed_failure_rate": failure_rate,
        "promotion_eligible": promotion_eligible,
        "promotion_blockers": blockers,
        "unexplained_oom_task_count": len(unexplained_oom),
        "unexplained_oom_task_ids": unexplained_oom,
        "recommended_walltime_seconds": recommended_walltime,
        "recommended_walltime": format_duration(recommended_walltime),
        "max_walltime_seconds": int(resource["max_walltime_seconds"]),
        "walltime_status": walltime_status,
        "projected_completion_hours": projected_completion,
        "projection_assumption": "continuous full utilization at recommended concurrency",
    }


def _allocation_projection(
    *,
    allocation: Mapping[str, Any],
    architecture_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    projected = {
        str(row.get("architecture")): row.get("projected_remaining_gpu_hours")
        for row in architecture_rows
    }
    reserve = float(allocation["reserve_fraction"])
    rows: list[dict[str, Any]] = []
    for architecture, record in sorted(allocation["architectures"].items()):
        spendable = (
            float(record["total_hours"]) * (1.0 - reserve)
            - float(record["consumed_hours"])
            - float(record["other_committed_hours"])
        )
        remaining = _finite_nonnegative(projected.get(architecture))
        rows.append(
            {
                "architecture": architecture,
                "total_hours": float(record["total_hours"]),
                "consumed_hours": float(record["consumed_hours"]),
                "other_committed_hours": float(record["other_committed_hours"]),
                "reserved_hours": float(record["total_hours"]) * reserve,
                "spendable_hours": spendable,
                "projected_campaign_remaining_gpu_hours": remaining,
                "reserve_guard": (
                    "unknown"
                    if remaining is None
                    else "pass"
                    if remaining <= spendable
                    else "blocked"
                ),
            }
        )
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "group_kind",
        "site",
        "resource_profile",
        "task_count",
        "success_count",
        "failure_count",
        "success_rate",
        "failure_rate",
        "retry_queue_task_count",
        "tasks_with_failed_attempts_count",
        "failed_attempt_count",
        "observed_runtime_count",
        "missing_runtime_count",
        "observed_task_hours",
        "runtime_p50_seconds",
        "runtime_p95_seconds",
        "success_runtime_p95_seconds",
        "peak_ram_observation_count",
        "peak_ram_bytes",
        "peak_vram_observation_count",
        "peak_vram_bytes",
        "projection_success_sample_count",
        "projected_total_task_hours",
        "projected_remaining_task_hours",
        "observed_gpu_hours",
        "projected_total_gpu_hours",
        "projected_remaining_gpu_hours",
        "architecture",
        "accelerators_per_task",
        "recommended_concurrency",
        "promotion_eligible",
        "unexplained_oom_task_count",
        "recommended_walltime_seconds",
        "recommended_walltime",
        "walltime_status",
        "projected_completion_hours",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name) for name in fieldnames})
    write_text_atomic(path, stream.getvalue())


def _write_runs_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task_id",
        "site",
        "resource_profile",
        "architecture",
        "accelerators_per_task",
        "task_status",
        "attempt_id",
        "attempt_kind",
        "runtime_seconds",
        "runtime_source",
        "peak_ram_bytes",
        "peak_ram_source",
        "peak_vram_bytes",
        "peak_vram_source",
        "run_json_path",
        "measurement_error",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow({name: record.get(name) for name in fieldnames})
    write_text_atomic(path, stream.getvalue())


def generate_daily_report(
    manifest_path: Path,
    *,
    reconcile_path: Path,
    output_dir: Path,
    meta_path: Path | None = None,
    resource_catalog_path: Path | None = None,
    allocation_path: Path | None = None,
    explained_oom_task_ids: Iterable[str] = (),
) -> DailyUsageReport:
    meta, tasks = load_manifest(manifest_path, meta_path=meta_path, verify_digest=True)
    reconcile, states = _load_reconcile_states(
        reconcile_path,
        campaign_id=str(meta["campaign_id"]),
        manifest_sha256=str(meta["manifest_sha256"]),
        tasks=tasks,
    )
    if resource_catalog_path is None:
        candidate = manifest_path.parent / "profiles" / "resources.json"
        resource_catalog_path = candidate if candidate.is_file() else None
    resources = (
        load_resource_catalog(resource_catalog_path) if resource_catalog_path is not None else {}
    )
    explained_ooms = set(explained_oom_task_ids)
    unknown_explained = sorted(explained_ooms - {task.task_id for task in tasks})
    if unknown_explained:
        raise CampaignError(
            "E_CAMPAIGN_DAILY_OOM",
            f"unknown explained OOM task_id(s): {unknown_explained}",
        )
    records = _measurement_records(tasks, states)
    task_by_id = {task.task_id: task for task in tasks}
    for record in records:
        task = task_by_id[str(record["task_id"])]
        record["site"] = task.assigned_site
        record["resource_profile"] = task.resource_profile
        resource = resources.get((task.assigned_site, task.resource_profile))
        record["architecture"] = None if resource is None else resource["architecture"]
        record["accelerators_per_task"] = (
            None if resource is None else resource["accelerators_per_task"]
        )
        record["task_status"] = states[task.task_id]["status"]
    records.sort(
        key=lambda item: (
            str(item["site"]),
            str(item["resource_profile"]),
            str(item["task_id"]),
            str(item.get("attempt_id") or ""),
        )
    )
    records_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_task[str(record["task_id"])].append(record)

    by_pair: dict[tuple[str, str], list[CampaignTask]] = defaultdict(list)
    by_site: dict[str, list[CampaignTask]] = defaultdict(list)
    by_resource: dict[str, list[CampaignTask]] = defaultdict(list)
    by_architecture: dict[str, list[CampaignTask]] = defaultdict(list)
    for task in tasks:
        by_pair[(task.assigned_site, task.resource_profile)].append(task)
        by_site[task.assigned_site].append(task)
        by_resource[task.resource_profile].append(task)
        resource = resources.get((task.assigned_site, task.resource_profile))
        if resource is not None:
            by_architecture[str(resource["architecture"])].append(task)

    total = _annotated_summary(
        group_kind="total",
        site=None,
        resource=None,
        summary=_summarize(tasks, states, records_by_task, resources),
    )
    resource_rows = [
        _annotated_summary(
            group_kind="resource",
            site=None,
            resource=resource,
            summary=_summarize(group_tasks, states, records_by_task, resources),
        )
        for resource, group_tasks in sorted(by_resource.items())
    ]
    site_rows = [
        _annotated_summary(
            group_kind="site",
            site=site,
            resource=None,
            summary=_summarize(group_tasks, states, records_by_task, resources),
        )
        for site, group_tasks in sorted(by_site.items())
    ]
    pair_rows = [
        _annotated_summary(
            group_kind="resource_site",
            site=site,
            resource=resource,
            summary=_summarize(group_tasks, states, records_by_task, resources),
        )
        for (site, resource), group_tasks in sorted(by_pair.items())
    ]
    pair_tasks = {
        (site, resource): group_tasks for (site, resource), group_tasks in by_pair.items()
    }
    for row in pair_rows:
        key = (str(row["site"]), str(row["resource_profile"]))
        row.update(
            _resource_policy(
                tasks=pair_tasks[key],
                states=states,
                summary=row,
                resource=resources.get(key),
                explained_oom_task_ids=explained_ooms,
            )
        )
    architecture_rows = [
        {
            **_annotated_summary(
                group_kind="architecture",
                site=None,
                resource=None,
                summary=_summarize(group_tasks, states, records_by_task, resources),
            ),
            "architecture": architecture,
        }
        for architecture, group_tasks in sorted(by_architecture.items())
    ]
    summary_rows = [total, *resource_rows, *site_rows, *pair_rows, *architecture_rows]

    generated_at = datetime.now(UTC)
    missing_runtime = int(total["missing_runtime_count"])
    limitations = [
        "observed task/GPU hours exclude attempts without a measured runtime",
        "projected task-hours use the arithmetic mean of successful measured runtimes and "
        "exclude retry overhead",
        "completion projections assume continuous full utilization at the recommended concurrency",
    ]
    if missing_runtime:
        limitations.append(
            f"{missing_runtime} attempt(s) have no measured runtime and are excluded from hours"
        )
    if int(total["peak_ram_observation_count"]) == 0:
        limitations.append("no supported peak RAM field was present in run.json")
    if int(total["peak_vram_observation_count"]) == 0:
        limitations.append("no supported peak VRAM field was present in run.json")
    if resources == {}:
        limitations.append(
            "resource catalog is missing; GPU-hour accounting and resource policy are unavailable"
        )

    allocations = None
    allocation_rows: list[dict[str, Any]] = []
    if allocation_path is not None:
        allocation = load_allocation_snapshot(allocation_path)
        allocation_rows = _allocation_projection(
            allocation=allocation,
            architecture_rows=architecture_rows,
        )
        allocations = {
            "updated_at": allocation["updated_at"],
            "reserve_fraction": allocation["reserve_fraction"],
            "by_architecture": allocation_rows,
        }
    governance_blockers = [
        f"{row['site']}.{row['resource_profile']}: walltime exceeds cap"
        for row in pair_rows
        if row.get("walltime_status") == "exceeds_cap"
    ]
    governance_blockers.extend(
        f"{row['architecture']}: projected GPU-hours breach the 15% reserve"
        for row in allocation_rows
        if row["reserve_guard"] == "blocked"
    )

    payload = {
        "schema_version": 1,
        "campaign_id": meta["campaign_id"],
        "manifest_sha256": meta["manifest_sha256"],
        "reconcile_path": str(reconcile_path.resolve()),
        "reconcile_status": reconcile.get("status"),
        "reconciled_at": reconcile.get("reconciled_at"),
        "generated_at": generated_at.isoformat(),
        "report_date_utc": generated_at.date().isoformat(),
        "hours_unit": "task wall-clock hours and accelerator-hours",
        "percentile_method": "linear interpolation over all measured success/failure attempts",
        "projection_basis": "mean runtime of measured successful tasks",
        "total": total,
        "by_resource": resource_rows,
        "by_site": site_rows,
        "by_resource_site": pair_rows,
        "by_architecture": architecture_rows,
        "allocations": allocations,
        "governance_status": "blocked" if governance_blockers else "pass",
        "governance_blockers": governance_blockers,
        "limitations": limitations,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "daily-usage.json"
    summary_csv_path = output_dir / "daily-usage-summary.csv"
    runs_csv_path = output_dir / "daily-usage-runs.csv"
    atomic_write_json(json_path, payload)
    _write_summary_csv(summary_csv_path, summary_rows)
    _write_runs_csv(runs_csv_path, records)
    return DailyUsageReport(
        campaign_id=str(meta["campaign_id"]),
        task_count=len(tasks),
        status=str(reconcile.get("status")),
        json_path=str(json_path),
        summary_csv_path=str(summary_csv_path),
        runs_csv_path=str(runs_csv_path),
    )


__all__ = ["DailyUsageReport", "generate_daily_report"]
