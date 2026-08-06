from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bench.utils.io import atomic_write_json

from .aggregate import aggregate_successes
from .attempts import validate_attempt_record, validate_authorization_event
from .bundles import immutable_bundle
from .errors import CampaignError
from .executor import validate_result_directory
from .generate import _AtomicCampaignDirectory
from .manifest import load_manifest, sha256_file, write_manifest, write_text_atomic
from .models import CampaignTask, ReconcileReport

_RETRYABLE = {"missing", "failed", "stale", "authorization_expired"}
_CONTINUABLE = {"continuation_pending"}


def _runtime_bindings_path(report_path: Path) -> Path:
    bundle = report_path.resolve(strict=False).parent
    return bundle.parent / f".{bundle.name}.runtime-roots.json"


def _resolve_logical_reference(value: str, *, bundle: Path, roots: Sequence[Path]) -> str:
    if value == "bundle://":
        return str(bundle)
    if value.startswith("bundle://"):
        relative = value.removeprefix("bundle://")
        candidate = (bundle / relative).resolve(strict=False)
        if not candidate.is_relative_to(bundle):
            raise CampaignError(
                "E_CAMPAIGN_RECONCILE_BINDING_INVALID",
                "bundle reference escapes the reconciliation bundle",
            )
        return str(candidate)
    if not value.startswith("result://root-"):
        return value
    match = re.fullmatch(r"result://root-([0-9]{3})(?:/(.*))?", value)
    if match is None:
        raise CampaignError("E_CAMPAIGN_RECONCILE_BINDING_INVALID", "result reference is malformed")
    index = int(match.group(1))
    if index >= len(roots):
        raise CampaignError(
            "E_CAMPAIGN_RECONCILE_BINDING_INVALID", "result reference has no runtime binding"
        )
    root = roots[index]
    relative = match.group(2)
    candidate = root if relative is None else (root / relative).resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise CampaignError(
            "E_CAMPAIGN_RECONCILE_BINDING_INVALID", "result reference escapes its bound root"
        )
    return str(candidate)


def _materialize_logical_values(value: Any, *, bundle: Path, roots: Sequence[Path]) -> Any:
    if isinstance(value, str):
        return _resolve_logical_reference(value, bundle=bundle, roots=roots)
    if isinstance(value, list):
        return [_materialize_logical_values(item, bundle=bundle, roots=roots) for item in value]
    if isinstance(value, dict):
        return {
            key: _materialize_logical_values(item, bundle=bundle, roots=roots)
            for key, item in value.items()
        }
    return value


def materialize_reconcile_paths(report_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve portable evidence URIs using an authenticated local runtime binding.

    The binding is deliberately outside the immutable bundle: it is operational,
    machine-local state and is never part of the scientific evidence.
    """

    if "result://" not in json.dumps(payload, sort_keys=True):
        return payload
    resolved_report = report_path.resolve(strict=True)
    binding_path = _runtime_bindings_path(resolved_report)
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(
            "E_CAMPAIGN_RECONCILE_BINDING_REQUIRED",
            f"portable reconciliation needs its local runtime binding: {binding_path}",
        ) from exc
    try:
        reconcile_sha256 = sha256_file(resolved_report)
        bundle_manifest_sha256 = sha256_file(resolved_report.parent / "BUNDLE.json")
    except OSError as exc:
        raise CampaignError(
            "E_CAMPAIGN_RECONCILE_BINDING_INVALID",
            "portable reconciliation bundle is incomplete",
        ) from exc
    if (
        not isinstance(binding, dict)
        or binding.get("schema_version") != 1
        or binding.get("reconcile_sha256") != reconcile_sha256
        or binding.get("bundle_manifest_sha256") != bundle_manifest_sha256
    ):
        raise CampaignError(
            "E_CAMPAIGN_RECONCILE_BINDING_INVALID",
            "reconciliation runtime binding does not authenticate this bundle",
        )
    raw_roots = binding.get("result_roots")
    if not isinstance(raw_roots, list) or any(
        not isinstance(root, str) or not Path(root).is_absolute() for root in raw_roots
    ):
        raise CampaignError(
            "E_CAMPAIGN_RECONCILE_BINDING_INVALID", "runtime result-root bindings are invalid"
        )
    roots = [Path(root).resolve(strict=False) for root in raw_roots]
    if payload.get("result_roots") != [f"result://root-{index:03d}" for index in range(len(roots))]:
        raise CampaignError(
            "E_CAMPAIGN_RECONCILE_BINDING_INVALID",
            "sealed result-root identities differ from the runtime binding",
        )
    materialized = _materialize_logical_values(
        payload,
        bundle=resolved_report.parent,
        roots=roots,
    )
    assert isinstance(materialized, dict)
    return materialized


def _lock_age(lock_dir: Path, *, now: datetime) -> timedelta:
    owner_path = lock_dir / "owner.json"
    if owner_path.is_file():
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(str(owner["created_at"]))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            return now - created.astimezone(UTC)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    modified = datetime.fromtimestamp(lock_dir.stat().st_mtime, tz=UTC)
    return now - modified


def _attempt_dirs(root: Path, task: CampaignTask) -> list[Path]:
    parent = root / "attempts" / task.task_id[:2] / task.task_id
    return (
        sorted((path for path in parent.iterdir() if path.is_dir()), reverse=True)
        if parent.is_dir()
        else []
    )


def _attempt_records(
    root: Path, task: CampaignTask
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for path in _attempt_dirs(root, task):
        attempt_path = path / "attempt.json"
        try:
            payload = json.loads(attempt_path.read_text(encoding="utf-8"))
            record = validate_attempt_record(payload, task=task, directory_name=path.name)
        except (OSError, json.JSONDecodeError, CampaignError) as exc:
            invalid.append({"path": str(attempt_path), "error": str(exc)})
            continue
        normalized = dict(record.payload)
        normalized["_finished_at_utc"] = record.finished_at_utc
        normalized["paths"] = [str(path)]
        records.append(normalized)
    return (
        sorted(records, key=lambda item: item["_finished_at_utc"], reverse=True),
        invalid,
    )


def _authorization_event_dirs(root: Path, task: CampaignTask) -> list[Path]:
    parent = root / "events" / task.task_id[:2] / task.task_id
    return (
        sorted((path for path in parent.iterdir() if path.is_dir()), reverse=True)
        if parent.is_dir()
        else []
    )


def _authorization_event_records(
    root: Path, task: CampaignTask
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for path in _authorization_event_dirs(root, task):
        event_path = path / "event.json"
        try:
            payload = json.loads(event_path.read_text(encoding="utf-8"))
            record = validate_authorization_event(
                payload,
                task=task,
                directory_name=path.name,
            )
        except (OSError, json.JSONDecodeError, CampaignError) as exc:
            invalid.append({"path": str(event_path), "error": str(exc)})
            continue
        normalized = dict(record.payload)
        normalized["_observed_at_utc"] = record.observed_at_utc
        normalized["paths"] = [str(path)]
        records.append(normalized)
    return (
        sorted(records, key=lambda item: item["_observed_at_utc"], reverse=True),
        invalid,
    )


def _classify_task(
    task: CampaignTask,
    *,
    result_roots: list[Path],
    stale_after: timedelta,
    now: datetime,
) -> dict[str, Any]:
    valid: list[tuple[Path, Path, str]] = []
    corrupt: list[dict[str, str]] = []
    locks: list[dict[str, Any]] = []
    raw_attempt_records: list[dict[str, Any]] = []
    raw_authorization_events: list[dict[str, Any]] = []
    invalid_attempts: list[dict[str, str]] = []
    for root in result_roots:
        result_dir = root / task.output_relpath
        if result_dir.exists():
            try:
                run_json, _, digest = validate_result_directory(result_dir, task)
                valid.append((result_dir, run_json, digest))
            except CampaignError as exc:
                corrupt.append(
                    {"root": str(root), "result_dir": str(result_dir), "error": str(exc)}
                )
        lock_dir = root / "locks" / f"{task.task_id}.lock"
        if lock_dir.is_dir():
            age = _lock_age(lock_dir, now=now)
            locks.append(
                {
                    "root": str(root),
                    "lock_dir": str(lock_dir),
                    "age_seconds": max(0.0, age.total_seconds()),
                    "stale": age > stale_after,
                }
            )
        root_attempts, root_invalid = _attempt_records(root, task)
        raw_attempt_records.extend(root_attempts)
        invalid_attempts.extend(root_invalid)
        root_events, event_invalid = _authorization_event_records(root, task)
        raw_authorization_events.extend(root_events)
        invalid_attempts.extend(event_invalid)

    attempts_by_id: dict[str, dict[str, Any]] = {}
    conflicting_attempts: list[dict[str, str]] = []
    for record in raw_attempt_records:
        attempt_id = str(record["attempt_id"])
        existing = attempts_by_id.get(attempt_id)
        if existing is None:
            attempts_by_id[attempt_id] = record
            continue
        if existing["record_sha256"] != record["record_sha256"]:
            conflicting_attempts.append(
                {
                    "attempt_id": attempt_id,
                    "error": "mirrored attempt records have different authenticated digests",
                }
            )
            continue
        existing["paths"] = sorted({*existing["paths"], *record["paths"]})
    attempt_records = sorted(
        attempts_by_id.values(),
        key=lambda item: item["_finished_at_utc"],
        reverse=True,
    )
    events_by_id: dict[str, dict[str, Any]] = {}
    conflicting_events: list[dict[str, str]] = []
    for record in raw_authorization_events:
        event_id = str(record["event_id"])
        existing = events_by_id.get(event_id)
        if existing is None:
            events_by_id[event_id] = record
            continue
        if existing["record_sha256"] != record["record_sha256"]:
            conflicting_events.append(
                {
                    "event_id": event_id,
                    "error": "mirrored authorization events have different authenticated digests",
                }
            )
            continue
        existing["paths"] = sorted({*existing["paths"], *record["paths"]})
    authorization_events = sorted(
        events_by_id.values(),
        key=lambda item: item["_observed_at_utc"],
        reverse=True,
    )
    continuation_attempts = [
        item
        for item in attempt_records
        if item.get("status") == "continuation"
        and item.get("event_class") == "planned_continuation"
    ]
    error_attempts = [item for item in attempt_records if item not in continuation_attempts]
    resource_attempts = [
        item
        for item in error_attempts
        if item.get("failure_class") in {"resource_oom", "resource_timeout"}
    ]
    deterministic_attempts = [
        item for item in error_attempts if item.get("failure_class") == "deterministic"
    ]
    latest_is_continuation = bool(attempt_records and attempt_records[0] in continuation_attempts)
    authorization_is_latest = bool(
        authorization_events
        and (
            not attempt_records
            or authorization_events[0]["_observed_at_utc"] > attempt_records[0]["_finished_at_utc"]
        )
    )

    if corrupt or invalid_attempts or conflicting_attempts or conflicting_events:
        status = "corrupt"
    elif len({item[2] for item in valid}) > 1:
        status = "conflict"
    elif valid:
        status = "success"
    elif deterministic_attempts:
        status = "blocked"
    elif resource_attempts:
        status = "resource_blocked"
    elif locks and any(not item["stale"] for item in locks):
        status = "running"
    elif latest_is_continuation and not authorization_is_latest:
        status = "continuation_pending"
    elif authorization_is_latest:
        status = "authorization_expired"
    elif locks:
        status = "stale"
    elif error_attempts:
        failure_classes = {
            str(item.get("failure_class", "deterministic")) for item in error_attempts
        }
        if failure_classes == {"infrastructure"} and len(error_attempts) <= 3:
            status = "failed"
        else:
            status = "blocked"
    else:
        status = "missing"

    return {
        "task_index": task.task_index,
        "task_id": task.task_id,
        "method_id": task.method_id,
        "dataset_id": task.dataset_id,
        "resource_profile": task.resource_profile,
        "assigned_site": task.assigned_site,
        "status": status,
        "result_dirs": [str(item[0]) for item in valid],
        "run_json_paths": [str(item[1]) for item in valid],
        "run_json_sha256": sorted({item[2] for item in valid}),
        "corrupt": corrupt,
        "invalid_attempts": invalid_attempts,
        "conflicting_attempts": conflicting_attempts,
        "conflicting_authorization_events": conflicting_events,
        "locks": locks,
        "attempts": [path for item in attempt_records for path in item["paths"]],
        "authorization_events": [path for item in authorization_events for path in item["paths"]],
        "authorization_event_count": len(authorization_events),
        "attempt_count": len(attempt_records),
        "error_attempt_count": len(error_attempts),
        "continuation_attempt_count": len(continuation_attempts),
        "latest_failure_class": (
            str(
                (deterministic_attempts or resource_attempts or error_attempts)[0].get(
                    "failure_class"
                )
            )
            if error_attempts
            else None
        ),
    }


def _retry_tasks(*, tasks: list[CampaignTask], states: list[dict[str, Any]]) -> list[CampaignTask]:
    retry_indices = {
        int(state["task_index"]) for state in states if str(state["status"]) in _RETRYABLE
    }
    return [task for task in tasks if task.task_index in retry_indices]


def _continuation_tasks(
    *, tasks: list[CampaignTask], states: list[dict[str, Any]]
) -> list[CampaignTask]:
    indices = {int(state["task_index"]) for state in states if str(state["status"]) in _CONTINUABLE}
    return [task for task in tasks if task.task_index in indices]


def _write_retry_outputs(
    *,
    output_dir: Path,
    retry_tasks: list[CampaignTask],
    manifest_meta: dict[str, Any],
) -> None:
    write_manifest(
        retry_tasks,
        output_dir=output_dir,
        campaign_id=str(manifest_meta["campaign_id"]),
        spec_sha256=str(manifest_meta["spec_sha256"]),
        expected_git_sha=str(manifest_meta["expected_git_sha"]),
        expected_git_diff_sha256=manifest_meta.get("expected_git_diff_sha256"),
        environment_lock_sha256=str(manifest_meta["environment_lock_sha256"]),
        manifest_filename="retry.jsonl",
        meta_filename="retry.meta.json",
        profile_dirname="retry",
        source_manifest_sha256=str(manifest_meta["manifest_sha256"]),
    )


def _write_continuation_outputs(
    *,
    output_dir: Path,
    continuation_tasks: list[CampaignTask],
    manifest_meta: dict[str, Any],
) -> None:
    write_manifest(
        continuation_tasks,
        output_dir=output_dir,
        campaign_id=str(manifest_meta["campaign_id"]),
        spec_sha256=str(manifest_meta["spec_sha256"]),
        expected_git_sha=str(manifest_meta["expected_git_sha"]),
        expected_git_diff_sha256=manifest_meta.get("expected_git_diff_sha256"),
        environment_lock_sha256=str(manifest_meta["environment_lock_sha256"]),
        manifest_filename="continuation.jsonl",
        meta_filename="continuation.meta.json",
        profile_dirname="continuation",
        source_manifest_sha256=str(manifest_meta["manifest_sha256"]),
    )


def _write_retry_campaign(
    *,
    output_dir: Path,
    retry_tasks: list[CampaignTask],
    manifest_meta: dict[str, Any],
    destination_name: str = "retry-campaign",
) -> Path:
    destination = output_dir / destination_name
    with _AtomicCampaignDirectory(destination) as staging_dir:
        write_manifest(
            retry_tasks,
            output_dir=staging_dir,
            campaign_id=str(manifest_meta["campaign_id"]),
            spec_sha256=str(manifest_meta["spec_sha256"]),
            expected_git_sha=str(manifest_meta["expected_git_sha"]),
            expected_git_diff_sha256=manifest_meta.get("expected_git_diff_sha256"),
            environment_lock_sha256=str(manifest_meta["environment_lock_sha256"]),
            source_manifest_sha256=str(manifest_meta["manifest_sha256"]),
        )
    return destination.resolve()


def _write_reprofile_requirements(
    *, output_dir: Path, tasks: list[CampaignTask], states: list[dict[str, Any]]
) -> int:
    task_by_id = {task.task_id: task for task in tasks}
    tasks_by_cell: dict[tuple[str, str], list[CampaignTask]] = {}
    for task in tasks:
        cell_key = (task.track, task.protocol_id or task.config_path)
        tasks_by_cell.setdefault(cell_key, []).append(task)
    blocked_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for state in states:
        if state.get("status") != "resource_blocked":
            continue
        task = task_by_id[str(state["task_id"])]
        cell_key = (task.track, task.protocol_id or task.config_path)
        blocked_by_cell.setdefault(cell_key, []).append(state)

    records: list[dict[str, Any]] = []
    for cell_key, blocked_states in sorted(blocked_by_cell.items()):
        cell_tasks = sorted(tasks_by_cell[cell_key], key=lambda item: item.seed)
        triggering_ids = sorted(str(state["task_id"]) for state in blocked_states)
        failure_classes = sorted(
            {str(state.get("latest_failure_class")) for state in blocked_states}
        )
        first = cell_tasks[0]
        records.append(
            {
                "schema_version": 1,
                "track": first.track,
                "cell_id": cell_key[1],
                "method_id": first.method_id,
                "dataset_id": first.dataset_id,
                "task_id": triggering_ids[0],
                "task_index": task_by_id[triggering_ids[0]].task_index,
                "triggering_task_ids": triggering_ids,
                "task_ids": [task.task_id for task in cell_tasks],
                "task_indices": [task.task_index for task in cell_tasks],
                "seeds": [task.seed for task in cell_tasks],
                "failure_class": (
                    failure_classes[0] if len(failure_classes) == 1 else "mixed_resource_failure"
                ),
                "failure_classes": failure_classes,
                "current_assignments": sorted(
                    {f"{task.assigned_site}.{task.resource_profile}" for task in cell_tasks}
                ),
                "action": (
                    "generate new immutable manifest rows with one reviewed resource profile "
                    "for every seed in this cell, then rerun the complete cell"
                ),
                "automatically_retryable": False,
            }
        )
    write_text_atomic(
        output_dir / "reprofile-required.jsonl",
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
        ),
    )
    return len(records)


def _reconcile_campaign_into(
    manifest_path: Path,
    *,
    result_roots: list[Path],
    output_dir: Path,
    stale_after: timedelta = timedelta(hours=120),
    emit_retry: bool = True,
    meta_path: Path | None = None,
) -> ReconcileReport:
    if not result_roots:
        raise CampaignError("E_CAMPAIGN_RECONCILE", "at least one result root is required")
    meta, tasks = load_manifest(manifest_path, meta_path=meta_path, verify_digest=True)
    roots = [root.resolve() for root in result_roots]
    now = datetime.now(UTC)
    states = [
        _classify_task(
            task,
            result_roots=roots,
            stale_after=stale_after,
            now=now,
        )
        for task in tasks
    ]
    counts = dict(sorted(Counter(str(state["status"]) for state in states).items()))
    if counts.get("corrupt", 0) or counts.get("conflict", 0) or counts.get("duplicate", 0):
        overall = "invalid"
    elif counts.get("success", 0) == len(tasks):
        overall = "complete"
    else:
        overall = "incomplete"

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregation = aggregate_successes(tasks=tasks, states=states, output_dir=output_dir)
    if overall == "complete" and int(aggregation["incomplete_cells"]) > 0:
        overall = "incomplete"
    retry_tasks = _retry_tasks(tasks=tasks, states=states) if emit_retry else []
    retry_count = len(retry_tasks)
    retry_campaign_path: Path | None = None
    if retry_tasks:
        retry_campaign_path = _write_retry_campaign(
            output_dir=output_dir,
            retry_tasks=retry_tasks,
            manifest_meta=meta,
        )
        _write_retry_outputs(
            output_dir=output_dir,
            retry_tasks=retry_tasks,
            manifest_meta=meta,
        )
    continuation_tasks = _continuation_tasks(tasks=tasks, states=states)
    continuation_count = len(continuation_tasks)
    continuation_campaign_path: Path | None = None
    if continuation_tasks:
        continuation_campaign_path = _write_retry_campaign(
            output_dir=output_dir,
            retry_tasks=continuation_tasks,
            manifest_meta=meta,
            destination_name="continuation-campaign",
        )
        _write_continuation_outputs(
            output_dir=output_dir,
            continuation_tasks=continuation_tasks,
            manifest_meta=meta,
        )
    reprofile_required_count = _write_reprofile_requirements(
        output_dir=output_dir, tasks=tasks, states=states
    )
    successful_run_paths = [
        path
        for state in states
        if state["status"] == "success"
        for path in state["run_json_paths"][:1]
    ]
    write_text_atomic(
        output_dir / "successful-run-json.txt",
        "".join(f"{path}\n" for path in successful_run_paths),
    )
    summary_lines = [
        "task_index\ttask_id\tstatus\tmethod_id\tdataset_id\tresource_profile\tassigned_site\n"
    ]
    for state in states:
        summary_lines.append(
            "\t".join(
                str(state[key])
                for key in (
                    "task_index",
                    "task_id",
                    "status",
                    "method_id",
                    "dataset_id",
                    "resource_profile",
                    "assigned_site",
                )
            )
            + "\n"
        )
    write_text_atomic(output_dir / "summary.tsv", "".join(summary_lines))
    report_payload = {
        "schema_version": 1,
        "campaign_id": meta["campaign_id"],
        "manifest_sha256": meta["manifest_sha256"],
        "reconciled_at": now.isoformat(),
        "result_roots": [str(root) for root in roots],
        "stale_after_seconds": stale_after.total_seconds(),
        "status": overall,
        "task_count": len(tasks),
        "counts": counts,
        "retry_count": retry_count,
        "retry_campaign_path": (
            str(retry_campaign_path) if retry_campaign_path is not None else None
        ),
        "continuation_count": continuation_count,
        "continuation_campaign_path": (
            str(continuation_campaign_path) if continuation_campaign_path is not None else None
        ),
        "reprofile_required_count": reprofile_required_count,
        "aggregation": aggregation,
        "tasks": states,
    }
    report_path = output_dir / "reconcile.json"
    atomic_write_json(report_path, report_payload)
    return ReconcileReport(
        campaign_id=str(meta["campaign_id"]),
        status=overall,
        task_count=len(tasks),
        counts=counts,
        report_path=str(report_path),
        retry_count=retry_count,
        retry_campaign_path=(str(retry_campaign_path) if retry_campaign_path is not None else None),
        continuation_count=continuation_count,
        continuation_campaign_path=(
            str(continuation_campaign_path) if continuation_campaign_path is not None else None
        ),
    )


def _portable_report_value(
    value: Any,
    *,
    staging: Path,
    result_roots: Sequence[Path],
) -> Any:
    if isinstance(value, str):
        replacements = [(str(staging), "bundle://")]
        replacements.extend(
            (str(root), f"result://root-{index:03d}") for index, root in enumerate(result_roots)
        )
        portable = value
        # Replace longer roots first in case one result root contains another.
        for prefix, logical in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
            logical_prefix = logical if logical.endswith("://") else f"{logical}/"
            portable = portable.replace(f"{prefix}/", logical_prefix)
            portable = portable.replace(prefix, logical)
        return portable
    if isinstance(value, list):
        return [
            _portable_report_value(item, staging=staging, result_roots=result_roots)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _portable_report_value(item, staging=staging, result_roots=result_roots)
            for key, item in value.items()
        }
    return value


def _assert_portable_report_value(value: Any) -> None:
    if isinstance(value, str):
        if value.startswith("/"):
            raise CampaignError(
                "E_CAMPAIGN_BUNDLE_INVALID",
                "sealed reconciliation metadata contains an absolute path",
            )
        return
    if isinstance(value, list):
        for item in value:
            _assert_portable_report_value(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_portable_report_value(item)


def reconcile_campaign(
    manifest_path: Path,
    *,
    result_roots: list[Path],
    output_dir: Path,
    stale_after: timedelta = timedelta(hours=120),
    emit_retry: bool = True,
    meta_path: Path | None = None,
) -> ReconcileReport:
    final = Path(output_dir).resolve(strict=False)
    roots = [Path(root).resolve(strict=False) for root in result_roots]
    runtime_bindings_path = _runtime_bindings_path(final / "reconcile.json")
    if runtime_bindings_path.exists() or runtime_bindings_path.is_symlink():
        raise CampaignError(
            "E_CAMPAIGN_RECONCILE_BINDING_EXISTS",
            f"immutable runtime binding already exists: {runtime_bindings_path}",
        )
    with immutable_bundle(final, kind="modssc.campaign.reconcile.v2") as staging:
        report = _reconcile_campaign_into(
            manifest_path,
            result_roots=result_roots,
            output_dir=staging,
            stale_after=stale_after,
            emit_retry=emit_retry,
            meta_path=meta_path,
        )
        report_path = staging / "reconcile.json"
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        portable_payload = _portable_report_value(payload, staging=staging, result_roots=roots)
        _assert_portable_report_value(portable_payload)
        atomic_write_json(report_path, portable_payload)
        successful_paths = staging / "successful-run-json.txt"
        portable_lines = [
            str(_portable_report_value(line, staging=staging, result_roots=roots))
            for line in successful_paths.read_text(encoding="utf-8").splitlines()
            if line
        ]
        _assert_portable_report_value(portable_lines)
        write_text_atomic(
            successful_paths,
            "".join(f"{line}\n" for line in portable_lines),
        )
    atomic_write_json(
        runtime_bindings_path,
        {
            "schema_version": 1,
            "reconcile_sha256": sha256_file(final / "reconcile.json"),
            "bundle_manifest_sha256": sha256_file(final / "BUNDLE.json"),
            "result_roots": [str(root) for root in roots],
        },
    )
    return replace(
        report,
        report_path=str(final / "reconcile.json"),
        retry_campaign_path=(
            str(final / "retry-campaign") if report.retry_campaign_path is not None else None
        ),
        continuation_campaign_path=(
            str(final / "continuation-campaign")
            if report.continuation_campaign_path is not None
            else None
        ),
    )
