from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.campaign.attempts import seal_attempt_record, validate_attempt_record
from bench.campaign.errors import CampaignError
from bench.campaign.manifest import load_manifest, select_task
from bench.campaign.models import CampaignTask
from bench.utils.hashing import hash_any
from bench.utils.io import atomic_write_json

from .execution_context import execution_metadata

_RESOURCE_FAILURES = frozenset({"resource_oom", "resource_timeout"})


@dataclass(frozen=True)
class SchedulerFailureResult:
    task_id: str
    failure_class: str
    attempt_dir: str
    scheduler_event_id: str
    skipped: bool
    orphan_lock_action: str
    orphan_lock_quarantine: str | None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _active_scheduler_metadata() -> dict[str, str]:
    return {f"slurm_{key}": value for key, value in execution_metadata().items()}


def _normalise_scheduler_metadata(raw: Mapping[str, Any]) -> dict[str, str]:
    scheduler: dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            scheduler[str(key).lower()] = text
    return scheduler


def _scheduler_event_identity(*, scheduler: Mapping[str, str], site_id: str) -> dict[str, str]:
    array_job_id = scheduler.get("slurm_array_job_id")
    array_task_id = scheduler.get("slurm_array_task_id")
    job_id = scheduler.get("slurm_job_id")
    identity = {
        "site_id": site_id,
        "cluster_name": scheduler.get("slurm_cluster_name", "unknown"),
    }
    if array_job_id is not None or array_task_id is not None:
        if array_job_id is None or array_task_id is None:
            raise CampaignError(
                "E_CAMPAIGN_SCHEDULER_IDENTITY",
                "Slurm array identity requires both SLURM_ARRAY_JOB_ID and SLURM_ARRAY_TASK_ID",
            )
        identity.update(
            {
                "kind": "slurm_array_element",
                "array_job_id": array_job_id,
                "array_task_id": array_task_id,
            }
        )
        return identity
    if job_id is None:
        raise CampaignError(
            "E_CAMPAIGN_SCHEDULER_IDENTITY",
            "SLURM_JOB_ID or a complete Slurm array identity is required",
        )
    identity.update({"kind": "slurm_job", "job_id": job_id})
    return identity


def _load_attempt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(
            "E_CAMPAIGN_SCHEDULER_FAILURE_CONFLICT",
            f"cannot validate an existing scheduler attempt: {path}",
        ) from exc
    if not isinstance(payload, dict):
        raise CampaignError(
            "E_CAMPAIGN_SCHEDULER_FAILURE_CONFLICT",
            f"existing scheduler attempt is not an object: {path}",
        )
    return payload


def _validate_existing_attempt(
    path: Path,
    *,
    task: CampaignTask,
    attempt_id: str,
    scheduler_event_id: str,
    failure_class: str,
) -> None:
    payload = _load_attempt(path / "attempt.json")
    validate_attempt_record(payload, task=task, directory_name=attempt_id)
    expected = {
        "task_id": task.task_id,
        "row_sha256": task.row_sha256,
        "attempt_id": attempt_id,
        "status": "failed",
        "failure_class": failure_class,
        "retryable": False,
        "resource_change_required": True,
        "external_event_id": scheduler_event_id,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise CampaignError(
            "E_CAMPAIGN_SCHEDULER_FAILURE_CONFLICT",
            f"scheduler event {scheduler_event_id} already has incompatible metadata",
        )


def _publish_scheduler_attempt(
    *,
    result_root: Path,
    task: CampaignTask,
    attempt_id: str,
    scheduler_event_id: str,
    scheduler_identity: Mapping[str, str],
    scheduler: Mapping[str, str],
    failure_class: str,
    scheduler_state: str | None,
    exit_code: int | None,
    site_id: str,
) -> tuple[Path, bool]:
    parent = result_root / "attempts" / task.task_id[:2] / task.task_id
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / attempt_id
    guard_path = parent / f".{attempt_id}.guard"
    guard_fd = os.open(guard_path, os.O_CREAT | os.O_RDWR, 0o600)
    staging: Path | None = None
    try:
        fcntl.flock(guard_fd, fcntl.LOCK_EX)
        if target.exists():
            if not target.is_dir() or target.is_symlink():
                raise CampaignError(
                    "E_CAMPAIGN_SCHEDULER_FAILURE_CONFLICT",
                    f"scheduler attempt target is unsafe: {target}",
                )
            _validate_existing_attempt(
                target,
                task=task,
                attempt_id=attempt_id,
                scheduler_event_id=scheduler_event_id,
                failure_class=failure_class,
            )
            return target, True

        staging = (
            result_root / ".staging" / f"scheduler-{task.task_id}.{attempt_id}.{uuid.uuid4().hex}"
        )
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=False, exist_ok=False)
        state = scheduler_state.strip() if scheduler_state is not None else None
        atomic_write_json(
            staging / "attempt.json",
            seal_attempt_record(
                {
                    "task_id": task.task_id,
                    "row_sha256": task.row_sha256,
                    "attempt_id": attempt_id,
                    "status": "failed",
                    "site_id": site_id,
                    "finished_at": _utc_now(),
                    "error_type": "SlurmSchedulerFailure",
                    "error": state or failure_class,
                    "traceback": "",
                    "failure_class": failure_class,
                    "retryable": False,
                    "resource_change_required": True,
                    "external_event_id": scheduler_event_id,
                    "scheduler_identity": dict(scheduler_identity),
                    "scheduler_state": state,
                    "exit_code": exit_code,
                    "scheduler": dict(scheduler),
                }
            ),
        )
        os.replace(staging, target)
        staging = None
        return target, False
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        try:
            fcntl.flock(guard_fd, fcntl.LOCK_UN)
        finally:
            os.close(guard_fd)


def _quarantine_orphaned_lock(
    *, result_root: Path, task: CampaignTask, scheduler_event_id: str
) -> tuple[str, Path | None]:
    lock_dir = result_root / "locks" / f"{task.task_id}.lock"
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    guard_path = lock_dir.parent / f".{lock_dir.name}.guard"
    guard_fd = os.open(guard_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(guard_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            return "guard_busy", None
        if not lock_dir.exists():
            return "absent", None
        if not lock_dir.is_dir() or lock_dir.is_symlink():
            return "unsafe", None
        owner_path = lock_dir / "owner.json"
        if owner_path.is_file():
            try:
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                owner = None
            if isinstance(owner, dict) and owner.get("task_id") not in {None, task.task_id}:
                return "owner_mismatch", None
        quarantine = (
            result_root
            / "orphaned-locks"
            / f"{lock_dir.name}.{scheduler_event_id}.{uuid.uuid4().hex}"
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        os.replace(lock_dir, quarantine)
        return "quarantined", quarantine
    finally:
        try:
            fcntl.flock(guard_fd, fcntl.LOCK_UN)
        finally:
            os.close(guard_fd)


def record_scheduler_failure(
    manifest_path: Path,
    *,
    meta_path: Path,
    result_root: Path,
    site_id: str,
    index: int,
    failure_class: str,
    scheduler_state: str | None = None,
    exit_code: int | None = None,
    scheduler_metadata: Mapping[str, Any] | None = None,
) -> SchedulerFailureResult:
    """Persist an idempotent Slurm resource failure for one manifest row."""

    if failure_class not in _RESOURCE_FAILURES:
        raise CampaignError(
            "E_CAMPAIGN_SCHEDULER_FAILURE_CLASS",
            f"unsupported scheduler failure class: {failure_class}",
        )
    _, tasks = load_manifest(manifest_path, meta_path=meta_path, verify_digest=True)
    task = select_task(tasks, index=index, task_id=None)
    if task.assigned_site != "any" and task.assigned_site != site_id:
        raise CampaignError(
            "E_CAMPAIGN_SITE_MISMATCH",
            f"task is assigned to {task.assigned_site}, not {site_id}",
        )
    scheduler = _normalise_scheduler_metadata(
        scheduler_metadata if scheduler_metadata is not None else _active_scheduler_metadata()
    )
    scheduler_identity = _scheduler_event_identity(scheduler=scheduler, site_id=site_id)
    scheduler_event_id = hash_any(scheduler_identity)
    attempt_id = f"slurm-{scheduler_event_id[:32]}"
    attempt_dir, skipped = _publish_scheduler_attempt(
        result_root=result_root,
        task=task,
        attempt_id=attempt_id,
        scheduler_event_id=scheduler_event_id,
        scheduler_identity=scheduler_identity,
        scheduler=scheduler,
        failure_class=failure_class,
        scheduler_state=scheduler_state,
        exit_code=exit_code,
        site_id=site_id,
    )
    lock_action, quarantine = _quarantine_orphaned_lock(
        result_root=result_root,
        task=task,
        scheduler_event_id=scheduler_event_id,
    )
    return SchedulerFailureResult(
        task_id=task.task_id,
        failure_class=failure_class,
        attempt_dir=str(attempt_dir),
        scheduler_event_id=scheduler_event_id,
        skipped=skipped,
        orphan_lock_action=lock_action,
        orphan_lock_quarantine=str(quarantine) if quarantine is not None else None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.hpc.scheduler_failure",
        description="Record an idempotent Slurm resource failure for a campaign task",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--site-id", required=True)
    parser.add_argument(
        "--failure-class",
        choices=tuple(sorted(_RESOURCE_FAILURES)),
        required=True,
    )
    parser.add_argument("--scheduler-state", default=None)
    parser.add_argument("--exit-code", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = record_scheduler_failure(
            args.manifest,
            meta_path=args.meta,
            result_root=args.result_root,
            site_id=args.site_id,
            index=args.index,
            failure_class=args.failure_class,
            scheduler_state=args.scheduler_state,
            exit_code=args.exit_code,
        )
    except CampaignError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
