from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.utils.io import atomic_write_json

from .errors import CampaignError
from .manifest import sha256_file
from .models import CampaignTask

CHECKPOINT_SCHEMA_VERSION = 1
PLANNED_CONTINUATION_EXIT_CODE = 85


@dataclass(frozen=True)
class RestoredCheckpoint:
    workspace: Path
    resumed: bool
    payload_sha256: str | None
    checkpoint_manifest_sha256: str | None


@dataclass(frozen=True)
class PublishedCheckpoint:
    task_dir: Path
    snapshot_dir: Path
    continue_path: Path
    payload_sha256: str
    checkpoint_manifest_sha256: str


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _contained(root: Path, relative: Path, *, label: str) -> Path:
    resolved_root = root.expanduser().resolve(strict=False)
    candidate = (resolved_root / relative).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_INVALID",
            f"{label} escapes checkpoint root",
        ) from exc
    return candidate


def checkpoint_identity(task: CampaignTask) -> dict[str, Any]:
    """Return the immutable identity authenticated by every task checkpoint."""

    partition = {
        "split_request_sha256": task.split_request_sha256,
        "expected_split_fingerprint": task.expected_split_fingerprint,
        "split_seed": task.split_seed,
        "sampling_component_seeds": task.sampling_component_seeds,
        "partition_selection": task.partition_selection,
    }
    identity = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task_id": task.task_id,
        "row_sha256": task.row_sha256,
        "commit_sha": task.expected_git_sha,
        "git_diff_sha256": task.expected_git_diff_sha256,
        "environment_lock_sha256": task.environment_lock_sha256,
        "method_profile": task.method_profile,
        "resource_profile": task.resource_profile,
        "assigned_site": task.assigned_site,
        "partition_sha256": _canonical_sha256(partition),
        "partition": partition,
    }
    return identity | {"identity_sha256": _canonical_sha256(identity)}


def checkpoint_task_dir(checkpoint_root: Path, task: CampaignTask) -> Path:
    return _contained(
        Path(checkpoint_root),
        Path("tasks") / task.task_id[:2] / task.task_id,
        label="checkpoint task directory",
    )


def _payload_file_records(payload_dir: Path) -> list[dict[str, Any]]:
    if not payload_dir.is_dir() or payload_dir.is_symlink():
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_MISSING",
            f"checkpoint payload directory is missing or unsafe: {payload_dir}",
        )
    records: list[dict[str, Any]] = []
    for path in sorted(payload_dir.rglob("*")):
        if path.is_symlink():
            raise CampaignError(
                "E_CAMPAIGN_CHECKPOINT_INVALID",
                f"checkpoint payload contains a symlink: {path}",
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise CampaignError(
                "E_CAMPAIGN_CHECKPOINT_INVALID",
                f"checkpoint payload contains a non-regular file: {path}",
            )
        relative = path.relative_to(payload_dir).as_posix()
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_MISSING",
            "planned continuation requires a non-empty checkpoint payload",
        )
    return records


def _read_mapping(path: Path, *, code: str = "E_CAMPAIGN_CHECKPOINT_INVALID") -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignError(code, f"cannot read checkpoint metadata: {path}") from exc
    if not isinstance(raw, dict):
        raise CampaignError(code, f"checkpoint metadata must be an object: {path}")
    return raw


def _validate_snapshot(
    snapshot_dir: Path,
    *,
    task: CampaignTask,
    expected_payload_sha256: str,
    expected_manifest_sha256: str,
) -> None:
    manifest_path = snapshot_dir / "checkpoint.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha256:
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_INVALID",
            f"checkpoint manifest digest differs for {task.task_id}",
        )
    manifest = _read_mapping(manifest_path)
    expected_identity = checkpoint_identity(task)
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or manifest.get("identity") != expected_identity
        or manifest.get("payload_sha256") != expected_payload_sha256
    ):
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_IDENTITY",
            f"checkpoint identity differs for {task.task_id}",
        )
    payload_dir = snapshot_dir / "payload"
    records = _payload_file_records(payload_dir)
    if records != manifest.get("files") or _canonical_sha256(records) != expected_payload_sha256:
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_INVALID",
            f"checkpoint payload digest differs for {task.task_id}",
        )


def _latest_checkpoint(
    checkpoint_root: Path,
    task: CampaignTask,
) -> tuple[Path, dict[str, Any]] | None:
    task_dir = checkpoint_task_dir(checkpoint_root, task)
    latest_path = task_dir / "LATEST.json"
    if not latest_path.exists():
        return None
    latest = _read_mapping(latest_path)
    payload_sha256 = latest.get("payload_sha256")
    manifest_sha256 = latest.get("checkpoint_manifest_sha256")
    if (
        latest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or latest.get("task_id") != task.task_id
        or latest.get("identity_sha256") != checkpoint_identity(task)["identity_sha256"]
        or not isinstance(payload_sha256, str)
        or len(payload_sha256) != 64
        or not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
    ):
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_IDENTITY",
            f"LATEST checkpoint identity is invalid for {task.task_id}",
        )
    snapshot_dir = _contained(
        task_dir,
        Path("snapshots") / payload_sha256,
        label="checkpoint snapshot",
    )
    _validate_snapshot(
        snapshot_dir,
        task=task,
        expected_payload_sha256=payload_sha256,
        expected_manifest_sha256=manifest_sha256,
    )
    return snapshot_dir, latest


def _validate_live_checkpoint(live_dir: Path, task: CampaignTask) -> None:
    records = _payload_file_records(live_dir)
    by_path = {str(record["path"]): record for record in records}
    pointer_path = live_dir / "CURRENT.json"
    if pointer_path.is_file():
        pointer = _read_mapping(pointer_path)
        generation_name = pointer.get("generation")
        if (
            pointer.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or pointer.get("task_id") != task.task_id
            or pointer.get("identity_sha256") != checkpoint_identity(task)["identity_sha256"]
            or not isinstance(generation_name, str)
            or Path(generation_name).name != generation_name
        ):
            raise CampaignError(
                "E_CAMPAIGN_CHECKPOINT_IDENTITY",
                f"live checkpoint pointer differs for {task.task_id}",
            )
        generation_dir = _contained(
            live_dir,
            Path("generations") / generation_name,
            label="live checkpoint generation",
        )
        payload_path = generation_dir / "checkpoint.pt"
        metadata_path = generation_dir / "checkpoint.json"
        payload_relative = payload_path.relative_to(live_dir).as_posix()
        metadata_relative = metadata_path.relative_to(live_dir).as_posix()
    else:
        payload_path = live_dir / "checkpoint.pt"
        metadata_path = live_dir / "checkpoint.json"
        payload_relative = "checkpoint.pt"
        metadata_relative = "checkpoint.json"
    if payload_relative not in by_path or metadata_relative not in by_path:
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_INVALID",
            "live checkpoint must contain checkpoint.pt and checkpoint.json",
        )
    metadata = _read_mapping(metadata_path)
    expected_identity_sha256 = checkpoint_identity(task)["identity_sha256"]
    if (
        metadata.get("task_id") != task.task_id
        or metadata.get("identity_sha256") != expected_identity_sha256
        or metadata.get("checkpoint_sha256") != by_path[payload_relative]["sha256"]
    ):
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_IDENTITY",
            f"live checkpoint identity or payload digest differs for {task.task_id}",
        )


def restore_checkpoint(
    checkpoint_root: Path,
    task: CampaignTask,
) -> RestoredCheckpoint:
    """Open the persistent task-scoped live checkpoint, recovering a sealed snapshot if needed."""

    task_dir = checkpoint_task_dir(checkpoint_root, task)
    task_dir.mkdir(parents=True, exist_ok=True)
    workspace = task_dir / "live"
    latest = _latest_checkpoint(checkpoint_root, task)
    if workspace.exists():
        if not workspace.is_dir() or workspace.is_symlink():
            raise CampaignError(
                "E_CAMPAIGN_CHECKPOINT_WORKSPACE",
                f"live checkpoint workspace is unsafe: {workspace}",
            )
        if any(workspace.iterdir()):
            try:
                _validate_live_checkpoint(workspace, task)
            except CampaignError:
                if latest is None:
                    raise
                quarantine = task_dir / "invalid-live" / uuid.uuid4().hex
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                os.replace(workspace, quarantine)
                staging = task_dir / f".live-{uuid.uuid4().hex}"
                shutil.copytree(latest[0] / "payload", staging)
                os.rename(staging, workspace)
                _validate_live_checkpoint(workspace, task)
            records = _payload_file_records(workspace)
            return RestoredCheckpoint(
                workspace=workspace,
                resumed=True,
                payload_sha256=_canonical_sha256(records),
                checkpoint_manifest_sha256=(
                    str(latest[1]["checkpoint_manifest_sha256"]) if latest is not None else None
                ),
            )
        if latest is not None:
            workspace.rmdir()
        else:
            # A previous worker may have crashed after the campaign created
            # its task-scoped workspace but before the trainer wrote the first
            # checkpoint.  Reuse that authenticated empty directory instead
            # of trying to create it again.  Non-empty workspaces have already
            # been validated above and are never removed by this path.
            return RestoredCheckpoint(
                workspace=workspace,
                resumed=False,
                payload_sha256=None,
                checkpoint_manifest_sha256=None,
            )
    if latest is None:
        workspace.mkdir(parents=True, exist_ok=False)
        return RestoredCheckpoint(
            workspace=workspace,
            resumed=False,
            payload_sha256=None,
            checkpoint_manifest_sha256=None,
        )
    snapshot_dir, pointer = latest
    staging = task_dir / f".live-{uuid.uuid4().hex}"
    shutil.copytree(snapshot_dir / "payload", staging)
    os.rename(staging, workspace)
    _validate_live_checkpoint(workspace, task)
    return RestoredCheckpoint(
        workspace=workspace,
        resumed=True,
        payload_sha256=str(pointer["payload_sha256"]),
        checkpoint_manifest_sha256=str(pointer["checkpoint_manifest_sha256"]),
    )


def archive_continue_marker(
    checkpoint_root: Path,
    task: CampaignTask,
    *,
    attempt_id: str,
    reason: str,
) -> Path | None:
    """Atomically consume a pending marker while preserving it as audit history."""

    task_dir = checkpoint_task_dir(checkpoint_root, task)
    source = task_dir / "CONTINUE.json"
    if not source.exists():
        return None
    marker = _read_mapping(source)
    if (
        marker.get("task_id") != task.task_id
        or marker.get("identity_sha256") != checkpoint_identity(task)["identity_sha256"]
    ):
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_IDENTITY",
            f"CONTINUE marker identity differs for {task.task_id}",
        )
    history = task_dir / "history"
    history.mkdir(parents=True, exist_ok=True)
    target = history / f"{reason}-{attempt_id}.CONTINUE.json"
    if target.exists():
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_INVALID",
            f"continuation history already exists: {target}",
        )
    os.replace(source, target)
    return target


def publish_checkpoint(
    checkpoint_root: Path,
    task: CampaignTask,
    *,
    workspace: Path,
    attempt_id: str,
    site_id: str,
) -> PublishedCheckpoint:
    """Atomically publish an immutable checkpoint snapshot and authenticated continuation marker."""

    root = Path(checkpoint_root).expanduser().resolve(strict=False)
    task_dir = checkpoint_task_dir(root, task)
    task_dir.mkdir(parents=True, exist_ok=True)
    staging = _contained(
        root,
        Path(".staging") / f"{task.task_id}.{attempt_id}.{uuid.uuid4().hex}",
        label="checkpoint staging directory",
    )
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    try:
        shutil.copytree(workspace, staging / "payload")
        records = _payload_file_records(staging / "payload")
        payload_sha256 = _canonical_sha256(records)
        identity = checkpoint_identity(task)
        atomic_write_json(
            staging / "checkpoint.json",
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity": identity,
                "payload_sha256": payload_sha256,
                "files": records,
            },
        )
        manifest_sha256 = sha256_file(staging / "checkpoint.json")
        snapshot_dir = _contained(
            task_dir,
            Path("snapshots") / payload_sha256,
            label="checkpoint snapshot",
        )
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        if snapshot_dir.exists():
            _validate_snapshot(
                snapshot_dir,
                task=task,
                expected_payload_sha256=payload_sha256,
                expected_manifest_sha256=manifest_sha256,
            )
        else:
            os.rename(staging, snapshot_dir)
        atomic_write_json(
            task_dir / "LATEST.json",
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "task_id": task.task_id,
                "identity_sha256": identity["identity_sha256"],
                "payload_sha256": payload_sha256,
                "checkpoint_manifest_sha256": manifest_sha256,
                "attempt_id": attempt_id,
                "updated_at": _utc_now(),
                "site_id": site_id,
            },
        )
        continue_path = task_dir / "CONTINUE.json"
        atomic_write_json(
            continue_path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "status": "pending",
                "task_id": task.task_id,
                "task_index": task.task_index,
                "campaign_id": task.campaign_id,
                "assigned_site": task.assigned_site,
                "resource_profile": task.resource_profile,
                "identity_sha256": identity["identity_sha256"],
                "payload_sha256": payload_sha256,
                "checkpoint_manifest_sha256": manifest_sha256,
                "attempt_id": attempt_id,
                "created_at": _utc_now(),
            },
        )
        return PublishedCheckpoint(
            task_dir=task_dir,
            snapshot_dir=snapshot_dir,
            continue_path=continue_path,
            payload_sha256=payload_sha256,
            checkpoint_manifest_sha256=manifest_sha256,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "PLANNED_CONTINUATION_EXIT_CODE",
    "PublishedCheckpoint",
    "RestoredCheckpoint",
    "archive_continue_marker",
    "checkpoint_identity",
    "checkpoint_task_dir",
    "publish_checkpoint",
    "restore_checkpoint",
]
