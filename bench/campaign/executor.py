from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import signal
import socket
import traceback
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bench.campaign.model_artifacts import (
    ModelArtifactError,
    model_artifact_lock_sha256,
    verify_model_artifact_attestations,
)
from bench.report_schema import validate_run_payload
from bench.schema import ExperimentConfig
from bench.seed_sweep import apply_global_seed
from bench.utils.hashing import derive_seed
from bench.utils.io import atomic_write_json, dump_yaml, load_yaml
from bench.utils.runtime import collect_runtime_versions
from modssc.data_loader import verify_dataset_content
from modssc.runtime.continuation import PlannedContinuation, request_continuation
from modssc.sampling.plan import SamplingPlan

from .attempts import (
    seal_attempt_record,
    seal_authorization_event,
    validate_attempt_record,
    validate_authorization_event,
)
from .checkpoint import (
    archive_continue_marker,
    checkpoint_identity,
    publish_checkpoint,
    restore_checkpoint,
)
from .dcl_partition_lock import (
    DCL_DATASET_ID,
    DCL_DIAGNOSTIC_METHOD_PROFILE,
    DCL_DIAGNOSTIC_PROTOCOL_IDS,
    DCL_METHOD_ID,
    DCL_METHOD_PROFILE,
    DCL_PAPER_PROTOCOL_ID,
    DCL_SCREENING_PROTOCOL_ID,
    is_dcl_vote_partition_replay_identity,
    resolve_repo_path,
    verify_dcl_partition_replay,
)
from .errors import CampaignError, TaskLockedError
from .identifiers import validate_safe_identifier
from .manifest import load_manifest, select_task, sha256_file
from .models import CampaignTask, TaskExecutionResult
from .preflight_coverage import validate_task_coverage
from .scientific_gates import discover_gate_registry, guard_task

Runner = Callable[..., Any]
VersionCollector = Callable[..., dict[str, Any]]


def _verify_task_pins(task: CampaignTask) -> None:
    scientific = task.claim_eligible
    canonical_dcl_vote_task = (
        task.track == "paper"
        and task.method_id == DCL_METHOD_ID
        and task.method_profile == DCL_METHOD_PROFILE
        and task.dataset_id == DCL_DATASET_ID
    )
    diagnostic_dcl_vote_task = (
        task.track == "paper"
        and task.method_id == DCL_METHOD_ID
        and task.method_profile == DCL_DIAGNOSTIC_METHOD_PROFILE
        and task.dataset_id == DCL_DATASET_ID
    )
    if task.protocol_id == DCL_PAPER_PROTOCOL_ID and not canonical_dcl_vote_task:
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
            "DCL Vote paper protocol is attached to the wrong task identity",
        )
    if canonical_dcl_vote_task and task.protocol_id not in {
        DCL_PAPER_PROTOCOL_ID,
        DCL_SCREENING_PROTOCOL_ID,
    }:
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
            "DCL Vote task uses an unrecognized protocol id",
        )
    if diagnostic_dcl_vote_task and task.protocol_id not in DCL_DIAGNOSTIC_PROTOCOL_IDS:
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
            "DCL Vote v2 diagnostic task uses an unrecognized control protocol id",
        )
    requires_partition_selection = is_dcl_vote_partition_replay_identity(
        track=task.track,
        method_id=task.method_id,
        method_profile=task.method_profile,
        dataset_id=task.dataset_id,
        protocol_id=task.protocol_id,
    )
    if requires_partition_selection and task.partition_selection is None:
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_REQUIRED",
            "DCL Vote paper execution requires a frozen partition selection",
        )
    if task.partition_selection is not None and not requires_partition_selection:
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
            "partition selection is attached to an ineligible task",
        )
    if not scientific:
        return
    values: dict[str, str | None] = {
        "expected_git_sha": task.expected_git_sha,
        "environment_lock_sha256": task.environment_lock_sha256,
        "expected_dataset_fingerprint": task.expected_dataset_fingerprint,
    }
    if (
        task.campaign_id.startswith("article10-")
        or task.expected_dataset_content_sha256 is not None
    ):
        values["expected_dataset_content_sha256"] = task.expected_dataset_content_sha256
    invalid = [
        field
        for field, value in values.items()
        if not isinstance(value, str)
        or not value
        or value == "unlocked"
        or value.startswith("REPLACE_WITH_")
    ]
    if invalid:
        raise CampaignError(
            "E_CAMPAIGN_TEMPLATE_PLACEHOLDER",
            f"scientific task contains unpinned fields: {', '.join(invalid)}",
        )


def _inject_and_verify_partition_replay(
    task: CampaignTask,
    seeded: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    sampling = seeded.get("sampling")
    if not isinstance(sampling, dict):
        raise CampaignError("E_CAMPAIGN_EFFECTIVE_CONFIG", "sampling must be a mapping")
    if task.partition_selection is None:
        if "replay" in sampling:
            raise CampaignError(
                "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
                "source configurations cannot inject an unbound sampling replay",
            )
        return
    if "replay" in sampling:
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
            "source configuration already contains sampling.replay",
        )
    evidence = dict(task.partition_selection)
    selection_path = resolve_repo_path(
        repo_root,
        str(evidence["selection_path"]),
        label="task.partition_selection.selection_path",
    )
    replay_path = resolve_repo_path(
        repo_root,
        str(evidence["replay_path"]),
        label="task.partition_selection.replay_path",
    )
    plan = sampling.get("plan")
    if not isinstance(plan, Mapping):
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_INVALID",
            "effective sampling plan is missing",
        )
    runtime_evidence = dict(evidence)
    runtime_evidence["selection_path"] = str(selection_path)
    runtime_evidence["replay_path"] = str(replay_path)
    verified = verify_dcl_partition_replay(
        runtime_evidence,
        expected_seed=task.seed,
        expected_dataset_fingerprint=str(task.expected_dataset_fingerprint),
        expected_plan=plan,
    )
    if (
        task.expected_split_fingerprint != verified.entry.split_fingerprint
        or evidence["split_fingerprint"] != task.expected_split_fingerprint
    ):
        raise CampaignError(
            "E_CAMPAIGN_PARTITION_SELECTION_MISMATCH",
            "task split fingerprint differs from the selected replay",
        )
    sampling["replay"] = runtime_evidence


def _verify_preflight_report(
    task: CampaignTask,
    manifest_meta: Mapping[str, Any],
    report_path: Path | None,
    *,
    environment_manifest_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, str] | None:
    scientific = task.claim_eligible
    resolved = report_path or (
        Path(os.environ["MODSSC_PREFLIGHT_REPORT"])
        if os.environ.get("MODSSC_PREFLIGHT_REPORT")
        else None
    )
    if resolved is None:
        if scientific:
            raise CampaignError(
                "E_CAMPAIGN_PREFLIGHT_REQUIRED",
                "scientific execution requires a successful preflight report",
            )
        return None
    report = _read_json(resolved)
    report_sha256 = sha256_file(resolved)
    if report.get("status") != "pass":
        raise CampaignError("E_CAMPAIGN_PREFLIGHT_INVALID", "preflight status is not pass")
    if report.get("schema_version") != 1:
        raise CampaignError("E_CAMPAIGN_PREFLIGHT_INVALID", "preflight schema_version must equal 1")
    try:
        created_at = datetime.fromisoformat(str(report["created_at"]))
        expires_at = datetime.fromisoformat(str(report["expires_at"]))
    except (KeyError, ValueError) as exc:
        raise CampaignError(
            "E_CAMPAIGN_PREFLIGHT_INVALID",
            "preflight created_at/expires_at must be ISO-8601 timestamps",
        ) from exc
    if (
        created_at.tzinfo is None
        or created_at.utcoffset() is None
        or expires_at.tzinfo is None
        or expires_at.utcoffset() is None
    ):
        raise CampaignError(
            "E_CAMPAIGN_PREFLIGHT_INVALID", "preflight timestamps must include a timezone"
        )
    max_age = report.get("max_authorization_age_hours")
    if (
        isinstance(max_age, bool)
        or not isinstance(max_age, int | float)
        or not math.isfinite(float(max_age))
        or float(max_age) <= 0.0
    ):
        raise CampaignError(
            "E_CAMPAIGN_PREFLIGHT_INVALID", "preflight maximum authorization age is invalid"
        )
    created_at = created_at.astimezone(UTC)
    expires_at = expires_at.astimezone(UTC)
    checked_at = now or datetime.now(UTC)
    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise CampaignError(
            "E_CAMPAIGN_PREFLIGHT_INVALID", "executor clock must include a timezone"
        )
    checked_at = checked_at.astimezone(UTC)
    if expires_at <= created_at or expires_at > created_at + timedelta(hours=float(max_age)):
        raise CampaignError(
            "E_CAMPAIGN_PREFLIGHT_INVALID", "preflight validity interval is invalid"
        )
    if created_at > checked_at:
        raise CampaignError(
            "E_CAMPAIGN_PREFLIGHT_INVALID", "preflight creation time is in the future"
        )
    expired_at_execution = checked_at >= expires_at
    # Freshness is checked when the task actually starts.  A timestamp exported
    # at submission time is not authority to run after the report expires.
    if expired_at_execution:
        raise CampaignError("E_CAMPAIGN_PREFLIGHT_EXPIRED", "preflight report has expired")
    if report.get("campaign_id") != task.campaign_id:
        raise CampaignError("E_CAMPAIGN_PREFLIGHT_INVALID", "preflight campaign_id differs")
    if report.get("manifest_sha256") != manifest_meta.get("manifest_sha256"):
        raise CampaignError("E_CAMPAIGN_PREFLIGHT_INVALID", "preflight manifest digest differs")
    if task.schema_version == 4:
        for field in (
            "claim_scope_id",
            "campaign_stage",
            "claim_eligible",
            "gate_policy_id",
            "gate_policy_sha256",
        ):
            if report.get(field) != getattr(task, field):
                raise CampaignError(
                    "E_CAMPAIGN_PREFLIGHT_INVALID",
                    f"preflight {field} differs from the task",
                )
    if task.environment_lock_sha256 != "unlocked":
        reported_lock = report.get("environment_lock_sha256")
        if reported_lock is not None and reported_lock != task.environment_lock_sha256:
            raise CampaignError(
                "E_CAMPAIGN_PREFLIGHT_INVALID",
                "preflight environment lock differs from the task",
            )
        resolved_environment = environment_manifest_path or (
            Path(os.environ["MODSSC_ENVIRONMENT_MANIFEST"])
            if os.environ.get("MODSSC_ENVIRONMENT_MANIFEST")
            else None
        )
        if resolved_environment is None:
            # An explicit digest remains sufficient for installations that do
            # not use optional deployment adapters.
            pass
        else:
            if report.get("environment_manifest_sha256") != sha256_file(resolved_environment):
                raise CampaignError(
                    "E_CAMPAIGN_PREFLIGHT_INVALID",
                    "preflight environment manifest digest differs",
                )
            environment_manifest = _read_json(resolved_environment)
            locked_identity = environment_manifest.get("environment_lock")
            model_lock = (
                locked_identity.get("model_artifacts")
                if isinstance(locked_identity, Mapping)
                else None
            )
            if not isinstance(model_lock, Mapping) or report.get(
                "model_artifacts_sha256"
            ) != model_artifact_lock_sha256(model_lock):
                raise CampaignError(
                    "E_CAMPAIGN_PREFLIGHT_INVALID",
                    "preflight model artifact lock differs",
                )
            try:
                verify_model_artifact_attestations(report.get("model_artifact_attestations"))
            except ModelArtifactError as exc:
                raise CampaignError("E_CAMPAIGN_PREFLIGHT_INVALID", str(exc)) from exc
    raw_coverage = report.get("task_coverage")
    if raw_coverage is None:
        if scientific:
            raise CampaignError(
                "E_CAMPAIGN_PREFLIGHT_INVALID",
                "scientific preflight has no immutable task coverage",
            )
    else:
        try:
            coverage = validate_task_coverage(raw_coverage)
        except ValueError as exc:
            raise CampaignError("E_CAMPAIGN_PREFLIGHT_INVALID", str(exc)) from exc
        if report.get("required_architecture") != coverage["architecture"]:
            raise CampaignError(
                "E_CAMPAIGN_PREFLIGHT_INVALID",
                "preflight architecture differs from its task coverage",
            )
        if task.task_id not in coverage["task_ids"]:
            raise CampaignError(
                "E_CAMPAIGN_PREFLIGHT_INVALID",
                "task is absent from the preflight coverage",
            )
    expected_content = task.expected_dataset_content_sha256
    if expected_content is None:
        return None
    dataset_check = next(
        (
            check
            for check in report.get("checks", [])
            if isinstance(check, Mapping) and check.get("name") == "datasets"
        ),
        None,
    )
    evidence_by_request = (
        dataset_check.get("evidence_by_request") if isinstance(dataset_check, Mapping) else None
    )
    evidence = (
        evidence_by_request.get(task.dataset_request_sha256)
        if isinstance(evidence_by_request, Mapping)
        else None
    )
    if not isinstance(evidence, Mapping):
        raise CampaignError(
            "E_CAMPAIGN_PREFLIGHT_INVALID",
            "preflight has no dataset content proof for this task",
        )
    required = (
        "content_sha256",
        "content_manifest_sha256",
        "cache_state_sha256",
        "cache_fingerprint",
    )
    if any(
        not isinstance(evidence.get(field), str) or not evidence.get(field) for field in required
    ):
        raise CampaignError(
            "E_CAMPAIGN_PREFLIGHT_INVALID", "preflight dataset content proof is incomplete"
        )
    if evidence.get("content_sha256") != expected_content:
        raise CampaignError(
            "E_CAMPAIGN_PREFLIGHT_INVALID", "preflight dataset content digest differs"
        )
    return {field: str(evidence[field]) for field in required} | {
        "preflight_report_sha256": report_sha256,
        "preflight_expires_at": expires_at.isoformat(),
        "preflight_expired_at_execution": str(expired_at_execution).lower(),
        "preflight_expiry_policy": "fresh",
    }


def _verify_dataset_content_state(
    raw: dict[str, Any],
    task: CampaignTask,
    preflight_evidence: Mapping[str, str] | None,
) -> dict[str, str] | None:
    if task.expected_dataset_content_sha256 is None:
        return None
    if preflight_evidence is None:
        raise CampaignError("E_CAMPAIGN_PREFLIGHT_INVALID", "dataset content proof is required")
    cfg = ExperimentConfig.from_dict(raw)
    cache_dir = (
        Path(cfg.dataset.cache_dir).expanduser().resolve() if cfg.dataset.cache_dir else None
    )
    try:
        actual = verify_dataset_content(
            cfg.dataset.id,
            cache_dir=cache_dir,
            options=dict(cfg.dataset.options),
            rehash=False,
        )
    except Exception as exc:
        raise CampaignError(
            "E_CAMPAIGN_DATASET_CONTENT_MISMATCH",
            f"cannot verify dataset content state: {type(exc).__name__}: {exc}",
        ) from exc
    for field in (
        "content_sha256",
        "content_manifest_sha256",
        "cache_state_sha256",
        "cache_fingerprint",
    ):
        if actual.get(field) != preflight_evidence.get(field):
            raise CampaignError(
                "E_CAMPAIGN_DATASET_CONTENT_MISMATCH",
                f"dataset cache changed after preflight ({field})",
            )
    preflight_metadata = {
        field: preflight_evidence[field]
        for field in (
            "preflight_expires_at",
            "preflight_expired_at_execution",
            "preflight_expiry_policy",
            "preflight_validated_at",
            "preflight_job_id",
        )
        if field in preflight_evidence
    }
    return {
        **actual,
        "preflight_report_sha256": preflight_evidence["preflight_report_sha256"],
        **preflight_metadata,
    }


@dataclass(frozen=True)
class _TaskLock:
    lock_dir: Path
    owner_token: str
    guard_fd: int


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def _checkpoint_runtime_environment(
    *,
    task: CampaignTask,
    workspace: Path,
    resumed: bool,
) -> Any:
    values = {
        "MODSSC_CHECKPOINT_ROOT": str(workspace),
        "MODSSC_TASK_ID": task.task_id,
        "MODSSC_CHECKPOINT_RESUME": "1" if resumed else "0",
        "MODSSC_CHECKPOINT_IDENTITY_SHA256": str(checkpoint_identity(task)["identity_sha256"]),
        "MODSSC_EXPECTED_GIT_SHA": task.expected_git_sha,
        "MODSSC_EXPECTED_GIT_DIFF_SHA256": task.expected_git_diff_sha256 or "",
        "MODSSC_ENVIRONMENT_LOCK_SHA256": task.environment_lock_sha256,
        "MODSSC_METHOD_PROFILE": task.method_profile,
        "MODSSC_RESOURCE_PROFILE": task.resource_profile,
        "MODSSC_EXPECTED_SPLIT_FINGERPRINT": task.expected_split_fingerprint or "",
        "MODSSC_PARTITION_SHA256": str(checkpoint_identity(task)["partition_sha256"]),
        "MODSSC_CONTINUATION_REQUESTED": "0",
        "MODSSC_CONTINUATION_SIGNAL": "",
        "MODSSC_CAMPAIGN_CHECKPOINT_DIR": str(workspace),
        "MODSSC_CAMPAIGN_CHECKPOINT_RESUME": "1" if resumed else "0",
        "MODSSC_CAMPAIGN_CHECKPOINT_IDENTITY_SHA256": str(
            checkpoint_identity(task)["identity_sha256"]
        ),
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _planned_continuation_signal(*, enabled: bool) -> Any:
    if not enabled:
        yield
        return
    user_signal = getattr(signal, "SIGUSR1", None)
    if user_signal is None:
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_SIGNAL",
            "checkpoint continuation requires SIGUSR1 support",
        )
    try:
        previous = signal.getsignal(user_signal)

        def handle_continuation_signal(signum: int, _frame: Any) -> None:
            request_continuation(signum)

        signal.signal(user_signal, handle_continuation_signal)
    except (OSError, ValueError) as exc:
        raise CampaignError(
            "E_CAMPAIGN_CHECKPOINT_SIGNAL",
            "cannot install the planned-continuation signal handler",
        ) from exc
    try:
        yield
    finally:
        signal.signal(user_signal, previous)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError("E_CAMPAIGN_RESULT_INVALID", f"invalid JSON file: {path}") from exc
    if not isinstance(raw, dict):
        raise CampaignError("E_CAMPAIGN_RESULT_INVALID", f"JSON root must be a mapping: {path}")
    return raw


def _result_run_json(result_dir: Path) -> Path:
    return result_dir / "run" / "run.json"


def _contained_file(root: Path, relative: str, *, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"{label} escapes the run directory"
        ) from exc
    if not candidate.is_file():
        raise CampaignError("E_CAMPAIGN_RESULT_INVALID", f"{label} is missing: {candidate}")
    return candidate


def _contained_result_path(result_root: Path, relative: str | Path, *, label: str) -> Path:
    root = Path(result_root).resolve(strict=False)
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_PATH_OUTSIDE_ROOT",
            f"{label} escapes result root: {relative}",
        ) from exc
    return candidate


def _validate_sampling_replay(
    *,
    run_json_path: Path,
    replay: Mapping[str, Any],
    dataset_fingerprint: str,
    split_fingerprint: str,
    task_id: str,
    partition_selection: Mapping[str, Any] | None = None,
) -> None:
    if replay.get("format") != "modssc.sampling.storage.v1":
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split replay format differs for {task_id}"
        )
    replay_path = replay.get("path")
    manifest_name = replay.get("manifest")
    manifest_sha256 = replay.get("manifest_sha256")
    if not isinstance(replay_path, str) or not replay_path:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split replay path is missing for {task_id}"
        )
    if not isinstance(manifest_name, str) or not manifest_name:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split manifest path is missing for {task_id}"
        )
    if not isinstance(manifest_sha256, str) or not manifest_sha256:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split manifest digest is missing for {task_id}"
        )

    run_dir = run_json_path.parent.resolve()
    replay_dir = (run_dir / replay_path).resolve()
    try:
        replay_dir.relative_to(run_dir)
    except ValueError as exc:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split replay escapes the run directory for {task_id}"
        ) from exc
    manifest_path = _contained_file(replay_dir, manifest_name, label="split manifest")
    if sha256_file(manifest_path) != manifest_sha256:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split manifest digest differs for {task_id}"
        )
    if partition_selection is not None and manifest_sha256 != partition_selection.get(
        "split_manifest_sha256"
    ):
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID",
            f"split manifest differs from the selected partition for {task_id}",
        )
    if partition_selection is not None:
        selection = replay.get("selection")
        expected_selection = {
            "kind": partition_selection["kind"],
            "selection_sha256": partition_selection["selection_sha256"],
            "selection_rank": partition_selection["selection_rank"],
            "source_task_id": partition_selection["source_task_id"],
            "source_task_row_sha256": partition_selection["source_task_row_sha256"],
        }
        if not isinstance(selection, Mapping) or dict(selection) != expected_selection:
            raise CampaignError(
                "E_CAMPAIGN_RESULT_INVALID",
                f"split selection provenance differs for {task_id}",
            )
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("format") != replay.get("format"):
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split manifest schema differs for {task_id}"
        )
    if manifest.get("dataset_fingerprint") != dataset_fingerprint:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"replay dataset fingerprint differs for {task_id}"
        )
    if manifest.get("split_fingerprint") != split_fingerprint:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"replay split fingerprint differs for {task_id}"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split manifest file table is missing for {task_id}"
        )
    for name in ("split.json", "arrays.npz"):
        record = files.get(name)
        expected_digest = record.get("sha256") if isinstance(record, Mapping) else None
        if not isinstance(expected_digest, str) or not expected_digest:
            raise CampaignError(
                "E_CAMPAIGN_RESULT_INVALID",
                f"split manifest digest for {name} is missing for {task_id}",
            )
        artifact_path = _contained_file(replay_dir, name, label=f"split artifact {name}")
        if sha256_file(artifact_path) != expected_digest:
            raise CampaignError(
                "E_CAMPAIGN_RESULT_INVALID",
                f"split artifact digest for {name} differs for {task_id}",
            )
        if partition_selection is not None:
            selection_field = "split_json_sha256" if name == "split.json" else "split_arrays_sha256"
            if expected_digest != partition_selection.get(selection_field):
                raise CampaignError(
                    "E_CAMPAIGN_RESULT_INVALID",
                    f"split artifact {name} differs from the selected partition for {task_id}",
                )
    split_metadata = _read_json(_contained_file(replay_dir, "split.json", label="split metadata"))
    if split_metadata.get("dataset_fingerprint") != dataset_fingerprint:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split metadata dataset differs for {task_id}"
        )
    if split_metadata.get("split_fingerprint") != split_fingerprint:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split metadata fingerprint differs for {task_id}"
        )


def validate_result_directory(
    result_dir: Path, task: CampaignTask
) -> tuple[Path, dict[str, Any], str]:
    success_path = result_dir / "SUCCESS.json"
    envelope_path = result_dir / "task.json"
    run_json_path = _result_run_json(result_dir)
    if not success_path.is_file() or not envelope_path.is_file() or not run_json_path.is_file():
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"result bundle is incomplete: {result_dir}"
        )

    success = _read_json(success_path)
    envelope = _read_json(envelope_path)
    run_payload = _read_json(run_json_path)
    validate_run_payload(run_payload)
    if success.get("status") != "success" or success.get("task_id") != task.task_id:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"invalid success marker for {task.task_id}"
        )
    if success.get("row_sha256") != task.row_sha256:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"result row hash differs for {task.task_id}"
        )
    if (
        task.expected_dataset_content_sha256 is not None
        and success.get("dataset_content_sha256") != task.expected_dataset_content_sha256
    ):
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID",
            f"success marker dataset content differs for {task.task_id}",
        )
    run_digest = sha256_file(run_json_path)
    if success.get("run_json_sha256") != run_digest:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"run.json digest differs for {task.task_id}"
        )
    effective_config_name = success.get("effective_config_path")
    effective_config_digest = success.get("effective_config_sha256")
    if not isinstance(effective_config_name, str) or not effective_config_name:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID",
            f"effective configuration path is missing for {task.task_id}",
        )
    if not isinstance(effective_config_digest, str) or not effective_config_digest:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID",
            f"effective configuration digest is missing for {task.task_id}",
        )
    effective_config_path = _contained_file(
        result_dir,
        effective_config_name,
        label="effective configuration",
    )
    if sha256_file(effective_config_path) != effective_config_digest:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID",
            f"effective configuration digest differs for {task.task_id}",
        )
    envelope_task = envelope.get("task")
    run = run_payload.get("run")
    task_info = run_payload.get("task_info")
    versions = run_payload.get("versions")
    if not isinstance(run, Mapping) or run.get("status") != "success":
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"run is not successful for {task.task_id}"
        )
    if run.get("seed") != task.seed:
        raise CampaignError("E_CAMPAIGN_RESULT_INVALID", f"run seed differs for {task.task_id}")
    if not isinstance(task_info, Mapping):
        raise CampaignError("E_CAMPAIGN_RESULT_INVALID", "run.json task_info is missing")
    if task_info.get("method_id") != task.method_id:
        raise CampaignError("E_CAMPAIGN_RESULT_INVALID", f"method differs for {task.task_id}")
    if task_info.get("dataset_id") != task.dataset_id:
        raise CampaignError("E_CAMPAIGN_RESULT_INVALID", f"dataset differs for {task.task_id}")
    if not isinstance(versions, Mapping) or versions.get("git_sha") != task.expected_git_sha:
        raise CampaignError("E_CAMPAIGN_RESULT_INVALID", f"Git revision differs for {task.task_id}")
    if versions.get("git_dirty") is not False:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"run used a dirty worktree for {task.task_id}"
        )
    if versions.get("git_diff_sha256") != task.expected_git_diff_sha256:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"Git worktree fingerprint differs for {task.task_id}"
        )
    artifacts = run_payload.get("artifacts")
    dataset = artifacts.get("dataset") if isinstance(artifacts, Mapping) else None
    fingerprint = dataset.get("fingerprint") if isinstance(dataset, Mapping) else None
    if not isinstance(fingerprint, str) or not fingerprint:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"dataset fingerprint is missing for {task.task_id}"
        )
    if (
        task.expected_dataset_fingerprint is not None
        and fingerprint != task.expected_dataset_fingerprint
    ):
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID",
            f"dataset fingerprint differs for {task.task_id}",
        )
    content_sha256 = dataset.get("content_sha256") if isinstance(dataset, Mapping) else None
    content_manifest_sha256 = (
        dataset.get("content_manifest_sha256") if isinstance(dataset, Mapping) else None
    )
    if task.expected_dataset_content_sha256 is not None:
        if content_sha256 != task.expected_dataset_content_sha256:
            raise CampaignError(
                "E_CAMPAIGN_RESULT_INVALID",
                f"dataset content digest differs for {task.task_id}",
            )
        content_proof = envelope.get("dataset_content_proof")
        if not isinstance(content_proof, Mapping):
            raise CampaignError(
                "E_CAMPAIGN_RESULT_INVALID",
                f"dataset content proof is missing for {task.task_id}",
            )
        if (
            content_proof.get("content_sha256") != task.expected_dataset_content_sha256
            or content_proof.get("content_manifest_sha256") != content_manifest_sha256
        ):
            raise CampaignError(
                "E_CAMPAIGN_RESULT_INVALID",
                f"dataset content proof differs for {task.task_id}",
            )
        required_proof = (
            "cache_state_sha256",
            "cache_fingerprint",
            "preflight_report_sha256",
        )
        if any(
            not isinstance(content_proof.get(field), str) or not content_proof.get(field)
            for field in required_proof
        ):
            raise CampaignError(
                "E_CAMPAIGN_RESULT_INVALID",
                f"dataset content proof is incomplete for {task.task_id}",
            )
    sampling = artifacts.get("sampling") if isinstance(artifacts, Mapping) else None
    split_fingerprint = sampling.get("split_fingerprint") if isinstance(sampling, Mapping) else None
    replay = sampling.get("replay") if isinstance(sampling, Mapping) else None
    if not isinstance(split_fingerprint, str) or not split_fingerprint:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split fingerprint is missing for {task.task_id}"
        )
    if (
        task.expected_split_fingerprint is not None
        and split_fingerprint != task.expected_split_fingerprint
    ):
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split fingerprint differs for {task.task_id}"
        )
    if not isinstance(replay, Mapping):
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"split replay is missing for {task.task_id}"
        )
    _validate_sampling_replay(
        run_json_path=run_json_path,
        replay=replay,
        dataset_fingerprint=fingerprint,
        split_fingerprint=split_fingerprint,
        task_id=task.task_id,
        partition_selection=task.partition_selection,
    )
    method_artifact = artifacts.get("method") if isinstance(artifacts, Mapping) else None
    profile = method_artifact.get("profile") if isinstance(method_artifact, Mapping) else None
    if profile != task.method_profile:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"method profile differs for {task.task_id}"
        )
    if not isinstance(envelope_task, Mapping) or dict(envelope_task) != task.to_dict():
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"task envelope differs for {task.task_id}"
        )
    if envelope.get("environment_lock_sha256") != task.environment_lock_sha256:
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"environment lock differs for {task.task_id}"
        )
    if envelope.get("site_id") != task.assigned_site and task.assigned_site != "any":
        raise CampaignError(
            "E_CAMPAIGN_RESULT_INVALID", f"execution site differs for {task.task_id}"
        )
    return run_json_path, run_payload, run_digest


def _verify_source_config(task: CampaignTask, *, repo_root: Path) -> Path:
    config_path = (repo_root / task.config_path).resolve()
    try:
        config_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise CampaignError(
            "E_CAMPAIGN_CONFIG_OUTSIDE_REPO", f"config escapes repository: {task.config_path}"
        ) from exc
    if not config_path.is_file():
        raise CampaignError(
            "E_CAMPAIGN_CONFIG_MISSING", f"configuration not found: {task.config_path}"
        )
    if sha256_file(config_path) != task.source_config_sha256:
        raise CampaignError(
            "E_CAMPAIGN_CONFIG_CHANGED", f"configuration changed: {task.config_path}"
        )
    return config_path


def _verify_code(
    task: CampaignTask, *, repo_root: Path, version_collector: VersionCollector
) -> dict[str, Any]:
    versions = version_collector(repo_root=repo_root)
    if versions.get("git_sha") != task.expected_git_sha:
        raise CampaignError(
            "E_CAMPAIGN_CODE_MISMATCH",
            f"task expects Git {task.expected_git_sha}, got {versions.get('git_sha')}",
        )
    if versions.get("git_diff_sha256") != task.expected_git_diff_sha256:
        raise CampaignError(
            "E_CAMPAIGN_CODE_MISMATCH", "worktree fingerprint differs from manifest"
        )
    return versions


def _verify_environment(
    task: CampaignTask,
    actual_digest: str | None,
    *,
    environment_manifest_path: Path | None,
) -> None:
    expected = task.environment_lock_sha256
    if expected == "unlocked":
        return
    manifest_value = environment_manifest_path or (
        Path(os.environ["MODSSC_ENVIRONMENT_MANIFEST"])
        if os.environ.get("MODSSC_ENVIRONMENT_MANIFEST")
        else None
    )
    if manifest_value is not None:
        from bench.campaign.build_manifest import (
            collect_environment_identity,
            environment_identity_sha256,
            python_environment_identity,
            validate_environment_lock,
        )

        manifest = _read_json(manifest_value)
        locked_identity = manifest.get("environment_lock")
        if not isinstance(locked_identity, dict):
            raise CampaignError(
                "E_CAMPAIGN_ENVIRONMENT_MISMATCH",
                "environment manifest has no immutable environment_lock payload",
            )
        try:
            validate_environment_lock(locked_identity)
        except ValueError as exc:
            raise CampaignError("E_CAMPAIGN_ENVIRONMENT_MISMATCH", str(exc)) from exc
        locked_digest = environment_identity_sha256(locked_identity)
        if manifest.get("environment_lock_sha256") != locked_digest or locked_digest != expected:
            raise CampaignError(
                "E_CAMPAIGN_ENVIRONMENT_MISMATCH",
                "environment manifest digest does not match the task",
            )
        actual_identity = collect_environment_identity()
        if python_environment_identity(actual_identity) != python_environment_identity(
            locked_identity
        ):
            raise CampaignError(
                "E_CAMPAIGN_ENVIRONMENT_MISMATCH",
                "active Python environment differs from the environment manifest",
            )
        model_lock = locked_identity["model_artifacts"]
        if manifest.get("model_artifacts_sha256") not in (
            None,
            model_artifact_lock_sha256(model_lock),
        ):
            raise CampaignError(
                "E_CAMPAIGN_ENVIRONMENT_MISMATCH",
                "top-level model artifact digest is invalid",
            )
        return
    actual = actual_digest or os.environ.get("MODSSC_ENVIRONMENT_LOCK_SHA256")
    if actual != expected:
        raise CampaignError(
            "E_CAMPAIGN_ENVIRONMENT_MISMATCH",
            "MODSSC_ENVIRONMENT_LOCK_SHA256 does not match the manifest",
        )


def _verify_execution_target(task: CampaignTask, *, site_id: str) -> None:
    validate_safe_identifier(
        site_id,
        field="site_id",
        code="E_CAMPAIGN_SITE_MISMATCH",
    )
    if task.assigned_site != "any" and site_id != task.assigned_site:
        raise CampaignError(
            "E_CAMPAIGN_SITE_MISMATCH",
            f"task is assigned to {task.assigned_site}, not {site_id}",
        )


def _existing_lock_age(lock_dir: Path) -> timedelta:
    owner_path = lock_dir / "owner.json"
    if owner_path.is_file():
        try:
            owner = _read_json(owner_path)
            created = datetime.fromisoformat(str(owner["created_at"]))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            return datetime.now(UTC) - created.astimezone(UTC)
        except (CampaignError, KeyError, TypeError, ValueError):
            pass
    return datetime.now(UTC) - datetime.fromtimestamp(lock_dir.stat().st_mtime, tz=UTC)


def _acquire_lock(
    lock_dir: Path,
    task: CampaignTask,
    *,
    site_id: str,
    reclaim_stale_after: timedelta | None,
) -> _TaskLock:
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    guard_path = lock_dir.parent / f".{lock_dir.name}.guard"
    guard_fd = os.open(guard_path, os.O_CREAT | os.O_RDWR, 0o600)
    created_lock = False
    try:
        try:
            fcntl.flock(guard_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise TaskLockedError(task.task_id) from exc

        if lock_dir.exists():
            if not lock_dir.is_dir():
                raise TaskLockedError(task.task_id)
            if reclaim_stale_after is None or _existing_lock_age(lock_dir) <= reclaim_stale_after:
                raise TaskLockedError(task.task_id)
            quarantine = (
                lock_dir.parent.parent
                / "stale-locks"
                / f"{lock_dir.name}.{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.{uuid.uuid4().hex}"
            )
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            os.replace(lock_dir, quarantine)

        try:
            lock_dir.mkdir()
            created_lock = True
        except (FileExistsError, OSError) as exc:
            raise TaskLockedError(task.task_id) from exc
        owner_token = uuid.uuid4().hex
        atomic_write_json(
            lock_dir / "owner.json",
            {
                "schema_version": 1,
                "task_id": task.task_id,
                "owner_token": owner_token,
                "created_at": utc_now(),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "site_id": site_id,
            },
        )
        return _TaskLock(
            lock_dir=lock_dir,
            owner_token=owner_token,
            guard_fd=guard_fd,
        )
    except Exception:
        if created_lock:
            shutil.rmtree(lock_dir, ignore_errors=True)
        try:
            fcntl.flock(guard_fd, fcntl.LOCK_UN)
        finally:
            os.close(guard_fd)
        raise


def _release_lock(lock: _TaskLock) -> bool:
    released = False
    try:
        owner_path = lock.lock_dir / "owner.json"
        try:
            owner = _read_json(owner_path)
        except CampaignError:
            owner = {}
        if owner.get("owner_token") == lock.owner_token and lock.lock_dir.is_dir():
            shutil.rmtree(lock.lock_dir)
            released = True
        return released
    finally:
        try:
            fcntl.flock(lock.guard_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock.guard_fd)


def _find_run_json(output_root: Path) -> Path | None:
    candidates = sorted(output_root.rglob("run.json")) if output_root.exists() else []
    return candidates[-1] if candidates else None


def _classify_failure(error: BaseException, *, failure_phase: str = "run") -> dict[str, Any]:
    message = f"{type(error).__name__}: {error}".lower()
    code = str(getattr(error, "code", "")).upper()
    if isinstance(error, MemoryError) or "out of memory" in message or "cuda oom" in message:
        return {
            "failure_class": "resource_oom",
            "retryable": False,
            "resource_change_required": True,
        }
    if isinstance(error, TimeoutError) or "time limit" in message or "timeout" in message:
        return {
            "failure_class": "resource_timeout",
            "retryable": False,
            "resource_change_required": True,
        }
    if failure_phase == "precondition" and isinstance(error, CampaignError):
        return {
            "failure_class": "deterministic",
            "retryable": False,
            "resource_change_required": False,
        }
    deterministic_tokens = (
        "CONFIG",
        "SCHEMA",
        "VALIDATION",
        "DEPENDENCY",
        "CHECKPOINT",
        "SHAPE",
        "DTYPE",
        "AUTO_FORBIDDEN",
    )
    if isinstance(error, (ValueError, TypeError, ImportError)) or any(
        token in code for token in deterministic_tokens
    ):
        return {
            "failure_class": "deterministic",
            "retryable": False,
            "resource_change_required": False,
        }
    return {
        "failure_class": "infrastructure",
        "retryable": True,
        "resource_change_required": False,
    }


def _publish_failure(
    *,
    result_root: Path,
    task: CampaignTask,
    attempt_id: str,
    attempt_work: Path,
    run_json_path: Path | None,
    error: BaseException,
    site_id: str,
    failure_phase: str = "run",
) -> Path:
    _ = attempt_work  # Operational workspace paths never enter authenticated evidence.
    target = _contained_result_path(
        result_root,
        Path("attempts") / task.task_id[:2] / task.task_id / attempt_id,
        label="failure attempt",
    )
    staging = _contained_result_path(
        result_root,
        Path(".staging") / f"failed-{task.task_id}.{attempt_id}",
        label="failure staging directory",
    )
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    try:
        if run_json_path is not None and run_json_path.parent.is_dir():
            shutil.copytree(run_json_path.parent, staging / "run")
        failure = _classify_failure(error, failure_phase=failure_phase)
        atomic_write_json(
            staging / "attempt.json",
            seal_attempt_record(
                {
                    "task_id": task.task_id,
                    "row_sha256": task.row_sha256,
                    "attempt_id": attempt_id,
                    "status": "failed",
                    "site_id": site_id,
                    "finished_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "failure_phase": failure_phase,
                    **failure,
                }
            ),
        )
        attempt_payload = _read_json(staging / "attempt.json")
        try:
            validate_attempt_record(
                attempt_payload,
                task=task,
                directory_name=attempt_id,
            )
        except CampaignError as exc:
            raise CampaignError(
                "E_CAMPAIGN_RESULT_INVALID",
                f"failure attempt staging is invalid for {task.task_id}",
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(target):
            raise CampaignError(
                "E_CAMPAIGN_ATTEMPT_EXISTS",
                f"failure attempt already exists: {target}",
            )
        os.rename(staging, target)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _publish_continuation_attempt(
    *,
    result_root: Path,
    task: CampaignTask,
    attempt_id: str,
    site_id: str,
    signal_number: int,
    checkpoint_payload_sha256: str,
    checkpoint_manifest_sha256: str,
) -> Path:
    target = _contained_result_path(
        result_root,
        Path("attempts") / task.task_id[:2] / task.task_id / attempt_id,
        label="continuation attempt",
    )
    staging = _contained_result_path(
        result_root,
        Path(".staging") / f"continuation-{task.task_id}.{attempt_id}",
        label="continuation staging directory",
    )
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    try:
        atomic_write_json(
            staging / "attempt.json",
            seal_attempt_record(
                {
                    "task_id": task.task_id,
                    "row_sha256": task.row_sha256,
                    "attempt_id": attempt_id,
                    "status": "continuation",
                    "site_id": site_id,
                    "finished_at": utc_now(),
                    "event_class": "planned_continuation",
                    "failure_class": None,
                    "retryable": False,
                    "resource_change_required": False,
                    "signal_number": int(signal_number),
                    "checkpoint_payload_sha256": checkpoint_payload_sha256,
                    "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
                    "checkpoint_reference": (
                        f"checkpoint://tasks/{task.task_id[:2]}/{task.task_id}/CONTINUE.json"
                    ),
                }
            ),
        )
        payload = _read_json(staging / "attempt.json")
        try:
            validate_attempt_record(payload, task=task, directory_name=attempt_id)
        except CampaignError as exc:
            raise CampaignError(
                "E_CAMPAIGN_RESULT_INVALID",
                f"continuation attempt staging is invalid for {task.task_id}",
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(target):
            raise CampaignError(
                "E_CAMPAIGN_ATTEMPT_EXISTS",
                f"continuation attempt already exists: {target}",
            )
        os.rename(staging, target)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _publish_authorization_expired(
    *,
    result_root: Path,
    task: CampaignTask,
    event_id: str,
    report_path: Path,
    site_id: str,
) -> Path:
    """Publish a non-attempt event proving an authorization expired before execution."""

    target = _contained_result_path(
        result_root,
        Path("events") / task.task_id[:2] / task.task_id / event_id,
        label="authorization event",
    )
    staging = _contained_result_path(
        result_root,
        Path(".staging") / f"authorization-{task.task_id}.{event_id}",
        label="authorization event staging directory",
    )
    report = _read_json(report_path)
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    try:
        atomic_write_json(
            staging / "event.json",
            seal_authorization_event(
                {
                    "task_id": task.task_id,
                    "row_sha256": task.row_sha256,
                    "event_id": event_id,
                    "event_class": "authorization_expired",
                    "site_id": site_id,
                    "observed_at": utc_now(),
                    "expires_at": str(report.get("expires_at")),
                    "preflight_report_sha256": sha256_file(report_path),
                }
            ),
        )
        payload = _read_json(staging / "event.json")
        validate_authorization_event(payload, task=task, directory_name=event_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(target):
            raise CampaignError(
                "E_CAMPAIGN_ATTEMPT_EXISTS",
                f"authorization event already exists: {target}",
            )
        os.rename(staging, target)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _publish_success(
    *,
    result_root: Path,
    task: CampaignTask,
    attempt_id: str,
    run_dir: Path,
    effective_config_path: Path,
    site_id: str,
    code_versions: Mapping[str, Any],
    dataset_content_proof: Mapping[str, str] | None,
) -> Path:
    target = _contained_result_path(
        result_root,
        task.output_relpath,
        label="successful task result",
    )
    if target.exists():
        validate_result_directory(target, task)
        return target

    staging = _contained_result_path(
        result_root,
        Path(".staging") / f"success-{task.task_id}.{attempt_id}",
        label="success staging directory",
    )
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    try:
        shutil.copytree(run_dir, staging / "run")
        if not effective_config_path.is_file():
            raise CampaignError(
                "E_CAMPAIGN_RESULT_INVALID",
                f"effective configuration is missing: {effective_config_path}",
            )
        staged_effective_config = staging / "effective.yaml"
        shutil.copy2(effective_config_path, staged_effective_config)
        staged_run_json = staging / "run" / "run.json"
        run_payload = _read_json(staged_run_json)
        validate_run_payload(run_payload)
        run_digest = sha256_file(staged_run_json)
        effective_config_digest = sha256_file(staged_effective_config)
        atomic_write_json(
            staging / "task.json",
            {
                "schema_version": 1,
                "task": task.to_dict(),
                "attempt_id": attempt_id,
                "site_id": site_id,
                "published_at": utc_now(),
                "executor_versions": dict(code_versions),
                "environment_lock_sha256": task.environment_lock_sha256,
                "dataset_content_proof": (
                    None if dataset_content_proof is None else dict(dataset_content_proof)
                ),
            },
        )
        atomic_write_json(
            staging / "SUCCESS.json",
            {
                "schema_version": 1,
                "task_id": task.task_id,
                "row_sha256": task.row_sha256,
                "status": "success",
                "run_json_sha256": run_digest,
                "effective_config_path": "effective.yaml",
                "effective_config_sha256": effective_config_digest,
                "dataset_content_sha256": task.expected_dataset_content_sha256,
            },
        )
        validate_result_directory(staging, task)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            validate_result_directory(target, task)
            return target
        try:
            os.rename(staging, target)
        except OSError:
            if target.exists():
                validate_result_directory(target, task)
                return target
            raise
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _default_runner(config_path: Path, *, raw: dict[str, Any], cfg: ExperimentConfig) -> Any:
    from bench.main import run_experiment_single

    return run_experiment_single(config_path, raw=raw, cfg=cfg)


def _verify_effective_sampling_seeds(task: CampaignTask, cfg: ExperimentConfig) -> None:
    sampling_seed = (
        int(cfg.sampling.seed)
        if cfg.sampling.seed is not None
        else int(derive_seed(task.seed, "sampling"))
    )
    plan = SamplingPlan.from_dict(cfg.sampling.plan)
    actual = plan.component_seeds.resolve(sampling_seed)
    if task.schema_version == 1:
        legacy_recorded_seed = (
            int(cfg.sampling.seed) if cfg.sampling.seed is not None else int(task.seed)
        )
        if task.split_seed != legacy_recorded_seed:
            raise CampaignError(
                "E_CAMPAIGN_EFFECTIVE_CONFIG",
                "legacy split_seed differs from the v1 campaign convention",
            )
        return
    if actual != task.sampling_component_seeds or task.split_seed != actual["split"]:
        raise CampaignError(
            "E_CAMPAIGN_EFFECTIVE_CONFIG",
            "effective sampling component seeds differ from the campaign manifest",
        )


def execute_task(
    manifest_path: Path,
    *,
    repo_root: Path,
    result_root: Path,
    work_root: Path,
    site_id: str,
    index: int | None = None,
    task_id: str | None = None,
    meta_path: Path | None = None,
    environment_lock_sha256: str | None = None,
    environment_manifest_path: Path | None = None,
    preflight_report_path: Path | None = None,
    reclaim_stale_lock_after: timedelta | None = None,
    gate_registry_path: Path | None = None,
    checkpoint_root: Path | None = None,
    runner: Runner | None = None,
    version_collector: VersionCollector = collect_runtime_versions,
) -> TaskExecutionResult:
    result_root = Path(result_root).resolve(strict=False)
    resolved_checkpoint_root = (
        Path(checkpoint_root).expanduser().resolve(strict=False)
        if checkpoint_root is not None
        else None
    )
    manifest_meta, tasks = load_manifest(manifest_path, meta_path=meta_path, verify_digest=True)
    task = select_task(tasks, index=index, task_id=task_id)
    final_dir = _contained_result_path(
        result_root,
        task.output_relpath,
        label="successful task result",
    )
    if final_dir.exists():
        validate_result_directory(final_dir, task)
        return TaskExecutionResult(
            task_id=task.task_id,
            status="success",
            result_dir=str(final_dir),
            attempt_dir=None,
            skipped=True,
        )

    attempt_id = uuid.uuid4().hex
    attempt_work = work_root / task.task_id / attempt_id
    output_root = attempt_work / "runs"
    try:
        _verify_task_pins(task)
        preflight_evidence = _verify_preflight_report(
            task,
            manifest_meta,
            preflight_report_path,
            environment_manifest_path=environment_manifest_path,
        )
        _verify_execution_target(task, site_id=site_id)
        resolved_gate_registry = discover_gate_registry(repo_root, gate_registry_path)
        guard_task(task, resolved_gate_registry)

        config_path = _verify_source_config(task, repo_root=repo_root)
        code_versions = _verify_code(
            task,
            repo_root=repo_root,
            version_collector=version_collector,
        )
        _verify_environment(
            task,
            environment_lock_sha256,
            environment_manifest_path=environment_manifest_path,
        )
    except CampaignError as exc:
        if exc.code == "E_CAMPAIGN_PREFLIGHT_EXPIRED":
            resolved_report = preflight_report_path or (
                Path(os.environ["MODSSC_PREFLIGHT_REPORT"])
                if os.environ.get("MODSSC_PREFLIGHT_REPORT")
                else None
            )
            if resolved_report is None:  # Defensive: expiry implies a readable report.
                raise CampaignError(
                    "E_CAMPAIGN_PREFLIGHT_INVALID",
                    "expired preflight has no report path",
                ) from exc
            _publish_authorization_expired(
                result_root=result_root,
                task=task,
                event_id=attempt_id,
                report_path=resolved_report,
                site_id=site_id,
            )
        else:
            _publish_failure(
                result_root=result_root,
                task=task,
                attempt_id=attempt_id,
                attempt_work=attempt_work,
                run_json_path=None,
                error=exc,
                site_id=site_id,
                failure_phase="precondition",
            )
        raise

    lock_dir = _contained_result_path(
        result_root,
        Path("locks") / f"{task.task_id}.lock",
        label="task lock",
    )
    task_lock = _acquire_lock(
        lock_dir,
        task,
        site_id=site_id,
        reclaim_stale_after=reclaim_stale_lock_after,
    )
    failure_published = False
    try:
        attempt_work.mkdir(parents=True, exist_ok=False)
        if final_dir.exists():
            validate_result_directory(final_dir, task)
            return TaskExecutionResult(
                task_id=task.task_id,
                status="success",
                result_dir=str(final_dir),
                attempt_dir=None,
                skipped=True,
            )
        raw = load_yaml(config_path)
        seeded = apply_global_seed(
            raw,
            seed=task.seed,
            run_name=f"campaign-{task.task_id[:16]}",
            seeded_sections=(None if task.seeded_sections is None else list(task.seeded_sections)),
        )
        run_block = seeded.get("run")
        if not isinstance(run_block, dict):
            raise CampaignError("E_CAMPAIGN_EFFECTIVE_CONFIG", "run must be a mapping")
        run_block["output_dir"] = str(output_root)
        run_block["model_seed"] = int(task.model_seed)
        if "seeds" in run_block:
            raise CampaignError(
                "E_CAMPAIGN_EFFECTIVE_CONFIG", "effective task must not contain run.seeds"
            )
        _inject_and_verify_partition_replay(task, seeded, repo_root=repo_root)
        cfg = ExperimentConfig.from_dict(seeded)
        if (
            cfg.run.seed != task.seed
            or cfg.run.seeds is not None
            or cfg.run.model_seed != task.model_seed
        ):
            raise CampaignError(
                "E_CAMPAIGN_EFFECTIVE_CONFIG",
                "effective task does not have exactly one run seed and its manifest model seed",
            )
        _verify_effective_sampling_seeds(task, cfg)
        dataset_content_proof = _verify_dataset_content_state(
            seeded,
            task,
            preflight_evidence,
        )
        effective_config_path = attempt_work / "effective.yaml"
        dump_yaml(seeded, effective_config_path)
        active_runner = runner or _default_runner
        restored_checkpoint = None
        if resolved_checkpoint_root is not None:
            restored_checkpoint = restore_checkpoint(
                resolved_checkpoint_root,
                task,
            )
            if restored_checkpoint.resumed:
                archive_continue_marker(
                    resolved_checkpoint_root,
                    task,
                    attempt_id=attempt_id,
                    reason="resumed",
                )
        try:
            if restored_checkpoint is None:
                result = active_runner(effective_config_path, raw=seeded, cfg=cfg)
            else:
                with (
                    _checkpoint_runtime_environment(
                        task=task,
                        workspace=restored_checkpoint.workspace,
                        resumed=restored_checkpoint.resumed,
                    ),
                    _planned_continuation_signal(enabled=True),
                ):
                    result = active_runner(effective_config_path, raw=seeded, cfg=cfg)
        except PlannedContinuation as exc:
            if resolved_checkpoint_root is None or restored_checkpoint is None:
                raise CampaignError(
                    "E_CAMPAIGN_CHECKPOINT_SIGNAL",
                    "planned continuation requires --checkpoint-root",
                ) from exc
            checkpoint = publish_checkpoint(
                resolved_checkpoint_root,
                task,
                workspace=restored_checkpoint.workspace,
                attempt_id=attempt_id,
                site_id=site_id,
            )
            attempt_dir = _publish_continuation_attempt(
                result_root=result_root,
                task=task,
                attempt_id=attempt_id,
                site_id=site_id,
                signal_number=exc.signum,
                checkpoint_payload_sha256=checkpoint.payload_sha256,
                checkpoint_manifest_sha256=checkpoint.checkpoint_manifest_sha256,
            )
            return TaskExecutionResult(
                task_id=task.task_id,
                status="continuation",
                result_dir=None,
                attempt_dir=str(attempt_dir),
                skipped=False,
            )
        except Exception as exc:
            run_json_path = _find_run_json(output_root)
            _publish_failure(
                result_root=result_root,
                task=task,
                attempt_id=attempt_id,
                attempt_work=attempt_work,
                run_json_path=run_json_path,
                error=exc,
                site_id=site_id,
            )
            failure_published = True
            raise CampaignError(
                "E_CAMPAIGN_TASK_FAILED", f"task {task.task_id} failed: {exc}"
            ) from exc

        run_json_path = Path(result.run_json_path)
        run_dir = Path(result.run_dir)
        if getattr(result, "code", 1) != 0:
            error = CampaignError(
                "E_CAMPAIGN_RUNNER_NONZERO", f"runner returned {getattr(result, 'code', None)}"
            )
            _publish_failure(
                result_root=result_root,
                task=task,
                attempt_id=attempt_id,
                attempt_work=attempt_work,
                run_json_path=run_json_path if run_json_path.is_file() else None,
                error=error,
                site_id=site_id,
            )
            failure_published = True
            raise CampaignError("E_CAMPAIGN_TASK_FAILED", f"task {task.task_id} returned non-zero")
        payload = _read_json(run_json_path)
        validate_run_payload(payload)
        dataset_content_proof = _verify_dataset_content_state(
            seeded,
            task,
            preflight_evidence,
        )
        published = _publish_success(
            result_root=result_root,
            task=task,
            attempt_id=attempt_id,
            run_dir=run_dir,
            effective_config_path=effective_config_path,
            site_id=site_id,
            code_versions=code_versions,
            dataset_content_proof=dataset_content_proof,
        )
        return TaskExecutionResult(
            task_id=task.task_id,
            status="success",
            result_dir=str(published),
            attempt_dir=None,
            skipped=False,
        )
    except Exception as exc:
        if not failure_published:
            _publish_failure(
                result_root=result_root,
                task=task,
                attempt_id=attempt_id,
                attempt_work=attempt_work,
                run_json_path=_find_run_json(output_root),
                error=exc,
                site_id=site_id,
            )
        raise
    finally:
        _release_lock(task_lock)
