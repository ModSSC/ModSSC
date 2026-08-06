from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from bench.utils.hashing import hash_any, stable_json_dumps
from bench.utils.io import atomic_write_json

from .errors import CampaignError
from .identifiers import validate_safe_identifier
from .models import CampaignTask

_IDENTITY_FIELDS_V1 = (
    "schema_version",
    "campaign_id",
    "track",
    "protocol_id",
    "config_path",
    "source_config_sha256",
    "method_profile",
    "label_budget",
    "required_seed_count",
    "seed",
    "data_seed",
    "split_seed",
    "model_seed",
    "seeded_sections",
    "method_id",
    "method_kind",
    "dataset_id",
    "modality",
    "regime",
    "resource_profile",
    "assigned_site",
    "expected_git_sha",
    "expected_git_diff_sha256",
    "environment_lock_sha256",
    "dataset_lock_sha256",
    "expected_dataset_fingerprint",
    "expected_dataset_content_sha256",
    "dataset_request_sha256",
    "split_request_sha256",
    "expected_split_fingerprint",
    "fidelity_status",
)
_IDENTITY_FIELDS_V2 = (
    *_IDENTITY_FIELDS_V1[:12],
    "sampling_component_seeds",
    *_IDENTITY_FIELDS_V1[12:],
)
_IDENTITY_FIELDS_V3 = (*_IDENTITY_FIELDS_V2, "partition_selection")
_IDENTITY_FIELDS_V4 = (
    *_IDENTITY_FIELDS_V3,
    "claim_scope_id",
    "campaign_stage",
    "claim_eligible",
    "gate_policy_id",
    "gate_policy_sha256",
)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def task_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = payload.get("schema_version")
    if schema_version == 1:
        fields = _IDENTITY_FIELDS_V1
    elif schema_version == 2:
        fields = _IDENTITY_FIELDS_V2
    elif schema_version == 3:
        fields = _IDENTITY_FIELDS_V3
    elif schema_version == 4:
        fields = _IDENTITY_FIELDS_V4
    else:
        raise CampaignError(
            "E_CAMPAIGN_MANIFEST_SCHEMA",
            "task identity requires schema_version 1, 2, 3, or 4",
        )
    return {key: payload.get(key) for key in fields}


def derive_task_id(payload: Mapping[str, Any]) -> str:
    return hash_any(task_identity(payload))


def derive_row_sha256(payload: Mapping[str, Any]) -> str:
    return hash_any({key: value for key, value in payload.items() if key != "row_sha256"})


def finalize_task_row(payload: dict[str, Any], *, task_index: int) -> CampaignTask:
    row = dict(payload)
    has_governance = all(
        key in row
        for key in (
            "claim_scope_id",
            "campaign_stage",
            "claim_eligible",
            "gate_policy_id",
            "gate_policy_sha256",
        )
    )
    if has_governance:
        row["schema_version"] = 4
        row.setdefault("partition_selection", None)
    else:
        row["schema_version"] = 3 if row.get("partition_selection") is not None else 2
    row["task_index"] = int(task_index)
    task_id = derive_task_id(row)
    row["task_id"] = task_id
    row["output_relpath"] = f"tasks/{task_id[:2]}/{task_id}"
    row["row_sha256"] = derive_row_sha256(row)
    task = CampaignTask.from_dict(row)
    validate_task(task)
    return task


def validate_task(task: CampaignTask) -> None:
    validate_safe_identifier(
        task.campaign_id,
        field="campaign_id",
        code="E_CAMPAIGN_MANIFEST_SCHEMA",
    )
    validate_safe_identifier(
        task.assigned_site,
        field="assigned_site",
        code="E_CAMPAIGN_MANIFEST_SCHEMA",
    )
    validate_safe_identifier(
        task.resource_profile,
        field="resource_profile",
        code="E_CAMPAIGN_MANIFEST_SCHEMA",
    )
    if task.required_seed_count <= 0:
        raise CampaignError(
            "E_CAMPAIGN_MANIFEST_SCHEMA",
            "required_seed_count must be a positive integer",
        )
    expected_output_relpath = f"tasks/{task.task_id[:2]}/{task.task_id}"
    if task.output_relpath != expected_output_relpath:
        raise CampaignError(
            "E_CAMPAIGN_OUTPUT_PATH_INVALID",
            f"output_relpath must equal {expected_output_relpath!r}",
        )
    payload = task.to_dict()
    expected_task_id = derive_task_id(payload)
    if task.task_id != expected_task_id:
        raise CampaignError(
            "E_CAMPAIGN_TASK_ID_MISMATCH",
            f"task_id mismatch at index {task.task_index}: {task.task_id}",
        )
    expected_row_hash = derive_row_sha256(payload)
    if task.row_sha256 != expected_row_hash:
        raise CampaignError(
            "E_CAMPAIGN_ROW_HASH_MISMATCH",
            f"row hash mismatch at index {task.task_index}: {task.task_id}",
        )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    _atomic_write_text(path, text)


def write_manifest(
    tasks: Iterable[CampaignTask],
    *,
    output_dir: Path,
    campaign_id: str,
    spec_sha256: str,
    expected_git_sha: str,
    expected_git_diff_sha256: str | None,
    environment_lock_sha256: str | None = None,
    manifest_filename: str = "manifest.jsonl",
    meta_filename: str = "manifest.meta.json",
    profile_dirname: str = "profiles",
    source_manifest_sha256: str | None = None,
    release_evidence: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    ordered = list(tasks)
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    previous_index = -1
    for task in ordered:
        if task.task_index in seen_indices or task.task_index <= previous_index:
            raise CampaignError(
                "E_CAMPAIGN_MANIFEST_ORDER",
                "task_index values must be unique and strictly increasing",
            )
        seen_indices.add(task.task_index)
        previous_index = task.task_index
        validate_task(task)
        if task.task_id in seen_ids:
            raise CampaignError("E_CAMPAIGN_DUPLICATE_TASK", f"duplicate task_id: {task.task_id}")
        seen_ids.add(task.task_id)

    commits = {task.expected_git_sha for task in ordered}
    diffs = {task.expected_git_diff_sha256 for task in ordered}
    environments = {task.environment_lock_sha256 for task in ordered}
    claim_scopes = {task.claim_scope_id for task in ordered}
    campaign_stages = {task.campaign_stage for task in ordered}
    claim_eligibility = {task.claim_eligible for task in ordered}
    gate_policy_ids = {task.gate_policy_id for task in ordered}
    gate_policy_digests = {task.gate_policy_sha256 for task in ordered}
    campaigns = {task.campaign_id for task in ordered}
    if campaigns and campaigns != {campaign_id}:
        raise CampaignError("E_CAMPAIGN_MANIFEST_MIXED", "manifest mixes campaign identifiers")
    if len(commits) > 1 or (commits and commits != {expected_git_sha}):
        raise CampaignError("E_CAMPAIGN_MANIFEST_MIXED", "manifest mixes Git revisions")
    if len(diffs) > 1 or (diffs and diffs != {expected_git_diff_sha256}):
        raise CampaignError("E_CAMPAIGN_MANIFEST_MIXED", "manifest mixes worktree fingerprints")
    if len(environments) > 1:
        raise CampaignError("E_CAMPAIGN_MANIFEST_MIXED", "manifest mixes environments")
    for values, label in (
        (claim_scopes, "claim scopes"),
        (campaign_stages, "campaign stages"),
        (claim_eligibility, "claim eligibility"),
        (gate_policy_ids, "gate policy identifiers"),
        (gate_policy_digests, "gate policy digests"),
    ):
        if len(values) > 1:
            raise CampaignError("E_CAMPAIGN_MANIFEST_MIXED", f"manifest mixes {label}")
    inferred_environment = next(iter(environments), environment_lock_sha256)
    if (
        environment_lock_sha256 is not None
        and inferred_environment is not None
        and inferred_environment != environment_lock_sha256
    ):
        raise CampaignError("E_CAMPAIGN_MANIFEST_MIXED", "manifest environment differs")

    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [stable_json_dumps(task.to_dict()) for task in ordered]
    manifest_bytes = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
    manifest_path = output_dir / manifest_filename
    _atomic_write_text(manifest_path, manifest_bytes.decode("utf-8"))
    manifest_sha256 = sha256_bytes(manifest_bytes)

    profile_indices: dict[str, list[int]] = defaultdict(list)
    for task in ordered:
        profile_indices[task.resource_profile].append(task.task_index)

    profile_hashes: dict[str, str] = {}
    profiles_dir = output_dir / profile_dirname
    for profile, indices in sorted(profile_indices.items()):
        text = "".join(f"{index}\n" for index in indices)
        path = profiles_dir / f"{profile}.indices"
        _atomic_write_text(path, text)
        profile_hashes[profile] = sha256_bytes(text.encode("utf-8"))

    counts_by_method = dict(sorted(Counter(task.method_id for task in ordered).items()))
    counts_by_profile = dict(sorted(Counter(task.resource_profile for task in ordered).items()))
    counts_by_site = dict(sorted(Counter(task.assigned_site for task in ordered).items()))
    meta = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "task_count": len(ordered),
        "manifest_sha256": manifest_sha256,
        "spec_sha256": spec_sha256,
        "expected_git_sha": expected_git_sha,
        "expected_git_diff_sha256": expected_git_diff_sha256,
        "environment_lock_sha256": inferred_environment,
        "source_manifest_sha256": source_manifest_sha256,
        "counts_by_method": counts_by_method,
        "counts_by_profile": counts_by_profile,
        "counts_by_site": counts_by_site,
        "profile_index_sha256": profile_hashes,
        "claim_scope_id": next(iter(claim_scopes), None),
        "campaign_stage": next(iter(campaign_stages), None),
        "claim_eligible": next(iter(claim_eligibility), None),
        "gate_policy_id": next(iter(gate_policy_ids), None),
        "gate_policy_sha256": next(iter(gate_policy_digests), None),
    }
    if release_evidence is not None:
        meta["release_evidence"] = dict(release_evidence)
    meta_path = output_dir / meta_filename
    atomic_write_json(meta_path, meta)
    return manifest_path, meta_path, meta


def _load_meta(meta_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(
            "E_CAMPAIGN_META_INVALID", f"cannot read campaign metadata: {meta_path}"
        ) from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise CampaignError("E_CAMPAIGN_META_INVALID", "unsupported campaign metadata")
    return raw


def load_manifest(
    manifest_path: Path,
    *,
    meta_path: Path | None = None,
    verify_digest: bool = True,
) -> tuple[dict[str, Any], list[CampaignTask]]:
    manifest_path = Path(manifest_path)
    if meta_path is None:
        meta_name = (
            "manifest.meta.json"
            if manifest_path.name == "manifest.jsonl"
            else f"{manifest_path.stem}.meta.json"
        )
        meta_path = manifest_path.with_name(meta_name)
    meta = _load_meta(meta_path)
    if verify_digest:
        actual = sha256_file(manifest_path)
        if actual != meta.get("manifest_sha256"):
            raise CampaignError(
                "E_CAMPAIGN_MANIFEST_HASH_MISMATCH",
                f"manifest digest does not match {meta_path}",
            )

    tasks: list[CampaignTask] = []
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                task = CampaignTask.from_dict(raw)
                validate_task(task)
                if tasks and task.task_index <= tasks[-1].task_index:
                    raise CampaignError(
                        "E_CAMPAIGN_MANIFEST_ORDER",
                        f"line {line_number} does not have a strictly increasing task_index",
                    )
                tasks.append(task)
    except json.JSONDecodeError as exc:
        raise CampaignError(
            "E_CAMPAIGN_MANIFEST_SCHEMA", f"invalid JSONL at line {exc.lineno}"
        ) from exc
    if len(tasks) != meta.get("task_count"):
        raise CampaignError(
            "E_CAMPAIGN_MANIFEST_COUNT",
            f"metadata expects {meta.get('task_count')} tasks, found {len(tasks)}",
        )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise CampaignError("E_CAMPAIGN_DUPLICATE_TASK", "manifest contains duplicate tasks")
    campaigns = {task.campaign_id for task in tasks}
    commits = {task.expected_git_sha for task in tasks}
    diffs = {task.expected_git_diff_sha256 for task in tasks}
    environments = {task.environment_lock_sha256 for task in tasks}
    if campaigns and campaigns != {meta.get("campaign_id")}:
        raise CampaignError("E_CAMPAIGN_MANIFEST_MIXED", "manifest mixes campaign identifiers")
    if len(commits) > 1 or (commits and commits != {meta.get("expected_git_sha")}):
        raise CampaignError("E_CAMPAIGN_MANIFEST_MIXED", "manifest mixes Git revisions")
    if len(diffs) > 1 or (diffs and diffs != {meta.get("expected_git_diff_sha256")}):
        raise CampaignError("E_CAMPAIGN_MANIFEST_MIXED", "manifest mixes worktree fingerprints")
    if len(environments) > 1 or (
        environments and environments != {meta.get("environment_lock_sha256")}
    ):
        raise CampaignError("E_CAMPAIGN_MANIFEST_MIXED", "manifest mixes environments")
    if tasks and any(task.schema_version == 4 for task in tasks):
        expected_governance = {
            "claim_scope_id": tasks[0].claim_scope_id,
            "campaign_stage": tasks[0].campaign_stage,
            "claim_eligible": tasks[0].claim_eligible,
            "gate_policy_id": tasks[0].gate_policy_id,
            "gate_policy_sha256": tasks[0].gate_policy_sha256,
        }
        if any(
            getattr(task, field) != value
            for field, value in expected_governance.items()
            for task in tasks
        ):
            raise CampaignError("E_CAMPAIGN_MANIFEST_MIXED", "manifest mixes scientific governance")
        if any(meta.get(field) != value for field, value in expected_governance.items()):
            raise CampaignError("E_CAMPAIGN_META_INVALID", "metadata scientific governance differs")
    return meta, tasks


def select_task(
    tasks: list[CampaignTask], *, index: int | None, task_id: str | None
) -> CampaignTask:
    if (index is None) == (task_id is None):
        raise CampaignError("E_CAMPAIGN_TASK_SELECTOR", "provide exactly one of index or task_id")
    if index is not None:
        for task in tasks:
            if task.task_index == index:
                return task
        raise CampaignError("E_CAMPAIGN_TASK_SELECTOR", f"unknown task index: {index}")
    assert task_id is not None
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise CampaignError("E_CAMPAIGN_TASK_SELECTOR", f"unknown task_id: {task_id}")
