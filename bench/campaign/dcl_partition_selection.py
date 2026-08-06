from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import CampaignError
from .manifest import load_manifest, sha256_bytes, sha256_file
from .models import CampaignTask
from .reconcile import materialize_reconcile_paths

_METHOD_ID = "democratic_co_learning"
_METHOD_PROFILE = "paper:zhou-goldman-2004-vote-table3"
_DATASET_ID = "vote"
_REQUIRED_SELECTION_COUNT = 20
_MAX_ITER = 20
_DIAGNOSTIC_PATH = "artifacts.method.diagnostics.pseudo_labels_added_total"
_RECONCILE_STATUSES = {
    "blocked",
    "conflict",
    "corrupt",
    "duplicate",
    "failed",
    "missing",
    "resource_blocked",
    "running",
    "stale",
    "success",
}


@dataclass(frozen=True)
class DCLPartitionSelectionResult:
    campaign_id: str
    protocol_id: str
    output_path: str
    output_sha256: str
    candidate_count: int
    evaluated_candidate_count: int
    selected_count: int
    rejected_count: int
    cutoff_seed: int


@dataclass(frozen=True)
class _CandidateEvidence:
    evaluation_rank: int
    selection_rank: int | None
    decision: str
    task_id: str
    task_row_sha256: str
    seed: int
    pseudo_labels_added_total: int
    converged: bool
    n_iter: int
    split_fingerprint: str
    run_json_sha256: str
    split_manifest_sha256: str
    split_json_sha256: str
    split_arrays_sha256: str


def _fail(code: str, message: str) -> None:
    raise CampaignError(code, message)


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(
            "E_DCL_SELECTION_INPUT",
            f"cannot read {label}: {path}",
        ) from exc
    if not isinstance(raw, dict):
        _fail("E_DCL_SELECTION_INPUT", f"{label} must be a JSON mapping: {path}")
    return raw


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("E_DCL_SELECTION_MISMATCH", f"{field} must be a mapping")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("E_DCL_SELECTION_MISMATCH", f"{field} must be a non-empty string")
    return value


def _sha256(value: Any, *, field: str) -> str:
    digest = _string(value, field=field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _fail("E_DCL_SELECTION_MISMATCH", f"{field} must be a lowercase SHA-256")
    return digest


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        _fail("E_DCL_SELECTION_MISMATCH", f"{field} must be a list of non-empty strings")
    return list(value)


def _contained_file(root: Path, relative: str, *, field: str) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        candidate = (resolved_root / relative).resolve(strict=True)
        candidate.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise CampaignError(
            "E_DCL_SELECTION_MISMATCH",
            f"{field} is missing or escapes its artifact directory",
        ) from exc
    if not candidate.is_file():
        _fail("E_DCL_SELECTION_MISMATCH", f"{field} is not a file")
    return candidate


def _meta_path(manifest_path: Path, meta_path: Path | None) -> Path:
    if meta_path is not None:
        return Path(meta_path)
    name = (
        "manifest.meta.json"
        if manifest_path.name == "manifest.jsonl"
        else f"{manifest_path.stem}.meta.json"
    )
    return manifest_path.with_name(name)


def _validate_reconcile(
    report: Mapping[str, Any],
    *,
    meta: Mapping[str, Any],
    tasks: list[CampaignTask],
) -> dict[str, Mapping[str, Any]]:
    if report.get("schema_version") != 1:
        _fail("E_DCL_SELECTION_INPUT", "reconcile report must use schema_version=1")
    if report.get("campaign_id") != meta.get("campaign_id"):
        _fail("E_DCL_SELECTION_MISMATCH", "reconcile campaign_id differs from the manifest")
    if report.get("manifest_sha256") != meta.get("manifest_sha256"):
        _fail("E_DCL_SELECTION_MISMATCH", "reconcile manifest digest differs")
    task_count = report.get("task_count")
    if isinstance(task_count, bool) or task_count != len(tasks):
        _fail("E_DCL_SELECTION_MISMATCH", "reconcile task_count differs from the manifest")

    raw_states = report.get("tasks")
    if not isinstance(raw_states, list):
        _fail("E_DCL_SELECTION_INPUT", "reconcile tasks must be a list")
    task_by_id = {task.task_id: task for task in tasks}
    states: dict[str, Mapping[str, Any]] = {}
    for position, raw_state in enumerate(raw_states):
        state = _mapping(raw_state, field=f"reconcile.tasks[{position}]")
        task_id = _string(state.get("task_id"), field=f"reconcile.tasks[{position}].task_id")
        if task_id in states:
            _fail("E_DCL_SELECTION_DUPLICATE", f"duplicate reconciled task_id: {task_id}")
        task = task_by_id.get(task_id)
        if task is None:
            _fail("E_DCL_SELECTION_MISMATCH", f"unknown reconciled task_id: {task_id}")
        if (
            state.get("task_index") != task.task_index
            or state.get("method_id") != task.method_id
            or state.get("dataset_id") != task.dataset_id
            or state.get("resource_profile") != task.resource_profile
            or state.get("assigned_site") != task.assigned_site
        ):
            _fail(
                "E_DCL_SELECTION_MISMATCH",
                f"reconciled task metadata differs for {task_id}",
            )
        status = state.get("status")
        if status not in _RECONCILE_STATUSES:
            _fail(
                "E_DCL_SELECTION_MISMATCH",
                f"unsupported reconciled status for {task_id}: {status!r}",
            )
        _string_list(state.get("result_dirs"), field=f"{task_id}.result_dirs")
        _string_list(state.get("run_json_paths"), field=f"{task_id}.run_json_paths")
        _string_list(state.get("run_json_sha256"), field=f"{task_id}.run_json_sha256")
        states[task_id] = state

    expected_ids = set(task_by_id)
    if set(states) != expected_ids:
        missing = sorted(expected_ids - set(states))
        _fail(
            "E_DCL_SELECTION_MISMATCH",
            f"reconcile report does not cover the manifest; missing={missing[:3]}",
        )
    return states


def _target_tasks(
    tasks: list[CampaignTask],
    *,
    protocol_id: str | None,
) -> tuple[str, list[CampaignTask]]:
    vote_tasks = [
        task for task in tasks if task.method_id == _METHOD_ID and task.dataset_id == _DATASET_ID
    ]
    if protocol_id is None:
        protocols = {task.protocol_id for task in vote_tasks}
        if len(protocols) != 1 or None in protocols:
            _fail(
                "E_DCL_SELECTION_INPUT",
                "cannot infer one DCL Vote protocol; provide --protocol-id",
            )
        selected_protocol = next(iter(protocols))
        assert selected_protocol is not None
    else:
        if not protocol_id:
            _fail("E_DCL_SELECTION_INPUT", "protocol_id must not be empty")
        selected_protocol = protocol_id

    selected = sorted(
        (task for task in vote_tasks if task.protocol_id == selected_protocol),
        key=lambda task: task.seed,
    )
    if not selected:
        _fail(
            "E_DCL_SELECTION_INPUT",
            f"no DCL Vote tasks found for protocol {selected_protocol!r}",
        )
    for task in selected:
        if (
            task.track != "paper"
            or task.method_kind != "inductive"
            or task.method_profile != _METHOD_PROFILE
            or task.fidelity_status != "not_claimable"
        ):
            _fail(
                "E_DCL_SELECTION_MISMATCH",
                f"task {task.task_id} is not a DCL Vote screening task",
            )
        if not task.expected_dataset_fingerprint or not task.expected_split_fingerprint:
            _fail(
                "E_DCL_SELECTION_MISMATCH",
                f"task {task.task_id} does not pin dataset and split fingerprints",
            )

    seeds = [task.seed for task in selected]
    if len(seeds) != len(set(seeds)):
        _fail("E_DCL_SELECTION_DUPLICATE", "DCL Vote candidate seeds are not unique")
    split_fingerprints = [str(task.expected_split_fingerprint) for task in selected]
    if len(split_fingerprints) != len(set(split_fingerprints)):
        _fail(
            "E_DCL_SELECTION_DUPLICATE",
            "DCL Vote candidate split fingerprints are not unique",
        )
    required_counts = {task.required_seed_count for task in selected}
    if required_counts != {len(selected)}:
        _fail(
            "E_DCL_SELECTION_MISMATCH",
            "DCL Vote candidate count differs from required_seed_count",
        )
    return selected_protocol, selected


def _validate_state_references(
    target_tasks: list[CampaignTask],
    states: Mapping[str, Mapping[str, Any]],
) -> None:
    result_paths: set[Path] = set()
    run_paths: set[Path] = set()
    run_digests: set[str] = set()
    for task in target_tasks:
        state = states[task.task_id]
        result_values = _string_list(state.get("result_dirs"), field=f"{task.task_id}.result_dirs")
        run_values = _string_list(
            state.get("run_json_paths"),
            field=f"{task.task_id}.run_json_paths",
        )
        digest_values = _string_list(
            state.get("run_json_sha256"),
            field=f"{task.task_id}.run_json_sha256",
        )
        if state.get("status") != "success":
            if result_values or run_values or digest_values:
                _fail(
                    "E_DCL_SELECTION_MISMATCH",
                    f"non-success task {task.task_id} exposes successful artifact references",
                )
            continue
        if len(result_values) != 1 or len(run_values) != 1 or len(digest_values) != 1:
            _fail(
                "E_DCL_SELECTION_DUPLICATE",
                f"successful task {task.task_id} must have exactly one result bundle",
            )
        try:
            result_path = Path(result_values[0]).resolve(strict=True)
            run_path = Path(run_values[0]).resolve(strict=True)
        except OSError as exc:
            raise CampaignError(
                "E_DCL_SELECTION_MISMATCH",
                f"successful artifact path is missing for {task.task_id}",
            ) from exc
        run_digest = _sha256(digest_values[0], field=f"{task.task_id}.run_json_sha256[0]")
        if result_path in result_paths or run_path in run_paths or run_digest in run_digests:
            _fail(
                "E_DCL_SELECTION_DUPLICATE",
                f"successful artifact reference is reused by {task.task_id}",
            )
        result_paths.add(result_path)
        run_paths.add(run_path)
        run_digests.add(run_digest)


def _validate_bundle(
    task: CampaignTask,
    state: Mapping[str, Any],
    *,
    evaluation_rank: int,
    selection_rank: int | None,
) -> _CandidateEvidence:
    result_value = _string_list(
        state.get("result_dirs"),
        field=f"{task.task_id}.result_dirs",
    )[0]
    run_value = _string_list(
        state.get("run_json_paths"),
        field=f"{task.task_id}.run_json_paths",
    )[0]
    reconciled_digest = _sha256(
        _string_list(
            state.get("run_json_sha256"),
            field=f"{task.task_id}.run_json_sha256",
        )[0],
        field=f"{task.task_id}.run_json_sha256[0]",
    )
    try:
        result_dir = Path(result_value).resolve(strict=True)
        run_json_path = Path(run_value).resolve(strict=True)
    except OSError as exc:
        raise CampaignError(
            "E_DCL_SELECTION_MISMATCH",
            f"successful artifact path is missing for {task.task_id}",
        ) from exc
    expected_suffix = Path(task.output_relpath).parts
    if tuple(result_dir.parts[-len(expected_suffix) :]) != expected_suffix:
        _fail(
            "E_DCL_SELECTION_MISMATCH",
            f"result directory does not match output_relpath for {task.task_id}",
        )
    expected_run_path = (result_dir / "run" / "run.json").resolve(strict=False)
    if run_json_path != expected_run_path or not run_json_path.is_file():
        _fail("E_DCL_SELECTION_MISMATCH", f"run.json path differs for {task.task_id}")

    success_path = _contained_file(result_dir, "SUCCESS.json", field="SUCCESS.json")
    envelope_path = _contained_file(result_dir, "task.json", field="task.json")
    success = _read_mapping(success_path, label="SUCCESS.json")
    envelope = _read_mapping(envelope_path, label="task.json")
    run_payload = _read_mapping(run_json_path, label="run.json")
    actual_run_digest = sha256_file(run_json_path)
    if actual_run_digest != reconciled_digest:
        _fail("E_DCL_SELECTION_MISMATCH", f"reconciled run digest differs for {task.task_id}")
    if (
        success.get("schema_version") != 1
        or success.get("status") != "success"
        or success.get("task_id") != task.task_id
        or success.get("row_sha256") != task.row_sha256
        or success.get("run_json_sha256") != actual_run_digest
        or success.get("dataset_content_sha256") != task.expected_dataset_content_sha256
    ):
        _fail("E_DCL_SELECTION_MISMATCH", f"SUCCESS.json differs for {task.task_id}")
    effective_name = _string(
        success.get("effective_config_path"),
        field=f"{task.task_id}.effective_config_path",
    )
    effective_digest = _sha256(
        success.get("effective_config_sha256"),
        field=f"{task.task_id}.effective_config_sha256",
    )
    effective_path = _contained_file(
        result_dir,
        effective_name,
        field="effective configuration",
    )
    if sha256_file(effective_path) != effective_digest:
        _fail(
            "E_DCL_SELECTION_MISMATCH",
            f"effective configuration digest differs for {task.task_id}",
        )
    if (
        envelope.get("schema_version") != 1
        or envelope.get("task") != task.to_dict()
        or envelope.get("environment_lock_sha256") != task.environment_lock_sha256
        or (task.assigned_site != "any" and envelope.get("site_id") != task.assigned_site)
    ):
        _fail("E_DCL_SELECTION_MISMATCH", f"task envelope differs for {task.task_id}")

    run = _mapping(run_payload.get("run"), field=f"{task.task_id}.run")
    task_info = _mapping(run_payload.get("task_info"), field=f"{task.task_id}.task_info")
    versions = _mapping(run_payload.get("versions"), field=f"{task.task_id}.versions")
    if run.get("status") != "success" or run.get("seed") != task.seed:
        _fail("E_DCL_SELECTION_MISMATCH", f"run identity differs for {task.task_id}")
    if (
        task_info.get("method_id") != task.method_id
        or task_info.get("dataset_id") != task.dataset_id
        or task_info.get("method_kind") != task.method_kind
    ):
        _fail("E_DCL_SELECTION_MISMATCH", f"run task identity differs for {task.task_id}")
    if (
        versions.get("git_sha") != task.expected_git_sha
        or versions.get("git_dirty") is not False
        or versions.get("git_diff_sha256") != task.expected_git_diff_sha256
    ):
        _fail("E_DCL_SELECTION_MISMATCH", f"run code identity differs for {task.task_id}")

    artifacts = _mapping(run_payload.get("artifacts"), field=f"{task.task_id}.artifacts")
    dataset = _mapping(artifacts.get("dataset"), field=f"{task.task_id}.artifacts.dataset")
    sampling = _mapping(artifacts.get("sampling"), field=f"{task.task_id}.artifacts.sampling")
    method = _mapping(artifacts.get("method"), field=f"{task.task_id}.artifacts.method")
    dataset_fingerprint = _string(
        dataset.get("fingerprint"),
        field=f"{task.task_id}.dataset_fingerprint",
    )
    if dataset_fingerprint != task.expected_dataset_fingerprint:
        _fail("E_DCL_SELECTION_MISMATCH", f"dataset fingerprint differs for {task.task_id}")
    if (
        task.expected_dataset_content_sha256 is not None
        and dataset.get("content_sha256") != task.expected_dataset_content_sha256
    ):
        _fail("E_DCL_SELECTION_MISMATCH", f"dataset content digest differs for {task.task_id}")
    if task.expected_dataset_content_sha256 is not None:
        content_manifest_digest = _sha256(
            dataset.get("content_manifest_sha256"),
            field=f"{task.task_id}.dataset_content_manifest_sha256",
        )
        content_proof = _mapping(
            envelope.get("dataset_content_proof"),
            field=f"{task.task_id}.dataset_content_proof",
        )
        if (
            content_proof.get("content_sha256") != task.expected_dataset_content_sha256
            or content_proof.get("content_manifest_sha256") != content_manifest_digest
        ):
            _fail(
                "E_DCL_SELECTION_MISMATCH",
                f"dataset content proof differs for {task.task_id}",
            )
        for field in (
            "cache_state_sha256",
            "cache_fingerprint",
            "preflight_report_sha256",
        ):
            _string(
                content_proof.get(field),
                field=f"{task.task_id}.dataset_content_proof.{field}",
            )
    split_fingerprint = _string(
        sampling.get("split_fingerprint"),
        field=f"{task.task_id}.split_fingerprint",
    )
    if split_fingerprint != task.expected_split_fingerprint:
        _fail("E_DCL_SELECTION_MISMATCH", f"split fingerprint differs for {task.task_id}")
    if method.get("profile") != task.method_profile:
        _fail("E_DCL_SELECTION_MISMATCH", f"method profile differs for {task.task_id}")

    diagnostics = _mapping(
        method.get("diagnostics"),
        field=f"{task.task_id}.artifacts.method.diagnostics",
    )
    pseudo_labels = diagnostics.get("pseudo_labels_added_total")
    if isinstance(pseudo_labels, bool) or not isinstance(pseudo_labels, int) or pseudo_labels < 0:
        _fail(
            "E_DCL_SELECTION_MISMATCH",
            f"{_DIAGNOSTIC_PATH} must be a non-negative integer for {task.task_id}",
        )
    if diagnostics.get("converged") is not True:
        _fail(
            "E_DCL_SELECTION_UNRESOLVED",
            f"DCL did not converge for seed {task.seed}; rerun the same partition",
        )
    n_iter = diagnostics.get("n_iter")
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 0:
        _fail(
            "E_DCL_SELECTION_MISMATCH",
            f"artifacts.method.diagnostics.n_iter is invalid for {task.task_id}",
        )
    if n_iter >= _MAX_ITER:
        _fail(
            "E_DCL_SELECTION_UNRESOLVED",
            f"DCL reached max_iter for seed {task.seed}; resolve the same partition",
        )

    replay = _mapping(
        sampling.get("replay"),
        field=f"{task.task_id}.artifacts.sampling.replay",
    )
    if replay.get("format") != "modssc.sampling.storage.v1":
        _fail("E_DCL_SELECTION_MISMATCH", f"split replay format differs for {task.task_id}")
    replay_relative = _string(replay.get("path"), field=f"{task.task_id}.replay.path")
    manifest_relative = _string(
        replay.get("manifest"),
        field=f"{task.task_id}.replay.manifest",
    )
    split_manifest_digest = _sha256(
        replay.get("manifest_sha256"),
        field=f"{task.task_id}.replay.manifest_sha256",
    )
    replay_dir = (run_json_path.parent / replay_relative).resolve(strict=False)
    try:
        replay_dir.relative_to(run_json_path.parent.resolve(strict=True))
    except ValueError as exc:
        raise CampaignError(
            "E_DCL_SELECTION_MISMATCH",
            f"split replay escapes run directory for {task.task_id}",
        ) from exc
    manifest_path = _contained_file(
        replay_dir,
        manifest_relative,
        field="split replay manifest",
    )
    if sha256_file(manifest_path) != split_manifest_digest:
        _fail("E_DCL_SELECTION_MISMATCH", f"split manifest digest differs for {task.task_id}")
    split_manifest = _read_mapping(manifest_path, label="split replay manifest")
    if (
        split_manifest.get("schema_version") != 1
        or split_manifest.get("format") != replay.get("format")
        or split_manifest.get("dataset_fingerprint") != dataset_fingerprint
        or split_manifest.get("split_fingerprint") != split_fingerprint
    ):
        _fail("E_DCL_SELECTION_MISMATCH", f"split manifest differs for {task.task_id}")
    files = _mapping(
        split_manifest.get("files"),
        field=f"{task.task_id}.split_manifest.files",
    )
    artifact_digests: dict[str, str] = {}
    for artifact_name in ("split.json", "arrays.npz"):
        record = _mapping(
            files.get(artifact_name),
            field=f"{task.task_id}.split_manifest.files.{artifact_name}",
        )
        artifact_digest = _sha256(
            record.get("sha256"),
            field=f"{task.task_id}.split_manifest.files.{artifact_name}.sha256",
        )
        artifact_path = _contained_file(
            replay_dir,
            artifact_name,
            field=f"split artifact {artifact_name}",
        )
        if sha256_file(artifact_path) != artifact_digest:
            _fail(
                "E_DCL_SELECTION_MISMATCH",
                f"split artifact {artifact_name} digest differs for {task.task_id}",
            )
        artifact_digests[artifact_name] = artifact_digest
    split_metadata = _read_mapping(
        _contained_file(replay_dir, "split.json", field="split metadata"),
        label="split metadata",
    )
    if (
        split_metadata.get("dataset_fingerprint") != dataset_fingerprint
        or split_metadata.get("split_fingerprint") != split_fingerprint
    ):
        _fail("E_DCL_SELECTION_MISMATCH", f"split metadata differs for {task.task_id}")

    accepted = pseudo_labels > 0
    return _CandidateEvidence(
        evaluation_rank=evaluation_rank,
        selection_rank=selection_rank if accepted else None,
        decision="accepted" if accepted else "rejected_no_pseudo_labels",
        task_id=task.task_id,
        task_row_sha256=task.row_sha256,
        seed=task.seed,
        pseudo_labels_added_total=pseudo_labels,
        converged=True,
        n_iter=n_iter,
        split_fingerprint=split_fingerprint,
        run_json_sha256=actual_run_digest,
        split_manifest_sha256=split_manifest_digest,
        split_json_sha256=artifact_digests["split.json"],
        split_arrays_sha256=artifact_digests["arrays.npz"],
    )


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    path = Path(path)
    if path.exists() or path.is_symlink():
        _fail("E_DCL_SELECTION_OUTPUT_EXISTS", f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CampaignError(
                "E_DCL_SELECTION_OUTPUT_EXISTS",
                f"immutable output already exists: {path}",
            ) from exc
        temporary.unlink()
        temporary = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_bytes(encoded)


def select_dcl_vote_partitions(
    manifest_path: Path,
    *,
    reconcile_path: Path,
    output_path: Path,
    meta_path: Path | None = None,
    protocol_id: str | None = None,
) -> DCLPartitionSelectionResult:
    """Select the first 20 converged, pseudo-label-active DCL Vote partitions.

    Selection is fail-closed and follows numeric seed order from the immutable
    manifest. Only method diagnostics and artifact identity fields are consulted;
    evaluation outcomes and test-derived statistics are outside this function's
    access path.
    """

    manifest_path = Path(manifest_path)
    reconcile_path = Path(reconcile_path)
    output_path = Path(output_path)
    resolved_meta_path = _meta_path(manifest_path, meta_path)
    if output_path.exists() or output_path.is_symlink():
        _fail(
            "E_DCL_SELECTION_OUTPUT_EXISTS",
            f"immutable output already exists: {output_path}",
        )
    meta, tasks = load_manifest(
        manifest_path,
        meta_path=resolved_meta_path,
        verify_digest=True,
    )
    report = materialize_reconcile_paths(
        reconcile_path,
        _read_mapping(reconcile_path, label="reconcile report"),
    )
    states = _validate_reconcile(report, meta=meta, tasks=tasks)
    selected_protocol, target_tasks = _target_tasks(tasks, protocol_id=protocol_id)
    _validate_state_references(target_tasks, states)

    evidence: list[_CandidateEvidence] = []
    selected_count = 0
    actual_split_fingerprints: set[str] = set()
    actual_split_arrays: set[str] = set()
    for task in target_tasks:
        state = states[task.task_id]
        if state.get("status") != "success":
            _fail(
                "E_DCL_SELECTION_PREFIX_INCOMPLETE",
                (
                    f"seed {task.seed} is {state.get('status')!r} before the "
                    "20th accepted partition; do not skip it"
                ),
            )
        candidate = _validate_bundle(
            task,
            state,
            evaluation_rank=len(evidence) + 1,
            selection_rank=selected_count + 1,
        )
        if candidate.split_fingerprint in actual_split_fingerprints:
            _fail(
                "E_DCL_SELECTION_DUPLICATE",
                f"replayed split fingerprint is duplicated at seed {task.seed}",
            )
        if candidate.split_arrays_sha256 in actual_split_arrays:
            _fail(
                "E_DCL_SELECTION_DUPLICATE",
                f"replayed split arrays are duplicated at seed {task.seed}",
            )
        actual_split_fingerprints.add(candidate.split_fingerprint)
        actual_split_arrays.add(candidate.split_arrays_sha256)
        evidence.append(candidate)
        if candidate.decision == "accepted":
            selected_count += 1
        if selected_count == _REQUIRED_SELECTION_COUNT:
            break

    if selected_count < _REQUIRED_SELECTION_COUNT:
        _fail(
            "E_DCL_SELECTION_INSUFFICIENT",
            (
                f"only {selected_count} eligible DCL Vote partitions were found; "
                f"{_REQUIRED_SELECTION_COUNT} are required"
            ),
        )

    evidence_payload = [asdict(item) for item in evidence]
    selected_payload = [item for item in evidence_payload if item["decision"] == "accepted"]
    rejected_payload = [item for item in evidence_payload if item["decision"] != "accepted"]
    target_dataset_fingerprints = {task.expected_dataset_fingerprint for task in target_tasks}
    target_dataset_content_digests = {task.expected_dataset_content_sha256 for task in target_tasks}
    if len(target_dataset_fingerprints) != 1 or len(target_dataset_content_digests) != 1:
        _fail(
            "E_DCL_SELECTION_MISMATCH",
            "DCL Vote candidates do not share one immutable dataset identity",
        )
    payload = {
        "schema_version": 1,
        "kind": "modssc.dcl-vote-conditioned-partition-selection",
        "campaign_id": meta["campaign_id"],
        "protocol_id": selected_protocol,
        "method_id": _METHOD_ID,
        "method_profile": _METHOD_PROFILE,
        "dataset_id": _DATASET_ID,
        "required_selection_count": _REQUIRED_SELECTION_COUNT,
        "candidate_count": len(target_tasks),
        "evaluated_candidate_count": len(evidence),
        "selected_count": len(selected_payload),
        "rejected_count": len(rejected_payload),
        "cutoff_seed": evidence[-1].seed,
        "selection_rule": {
            "order_by": "manifest_seed_ascending",
            "diagnostic_path": _DIAGNOSTIC_PATH,
            "operator": "gt",
            "value": 0,
            "required_converged": True,
            "required_n_iter_lt": _MAX_ITER,
            "unresolved_prefix_policy": "fail_closed",
            "test_information_used": False,
        },
        "source": {
            "manifest_sha256": meta["manifest_sha256"],
            "manifest_meta_sha256": sha256_file(resolved_meta_path),
            "reconcile_sha256": sha256_file(reconcile_path),
            "expected_git_sha": meta["expected_git_sha"],
            "expected_git_diff_sha256": meta.get("expected_git_diff_sha256"),
            "environment_lock_sha256": meta["environment_lock_sha256"],
            "expected_dataset_fingerprint": next(iter(target_dataset_fingerprints)),
            "expected_dataset_content_sha256": next(iter(target_dataset_content_digests)),
        },
        "evaluated_candidates": evidence_payload,
        "selected": selected_payload,
        "rejected": rejected_payload,
    }
    output_digest = _write_immutable_json(output_path, payload)
    return DCLPartitionSelectionResult(
        campaign_id=str(meta["campaign_id"]),
        protocol_id=selected_protocol,
        output_path=str(output_path.resolve()),
        output_sha256=output_digest,
        candidate_count=len(target_tasks),
        evaluated_candidate_count=len(evidence),
        selected_count=len(selected_payload),
        rejected_count=len(rejected_payload),
        cutoff_seed=evidence[-1].seed,
    )
