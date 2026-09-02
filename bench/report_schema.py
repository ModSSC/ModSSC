from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from modssc.runtime.execution import RunIdentity
from modssc.runtime.software import SoftwareManifest, SoftwareProvenanceError

from .errors import BenchRuntimeError
from .execution_contracts import (
    EXECUTION_CONTRACT_KEY,
    EXECUTION_CONTRACT_SHA256_KEY,
    execution_contract_payload_sha256,
)


def _require_mapping(obj: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise BenchRuntimeError("E_BENCH_RUN_JSON_SCHEMA", f"{path} must be a mapping")
    return obj


def _require_keys(obj: Mapping[str, Any], *, path: str, keys: list[str]) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            f"{path} missing keys: {sorted(missing)}",
        )


def _require_sha256(obj: Mapping[str, Any], *, path: str, key: str) -> None:
    value = obj.get(key)
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            f"{path}.{key} must be a lowercase SHA-256 hex digest",
        )


def _validate_git_provenance(versions: Mapping[str, Any]) -> None:
    distribution_sha256 = versions.get("distribution_sha256")
    if distribution_sha256 is not None and not (
        isinstance(distribution_sha256, str)
        and len(distribution_sha256) == 64
        and all(character in "0123456789abcdef" for character in distribution_sha256)
    ):
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "versions.distribution_sha256 must be null or a lowercase SHA-256 hex digest",
        )
    provenance_keys = {"git_dirty", "git_diff_sha256"}
    present = provenance_keys.intersection(versions)
    if not present:
        # Reports created before worktree provenance was introduced remain readable.
        return
    if present != provenance_keys:
        missing = sorted(provenance_keys - present)
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            f"versions missing Git provenance keys: {missing}",
        )

    dirty = versions["git_dirty"]
    fingerprint = versions["git_diff_sha256"]
    if dirty is None:
        if fingerprint is not None:
            raise BenchRuntimeError(
                "E_BENCH_RUN_JSON_SCHEMA",
                "versions.git_diff_sha256 must be null when versions.git_dirty is null",
            )
        return
    if type(dirty) is not bool:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "versions.git_dirty must be a boolean or null",
        )
    if not (
        isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and all(character in "0123456789abcdef" for character in fingerprint)
    ):
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "versions.git_diff_sha256 must be a lowercase SHA-256 hex digest",
        )


def _validate_software_manifest(versions: Mapping[str, Any]) -> None:
    manifest = versions.get("software_manifest")
    if manifest is None:
        # Reports created before selective manifests were introduced stay readable.
        return
    try:
        SoftwareManifest.from_dict(manifest)
    except (SoftwareProvenanceError, TypeError) as exc:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            f"versions.software_manifest is invalid: {exc}",
        ) from exc


def _validate_execution_identity(
    root: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    hashes: Mapping[str, Any],
    required: bool,
) -> None:
    payload = root.get("execution_identity")
    digest = hashes.get("execution_identity_sha256")
    if payload is None and digest is None:
        if required:
            raise BenchRuntimeError(
                "E_BENCH_RUN_JSON_SCHEMA",
                "execution_identity is required for a modern run report",
            )
        # Historical reports remain readable only through an explicit opt-in.
        return
    if payload is None or digest is None:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "execution_identity and hashes.execution_identity_sha256 must be present together",
        )
    _require_sha256(hashes, path="hashes", key="execution_identity_sha256")
    try:
        identity = RunIdentity.from_dict(_require_mapping(payload, path="execution_identity"))
    except (TypeError, ValueError) as exc:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            f"execution_identity is invalid: {exc}",
        ) from exc
    if identity.sha256 != digest:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "execution_identity does not match hashes.execution_identity_sha256",
        )
    if identity.seed != run.get("seed"):
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "execution_identity seed differs from run.seed",
        )
    if identity.config_sha256 != hashes.get("protocol_sha256"):
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "execution_identity config digest differs from hashes.protocol_sha256",
        )
    if identity.code_sha256 != hashes.get("software_sha256"):
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "execution_identity code digest differs from hashes.software_sha256",
        )
    if run.get("run_id") != identity.short_id:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "run.run_id differs from the portable execution identity",
        )


def _validate_execution_contract(
    artifacts: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> None:
    method = artifacts.get("method")
    method_mapping = method if isinstance(method, Mapping) else None
    report_present = method_mapping is not None and EXECUTION_CONTRACT_KEY in method_mapping
    digest_present = method_mapping is not None and EXECUTION_CONTRACT_SHA256_KEY in method_mapping
    summary_present = EXECUTION_CONTRACT_KEY in resolution
    if not report_present and not digest_present and not summary_present:
        # Reports created before execution-contract composition remain readable.
        return
    if method_mapping is None or not (report_present and digest_present and summary_present):
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "execution contract report, artifact SHA-256, and resolution summary "
            "must be present together",
        )

    report = _require_mapping(
        method_mapping[EXECUTION_CONTRACT_KEY],
        path="artifacts.method.execution_contract",
    )
    _require_sha256(
        method_mapping,
        path="artifacts.method",
        key=EXECUTION_CONTRACT_SHA256_KEY,
    )
    summary = _require_mapping(
        resolution[EXECUTION_CONTRACT_KEY],
        path="resolution.execution_contract",
    )
    _require_keys(
        summary,
        path="resolution.execution_contract",
        keys=["status", "sha256"],
    )
    _require_sha256(summary, path="resolution.execution_contract", key="sha256")

    report_status = report.get("status")
    if report_status not in {"compatible", "incompatible", "unverified"}:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "artifacts.method.execution_contract.status is invalid",
        )
    if summary["status"] != report_status:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "resolution execution contract status differs from the artifact report",
        )
    artifact_digest = method_mapping[EXECUTION_CONTRACT_SHA256_KEY]
    if summary["sha256"] != artifact_digest:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "resolution execution contract SHA-256 differs from the artifact digest",
        )
    try:
        computed_digest = execution_contract_payload_sha256(report)
    except BenchRuntimeError as exc:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            f"artifacts.method.execution_contract is invalid: {exc.message}",
        ) from exc
    if computed_digest != artifact_digest:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "execution contract artifact does not match its SHA-256 digest",
        )


def validate_run_payload(
    payload: Any,
    *,
    require_execution_identity: bool = True,
) -> None:
    root = _require_mapping(payload, path="run.json")
    _require_keys(
        root,
        path="run.json",
        keys=[
            "run",
            "hashes",
            "resolution",
            "protocol",
            "versions",
            "config",
            "artifacts",
            "fallback_events",
            "run_info",
            "task_info",
            "graph_info",
            "metrics",
            "hpo",
            "error",
        ],
    )

    run = _require_mapping(root["run"], path="run")
    _require_keys(
        run,
        path="run",
        keys=[
            "name",
            "seed",
            "run_id",
            "started_at",
            "finished_at",
            "status",
            "benchmark_mode",
            "config_path",
            "error_code",
        ],
    )

    hashes = _require_mapping(root["hashes"], path="hashes")
    _require_keys(
        hashes,
        path="hashes",
        keys=[
            "config_hash",
            "effective_config_hash",
            "protocol_sha256",
            "software_sha256",
        ],
    )
    for hash_key in (
        "config_hash",
        "effective_config_hash",
        "protocol_sha256",
        "software_sha256",
    ):
        _require_sha256(hashes, path="hashes", key=hash_key)
    _validate_execution_identity(
        root,
        run=run,
        hashes=hashes,
        required=require_execution_identity,
    )

    resolution = _require_mapping(root["resolution"], path="resolution")
    _require_keys(
        resolution,
        path="resolution",
        keys=["device", "backend", "dtype", "normalization", "splits", "limits"],
    )
    _require_keys(
        _require_mapping(resolution["device"], path="resolution.device"),
        path="resolution.device",
        keys=["requested", "resolved"],
    )
    _require_keys(
        _require_mapping(resolution["backend"], path="resolution.backend"),
        path="resolution.backend",
        keys=["requested", "resolved"],
    )
    _require_keys(
        _require_mapping(resolution["dtype"], path="resolution.dtype"),
        path="resolution.dtype",
        keys=["requested", "resolved"],
    )
    _require_keys(
        _require_mapping(resolution["normalization"], path="resolution.normalization"),
        path="resolution.normalization",
        keys=["requested", "resolved"],
    )
    _require_keys(
        _require_mapping(resolution["splits"], path="resolution.splits"),
        path="resolution.splits",
        keys=["requested", "resolved"],
    )
    _require_keys(
        _require_mapping(resolution["limits"], path="resolution.limits"),
        path="resolution.limits",
        keys=["requested", "resolved", "changes"],
    )

    protocol = _require_mapping(root["protocol"], path="protocol")
    _require_keys(
        protocol,
        path="protocol",
        keys=["kind", "use_test_split", "report_splits", "split_for_model_selection"],
    )
    test_selection_policy = protocol.get("test_selection_policy")
    if test_selection_policy is not None and test_selection_policy not in {
        "forbid",
        "paper_protocol",
    }:
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "protocol.test_selection_policy must be 'forbid' or 'paper_protocol' when present",
        )

    versions = _require_mapping(root["versions"], path="versions")
    _require_keys(
        versions,
        path="versions",
        keys=["python", "modssc", "numpy", "git_sha"],
    )
    _validate_git_provenance(versions)
    _validate_software_manifest(versions)

    _require_mapping(root["config"], path="config")
    artifacts = _require_mapping(root["artifacts"], path="artifacts")
    _validate_execution_contract(artifacts, resolution)
    for optional_mapping in ("run_info", "task_info", "graph_info", "metrics", "hpo"):
        value = root[optional_mapping]
        if value is not None:
            _require_mapping(value, path=optional_mapping)
    error = root["error"]
    if error is not None and not isinstance(error, str):
        raise BenchRuntimeError(
            "E_BENCH_RUN_JSON_SCHEMA",
            "error must be a string or null",
        )

    fallback_events = root["fallback_events"]
    if not isinstance(fallback_events, list):
        raise BenchRuntimeError("E_BENCH_RUN_JSON_SCHEMA", "fallback_events must be a list")
