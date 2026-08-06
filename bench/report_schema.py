from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import BenchRuntimeError


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


def validate_run_payload(payload: Mapping[str, Any]) -> None:
    _require_keys(
        payload,
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
            "metrics",
            "hpo",
            "error",
        ],
    )

    run = _require_mapping(payload["run"], path="run")
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

    hashes = _require_mapping(payload["hashes"], path="hashes")
    _require_keys(
        hashes,
        path="hashes",
        keys=["config_hash", "effective_config_hash"],
    )

    resolution = _require_mapping(payload["resolution"], path="resolution")
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

    protocol = _require_mapping(payload["protocol"], path="protocol")
    _require_keys(
        protocol,
        path="protocol",
        keys=["kind", "use_test_split", "report_splits", "split_for_model_selection"],
    )

    versions = _require_mapping(payload["versions"], path="versions")
    _require_keys(
        versions,
        path="versions",
        keys=["python", "modssc", "numpy", "git_sha"],
    )
    _validate_git_provenance(versions)

    fallback_events = payload["fallback_events"]
    if not isinstance(fallback_events, list):
        raise BenchRuntimeError("E_BENCH_RUN_JSON_SCHEMA", "fallback_events must be a list")
