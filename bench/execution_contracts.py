"""Generic benchmark persistence for native execution-contract reports."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from modssc.runtime.contracts import ExecutionContractError

from .errors import BenchRuntimeError

EXECUTION_CONTRACT_ERROR_CODE = "E_BENCH_EXECUTION_CONTRACT"
EXECUTION_CONTRACT_KEY = "execution_contract"
EXECUTION_CONTRACT_SHA256_KEY = "execution_contract_sha256"
_CONTRACT_STATUSES = frozenset({"compatible", "incompatible", "unverified"})
_MISSING = object()


def execution_contract_payload_sha256(report: Mapping[str, Any]) -> str:
    """Hash one serialized report exactly like the native contract runtime."""

    try:
        payload = json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BenchRuntimeError(
            EXECUTION_CONTRACT_ERROR_CODE,
            f"execution contract report is not strict-JSON serializable: {exc}",
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _validated_payload(report: Any, digest: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(report, Mapping):
        raise BenchRuntimeError(
            EXECUTION_CONTRACT_ERROR_CODE,
            "native execution_contract must be a mapping",
        )
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    ):
        raise BenchRuntimeError(
            EXECUTION_CONTRACT_ERROR_CODE,
            "native execution_contract_sha256 must be a lowercase SHA-256 digest",
        )

    payload = copy.deepcopy(dict(report))
    status = payload.get("status")
    if status not in _CONTRACT_STATUSES:
        raise BenchRuntimeError(
            EXECUTION_CONTRACT_ERROR_CODE,
            "native execution contract report has an invalid status",
        )
    computed = execution_contract_payload_sha256(payload)
    if digest != computed:
        raise BenchRuntimeError(
            EXECUTION_CONTRACT_ERROR_CODE,
            "native execution contract report does not match its SHA-256 digest",
        )
    return payload, digest


def _persist_payload(
    report: Any,
    digest: Any,
    *,
    artifacts: dict[str, Any],
    resolution: dict[str, Any],
) -> None:
    payload, normalized_digest = _validated_payload(report, digest)
    method_artifacts = artifacts.get("method")
    if not isinstance(method_artifacts, dict):
        method_artifacts = {}
        artifacts["method"] = method_artifacts
    method_artifacts[EXECUTION_CONTRACT_KEY] = payload
    method_artifacts[EXECUTION_CONTRACT_SHA256_KEY] = normalized_digest
    resolution[EXECUTION_CONTRACT_KEY] = {
        "status": payload["status"],
        "sha256": normalized_digest,
    }


def persist_execution_contract_from_resolution(
    native_resolution: Mapping[str, Any],
    *,
    artifacts: dict[str, Any],
    resolution: dict[str, Any],
) -> bool:
    """Copy an optional native report without knowing the concrete method."""

    report = native_resolution.get(EXECUTION_CONTRACT_KEY, _MISSING)
    digest = native_resolution.get(EXECUTION_CONTRACT_SHA256_KEY, _MISSING)
    if report is _MISSING and digest is _MISSING:
        return False
    if report is _MISSING or digest is _MISSING:
        raise BenchRuntimeError(
            EXECUTION_CONTRACT_ERROR_CODE,
            "native resolution must provide execution contract report and SHA-256 together",
        )
    _persist_payload(report, digest, artifacts=artifacts, resolution=resolution)
    return True


def find_execution_contract_error(error: BaseException) -> ExecutionContractError | None:
    """Find a native contract error through explicit causes or implicit contexts."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ExecutionContractError):
            return current
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return None


def persist_execution_contract_from_error(
    error: BaseException,
    *,
    artifacts: dict[str, Any],
    resolution: dict[str, Any],
) -> bool:
    """Persist a rejected native report retained in an exception chain."""

    contract_error = find_execution_contract_error(error)
    if contract_error is None:
        return False
    payload = contract_error.report.to_dict()
    digest = execution_contract_payload_sha256(payload)
    _persist_payload(payload, digest, artifacts=artifacts, resolution=resolution)
    return True


__all__ = [
    "EXECUTION_CONTRACT_ERROR_CODE",
    "EXECUTION_CONTRACT_KEY",
    "EXECUTION_CONTRACT_SHA256_KEY",
    "execution_contract_payload_sha256",
    "find_execution_contract_error",
    "persist_execution_contract_from_error",
    "persist_execution_contract_from_resolution",
]
