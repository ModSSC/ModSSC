from __future__ import annotations

import copy
from typing import Any

import pytest

from bench.errors import BenchRuntimeError
from bench.execution_contracts import (
    EXECUTION_CONTRACT_ERROR_CODE,
    execution_contract_payload_sha256,
    persist_execution_contract_from_error,
    persist_execution_contract_from_resolution,
)
from bench.report_schema import validate_run_payload
from modssc.runtime.contracts import (
    ContractIssue,
    ExecutionContractError,
    ExecutionContractReport,
)


def _report_payload(*, status: str = "compatible") -> dict[str, Any]:
    issues = (
        [] if status != "incompatible" else [{"code": "E_INPUT_RANK", "message": "rank mismatch"}]
    )
    return {
        "method_id": "dummy",
        "status": status,
        "issues": issues,
        "unverified": [],
        "contract": None,
        "inputs": {},
        "components": [],
    }


def _run_payload() -> dict[str, Any]:
    return {
        "run": {
            "name": "contract",
            "seed": 1,
            "run_id": "run",
            "started_at": "start",
            "finished_at": "finish",
            "status": "success",
            "benchmark_mode": False,
            "config_path": None,
            "error_code": None,
        },
        "hashes": {
            "config_hash": "a" * 64,
            "effective_config_hash": "b" * 64,
            "protocol_sha256": "c" * 64,
            "software_sha256": "d" * 64,
        },
        "resolution": {
            "device": {"requested": "cpu", "resolved": "cpu"},
            "backend": {"requested": {}, "resolved": {}},
            "dtype": {"requested": {}, "resolved": {}},
            "normalization": {"requested": {}, "resolved": {}},
            "splits": {"requested": [], "resolved": {}},
            "limits": {"requested": None, "resolved": None, "changes": []},
        },
        "protocol": {
            "kind": "inductive",
            "use_test_split": True,
            "report_splits": ["test"],
            "split_for_model_selection": "val",
        },
        "versions": {"python": "x", "modssc": "x", "numpy": "x", "git_sha": "x"},
        "config": {},
        "artifacts": {"method": {"id": "dummy"}},
        "fallback_events": [],
        "run_info": None,
        "task_info": None,
        "graph_info": None,
        "metrics": None,
        "hpo": None,
        "error": None,
    }


def test_native_resolution_is_persisted_once_with_compact_summary() -> None:
    report = _report_payload()
    digest = execution_contract_payload_sha256(report)
    artifacts: dict[str, Any] = {"method": {"id": "dummy"}}
    resolution: dict[str, Any] = {}

    assert persist_execution_contract_from_resolution(
        {"execution_contract": report, "execution_contract_sha256": digest},
        artifacts=artifacts,
        resolution=resolution,
    )

    assert artifacts["method"]["execution_contract"] == report
    assert artifacts["method"]["execution_contract_sha256"] == digest
    assert resolution["execution_contract"] == {
        "status": "compatible",
        "sha256": digest,
    }


def test_execution_contract_error_is_found_through_cause_chain() -> None:
    report = ExecutionContractReport(
        method_id="dummy",
        issues=(ContractIssue(code="E_INPUT_RANK", message="rank mismatch"),),
    )
    try:
        try:
            raise ExecutionContractError(report)
        except ExecutionContractError as exc:
            raise BenchRuntimeError(EXECUTION_CONTRACT_ERROR_CODE, "rejected") from exc
    except BenchRuntimeError as exc:
        wrapped = exc

    artifacts: dict[str, Any] = {}
    resolution: dict[str, Any] = {}
    assert persist_execution_contract_from_error(
        wrapped,
        artifacts=artifacts,
        resolution=resolution,
    )
    persisted = artifacts["method"]["execution_contract"]
    assert persisted["status"] == "incompatible"
    assert resolution["execution_contract"]["status"] == "incompatible"


def test_run_schema_accepts_absence_and_validates_optional_contract_digest() -> None:
    legacy = _run_payload()
    validate_run_payload(legacy, require_execution_identity=False)

    payload = copy.deepcopy(legacy)
    report = _report_payload()
    persist_execution_contract_from_resolution(
        {
            "execution_contract": report,
            "execution_contract_sha256": execution_contract_payload_sha256(report),
        },
        artifacts=payload["artifacts"],
        resolution=payload["resolution"],
    )
    validate_run_payload(payload, require_execution_identity=False)

    payload["artifacts"]["method"]["execution_contract"]["method_id"] = "tampered"
    with pytest.raises(BenchRuntimeError, match="does not match its SHA-256"):
        validate_run_payload(payload, require_execution_identity=False)
