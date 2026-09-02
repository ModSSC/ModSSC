"""Assembly, enforcement, and hashing of resolved execution contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from modssc.runtime.contracts import (
    ComponentProvision,
    ContractIssue,
    ExecutionContractError,
    ExecutionContractReport,
    MethodExecutionContract,
    ValueDescriptor,
)


def build_execution_contract_report(
    *,
    method_id: str,
    contract: MethodExecutionContract,
    input_provisions: Iterable[tuple[str, ValueDescriptor]],
    component_provisions: Iterable[ComponentProvision] = (),
    issues: Iterable[ContractIssue] = (),
    unverified: Iterable[ContractIssue] = (),
) -> ExecutionContractReport:
    """Build one deterministic report from independently validated layers."""

    return ExecutionContractReport(
        method_id=method_id,
        issues=tuple(issues),
        unverified=tuple(unverified),
        input_provisions=tuple(input_provisions),
        component_provisions=tuple(component_provisions),
        contract=contract,
    )


def enforce_execution_contract(
    report: ExecutionContractReport,
    *,
    strict: bool,
) -> ExecutionContractReport:
    """Reject incompatibilities and reject unknown proof only in strict mode."""

    if report.issues or (strict and report.unverified):
        raise ExecutionContractError(report)
    return report


def execution_contract_sha256(report: ExecutionContractReport) -> str:
    """Hash the complete metadata-only report using canonical strict JSON."""

    payload = json.dumps(
        report.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "build_execution_contract_report",
    "enforce_execution_contract",
    "execution_contract_sha256",
]
