from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_RECONCILIATION = REPO_ROOT / "src" / "modssc" / "evaluation" / "reconciliation.py"
NATIVE_PROTOCOL = REPO_ROOT / "src" / "modssc" / "runtime" / "protocol.py"
BENCH_IDENTITY = REPO_ROOT / "bench" / "utils" / "identity.py"
REPORTING = REPO_ROOT / "bench" / "orchestrators" / "reporting.py"
MAIN_RUNNER = REPO_ROOT / "bench" / "main.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_native_seed_reconciliation_has_no_runner_or_operational_dependency() -> None:
    imports = _imports(NATIVE_RECONCILIATION) | _imports(NATIVE_PROTOCOL)

    assert not {
        module
        for module in imports
        if module == "bench"
        or module.startswith("bench.")
        or module == "tools"
        or module.startswith("tools.")
        or module == "provenance"
        or module.startswith("provenance.")
    }


def test_bench_identity_delegates_protocol_semantics_to_native_runtime() -> None:
    tree = _tree(BENCH_IDENTITY)
    imported_from_native_protocol = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "modssc.runtime.protocol"
        for alias in node.names
    }
    local_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    assert {
        "build_resume_identity",
        "effective_config_sha256",
        "protocol_identity_payload",
        "protocol_sha256",
    } <= imported_from_native_protocol
    assert not (
        {"build_resume_identity", "protocol_identity_payload", "protocol_sha256"} & local_functions
    )


def test_seed_reporting_delegates_partition_and_aggregation_to_native_api() -> None:
    tree = _tree(REPORTING)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "write_seed_sweep_summary"
    )
    called_names = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert {"evaluate_acceptance", "reconcile_seed_reports"} <= imported_names
    assert {
        "evaluate_acceptance",
        "reconcile_seed_reports",
        "validate_run_payload",
    } <= called_names
    assert not ({"aggregate_metric_records", "summarize_numeric"} & imported_names)
    assert not ({"aggregate_metric_records", "summarize_numeric"} & called_names)


def test_separate_seed_reconciliation_reconstructs_declared_run_identity() -> None:
    tree = _tree(MAIN_RUNNER)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "reconcile_seed_runs"
    )
    called_names = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert {"_expected_report_hashes", "apply_global_seed", "sweep_run_name"} <= called_names

    identity_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_expected_report_hashes"
    )
    identity_calls = {
        node.func.id
        for node in ast.walk(identity_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"apply_limits", "hash_any", "protocol_sha256"} <= identity_calls
