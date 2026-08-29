from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "bench"
EVALUATION_ADAPTER = BENCH_ROOT / "orchestrators" / "evaluation.py"


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def test_bench_never_reads_private_fitted_method_state() -> None:
    violations: list[str] = []
    for path in _python_files(BENCH_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        private_attributes = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and not node.attr.startswith("__")
        }
        private_getattrs = {
            node.args[1].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and node.args[1].value.startswith("_")
            and not node.args[1].value.startswith("__")
        }
        found = sorted(private_attributes | private_getattrs)
        if found:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {found}")

    assert not violations, "bench reads private fitted state:\n" + "\n".join(violations)


def test_evaluation_adapter_only_adapts_and_calls_native_runtime() -> None:
    tree = ast.parse(
        EVALUATION_ADAPTER.read_text(encoding="utf-8"),
        filename=str(EVALUATION_ADAPTER),
    )
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    forbidden_scientific_primitives = {
        "compute_metrics",
        "labels_1d",
        "predict_labels",
        "select_rows",
        "to_numpy",
    }
    local_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert {
        "evaluate_inductive_method",
        "evaluate_transductive_method",
        "make_inductive_split_provider",
    } <= imported_names
    assert {
        "evaluate_inductive_method",
        "evaluate_transductive_method",
        "make_inductive_split_provider",
    } <= called_names
    assert local_functions == {
        "_bench_evaluation_error",
        "evaluate_inductive",
        "evaluate_transductive",
    }
    assert not (forbidden_scientific_primitives & imported_names)
    assert not (forbidden_scientific_primitives & called_names)
    assert "InductiveDataset" not in imported_names
    assert "is_torch_tensor" not in imported_names
    assert "importlib" not in {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
