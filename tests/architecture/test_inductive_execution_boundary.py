from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = REPO_ROOT / "bench" / "orchestrators" / "method_inductive.py"


def _tree() -> ast.Module:
    return ast.parse(ORCHESTRATOR.read_text(encoding="utf-8"), filename=str(ORCHESTRATOR))


def test_bench_inductive_orchestrator_depends_only_on_the_public_native_api() -> None:
    tree = _tree()
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "modssc.inductive" in imported_modules
    assert not {
        module
        for module in imported_modules
        if module.startswith(
            (
                "modssc.data_augmentation",
                "modssc.data_loader",
                "modssc.inductive.deep",
                "modssc.inductive.model_binding",
                "modssc.inductive.registry",
                "modssc.runtime.device",
                "modssc.runtime.method_spec",
            )
        )
    }
    direct_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not (direct_imports & {"importlib", "numpy", "torch"})


def test_bench_inductive_orchestrator_contains_no_scientific_execution_logic() -> None:
    tree = _tree()
    local_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert local_functions == {
        "_bench_execution_error",
        "_native_execution_config",
        "_native_model_config",
        "run",
    }
    assert "execute_inductive_method" in called_names
    assert not (
        called_names
        & {
            "InductiveDataset",
            "InductiveExecutionInput",
            "bind_model_to_spec",
            "build_method_spec",
            "get_method_class",
            "get_method_info",
            "select_rows",
        }
    )
    assert "fit" not in called_attributes
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            (
                ast.For,
                ast.While,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        )
    ]
