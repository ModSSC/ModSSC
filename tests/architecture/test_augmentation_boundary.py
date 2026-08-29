from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "bench"


def test_bench_has_no_augmentation_orchestrator() -> None:
    assert not (BENCH_ROOT / "orchestrators" / "augmentation.py").exists()


def test_bench_calls_the_native_augmentation_runtime() -> None:
    main_path = BENCH_ROOT / "main.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"), filename=str(main_path))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "modssc.data_augmentation"
        for alias in node.names
    }
    assert "prepare_unlabeled_augmentation" in imports

    local_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "materialize_views" not in local_functions
    assert "build_online_augmentation" not in local_functions
