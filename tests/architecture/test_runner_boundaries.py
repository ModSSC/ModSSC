from __future__ import annotations

import ast
from pathlib import Path

from modssc.data_loader import available_datasets
from modssc.inductive.registry import available_methods as available_inductive_methods
from modssc.transductive.registry import available_methods as available_transductive_methods

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "bench"
SRC_ROOT = REPO_ROOT / "src" / "modssc"
SAMPLING_ORCHESTRATOR = BENCH_ROOT / "orchestrators" / "sampling.py"
TRANSDUCTIVE_ORCHESTRATOR = BENCH_ROOT / "orchestrators" / "method_transductive.py"
HPO_ORCHESTRATOR = BENCH_ROOT / "orchestrators" / "hpo.py"
MAIN_RUNNER = BENCH_ROOT / "main.py"
DISTRIBUTION_AUDIT = REPO_ROOT / ".github" / "scripts" / "audit_distribution.py"
ACTIVE_TEXT_ROOTS = (SRC_ROOT, BENCH_ROOT, REPO_ROOT / "docs")
ACTIVE_TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml"}
REMOVED_RUNTIME_REFERENCES = {
    "bench/assets",
    "bench/campaign",
    "bench/campaigns",
    "provenance/article10",
    "tools.campaign",
    "tools/campaign",
    "tools.replication_audit",
}

REGISTERED_METHOD_IDS = set(available_inductive_methods()) | set(available_transductive_methods())
REGISTERED_DATASET_IDS = set(available_datasets())
MODEL_BINDING_FIELDS = {
    "model_bundle",
    "teacher_bundle",
    "student_bundle",
    "model_bundle_1",
    "model_bundle_2",
    "pretrain_bundle",
    "finetune_bundle",
    "shared_bundle",
    "head_bundles",
}


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _compared_string_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for operand in (node.left, *node.comparators):
            if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                values.add(operand.value)
            elif isinstance(operand, (ast.Set, ast.Tuple, ast.List)):
                values.update(
                    item.value
                    for item in operand.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )
    return values


def test_bench_contains_only_runner_owned_locations() -> None:
    forbidden = [
        BENCH_ROOT / "assets",
        BENCH_ROOT / "campaign",
        BENCH_ROOT / "campaigns",
        BENCH_ROOT / "slurm",
        BENCH_ROOT / "reproduce.py",
        BENCH_ROOT / "partition_selection_schema.py",
    ]
    assert not [path for path in forbidden if path.exists()]


def test_removed_root_frameworks_and_legacy_tests_stay_absent() -> None:
    forbidden = [
        REPO_ROOT / "tools",
        REPO_ROOT / "provenance",
        REPO_ROOT / "tests" / "tools",
        REPO_ROOT / "tests" / "bench" / "test_hpc_build_manifest.py",
        REPO_ROOT / "tests" / "bench" / "test_hpc_job_env.py",
        REPO_ROOT / "tests" / "bench" / "test_hpc_model_artifacts.py",
        REPO_ROOT / "tests" / "bench" / "test_hpc_submit_chained_arrays.py",
        REPO_ROOT / "tests" / "bench" / "test_match_continuation_controller.py",
        REPO_ROOT / "tests" / "bench" / "test_public_hpc_portability.py",
    ]
    assert not [path for path in forbidden if path.exists()]


def test_distribution_audit_rejects_removed_legacy_tests() -> None:
    content = DISTRIBUTION_AUDIT.read_text(encoding="utf-8")
    forbidden_fragments = {
        "tests/bench/test_hpc_",
        "tests/bench/test_match_continuation_controller.py",
        "tests/bench/test_public_hpc_portability.py",
    }
    assert not [fragment for fragment in forbidden_fragments if fragment not in content]


def test_active_runtime_and_docs_do_not_reference_removed_frameworks() -> None:
    violations: list[str] = []
    for root in ACTIVE_TEXT_ROOTS:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in ACTIVE_TEXT_SUFFIXES:
                continue
            content = path.read_text(encoding="utf-8")
            found = sorted(token for token in REMOVED_RUNTIME_REFERENCES if token in content)
            if found:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {found}")
    assert not violations, "active files reference removed frameworks:\n" + "\n".join(violations)


def test_bench_does_not_depend_on_root_operational_code() -> None:
    violations: list[str] = []
    for path in _python_files(BENCH_ROOT):
        forbidden = sorted(
            module
            for module in _imports(path)
            if module == "tools" or module.startswith(("tools.", "provenance."))
        )
        if forbidden:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {forbidden}")
    assert not violations, "bench has reverse dependencies:\n" + "\n".join(violations)


def test_bench_python_is_scheduler_agnostic() -> None:
    forbidden_tokens = ("SLURM_", "sbatch", "srun", "PBS_", "LSB_JOB")
    violations: list[str] = []
    for path in _python_files(BENCH_ROOT):
        content = path.read_text(encoding="utf-8")
        found = sorted(token for token in forbidden_tokens if token in content)
        if found:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {found}")
    assert not violations, "bench embeds scheduler behavior:\n" + "\n".join(violations)


def test_bench_never_dispatches_on_article_method_identity() -> None:
    violations: list[str] = []
    for path in _python_files(BENCH_ROOT):
        identities = sorted(_compared_string_literals(path) & REGISTERED_METHOD_IDS)
        if identities:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {identities}")
    assert not violations, "bench dispatches on method identity:\n" + "\n".join(violations)


def test_bench_never_dispatches_on_dataset_identity() -> None:
    violations: list[str] = []
    for path in _python_files(BENCH_ROOT):
        identities = sorted(_compared_string_literals(path) & REGISTERED_DATASET_IDS)
        if identities:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {identities}")
    assert not violations, "bench dispatches on dataset identity:\n" + "\n".join(violations)


def test_bench_does_not_own_inductive_model_binding() -> None:
    violations: list[str] = []
    for path in _python_files(BENCH_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        referenced = {
            value
            for node in ast.walk(tree)
            for value in (
                node.attr if isinstance(node, ast.Attribute) else None,
                node.id if isinstance(node, ast.Name) else None,
                node.value
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                else None,
            )
            if isinstance(value, str)
        }
        fields = sorted(referenced & MODEL_BINDING_FIELDS)
        imports_deep = any(
            module == "modssc.inductive.deep" or module.startswith("modssc.inductive.deep.")
            for module in _imports(path)
        )
        if fields or imports_deep:
            violations.append(
                f"{path.relative_to(REPO_ROOT)}: fields={fields} imports_deep={imports_deep}"
            )
    assert not violations, "bench owns inductive model binding:\n" + "\n".join(violations)


def test_bench_does_not_name_preprocessing_implementations() -> None:
    forbidden = {"features.vae", "features.aet", "poisson_mnist"}
    violations: list[str] = []
    for path in _python_files(BENCH_ROOT):
        content = path.read_text(encoding="utf-8")
        found = sorted(token for token in forbidden if token in content)
        if found:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {found}")

    assert not violations, "bench names preprocessing implementations:\n" + "\n".join(violations)


def test_bench_does_not_implement_pipeline_capability_semantics() -> None:
    forbidden = {
        "_build_pipeline_capabilities",
        "_dataset_has_graph",
        "_expected_labeled_count",
        "_graph_sampling_to_inductive",
        "_pipeline_backend",
        "_pipeline_representation",
        "_plan_from_dict",
        "_resolved_pipeline_backend",
        "_sampling_has_unlabeled",
        "_preprocess_step_ids",
        "_requires_fit_indices",
        "_use_test_split",
        "_validate_materialized_pipeline",
        "_views_preprocess_step_ids",
    }
    violations: list[str] = []
    for path in _python_files(BENCH_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        local_functions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        found = sorted(local_functions & forbidden)
        if found:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {found}")

    assert not violations, "bench owns pipeline semantics:\n" + "\n".join(violations)


def test_main_runner_uses_native_resolution_and_input_routing_boundaries() -> None:
    imports = _imports(MAIN_RUNNER)
    assert {
        "modssc.runtime.dependencies",
        "modssc.runtime.input_routing",
        "modssc.runtime.pipeline",
    } <= imports
    assert not (
        imports
        & {
            "modssc.capabilities",
            "modssc.data_loader",
            "modssc.graph.construction.builder",
            "modssc.graph.specs",
            "modssc.inductive.registry",
            "modssc.preprocess.registry",
            "modssc.runtime.device",
            "modssc.runtime.method_spec",
            "modssc.supervised.registry",
            "modssc.transductive.registry",
        }
    )

    tree = ast.parse(MAIN_RUNNER.read_text(encoding="utf-8"), filename=str(MAIN_RUNNER))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    direct_sampling_index_accesses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "indices"
    ]
    assert "as_inductive_indices" not in called_attributes
    assert "masks_from_sampling" not in called_attributes
    assert not direct_sampling_index_accesses


def test_bench_does_not_implement_graph_science() -> None:
    graph_orchestrator = BENCH_ROOT / "orchestrators" / "graph.py"
    tree = ast.parse(
        graph_orchestrator.read_text(encoding="utf-8"), filename=str(graph_orchestrator)
    )
    local_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert local_functions == {"build"}
    assert "modssc.graph" in _imports(graph_orchestrator)


def test_bench_does_not_implement_augmentation_materialization() -> None:
    assert not (BENCH_ROOT / "orchestrators" / "augmentation.py").exists()
    content = (BENCH_ROOT / "main.py").read_text(encoding="utf-8")
    assert "select_rows" not in content
    assert "_wrapg" not in content


def test_bench_delegates_sampling_dataset_transforms_to_src() -> None:
    tree = ast.parse(SAMPLING_ORCHESTRATOR.read_text(encoding="utf-8"))
    local_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    imported_modules = _imports(SAMPLING_ORCHESTRATOR)

    assert "_concat_rows" not in local_functions
    assert "modssc.sampling.dataset" in imported_modules


def test_bench_hpo_delegates_search_algorithm_to_src() -> None:
    tree = ast.parse(HPO_ORCHESTRATOR.read_text(encoding="utf-8"))
    local_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    imported_modules = _imports(HPO_ORCHESTRATOR)

    assert "_aggregate" not in local_functions
    assert "_is_better" not in local_functions
    assert "modssc.hpo" in imported_modules


def test_bench_does_not_implement_data_selection_or_transductive_materialization() -> None:
    assert not (BENCH_ROOT / "orchestrators" / "slicing.py").exists()

    tree = ast.parse(TRANSDUCTIVE_ORCHESTRATOR.read_text(encoding="utf-8"))
    local_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden = {
        "_build_masks_from_indices",
        "_combine_splits",
        "_mask_from_indices",
        "_to_numpy",
        "graph_from_dataset",
    }
    constructor_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert not (local_functions & forbidden)
    assert "NodeDataset" not in constructor_calls


def test_transductive_runner_keeps_evaluation_truth_outside_fit_data() -> None:
    tree = ast.parse(TRANSDUCTIVE_ORCHESTRATOR.read_text(encoding="utf-8"))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    meta_truth_accesses = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "meta"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }

    assert "execute_transductive_method" in called_names
    assert "prepare_node_data" not in called_names
    assert "build_node_dataset" not in called_names
    assert "y_true" not in meta_truth_accesses


def test_src_does_not_import_bench_or_tools() -> None:
    violations: list[str] = []
    for path in _python_files(SRC_ROOT):
        forbidden = sorted(
            module
            for module in _imports(path)
            if module == "bench"
            or module.startswith("bench.")
            or module == "tools"
            or module.startswith("tools.")
        )
        if forbidden:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {forbidden}")
    assert not violations, "src has orchestration dependencies:\n" + "\n".join(violations)
