from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CALDER_ROOT = REPO_ROOT / "bench/assets/calder2020/protocol_inputs"
MATCH_ROOT = REPO_ROOT / "provenance/article10/match_audit"


def test_precommit_excludes_only_authenticated_non_source_assets() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = {hook["id"]: hook for repository in config["repos"] for hook in repository["hooks"]}

    assert hooks["end-of-file-fixer"]["exclude"] == (
        r"^provenance/article10/match_audit/LICENSE\.usb\.mit\.txt$"
    )
    assert "exclude" not in hooks["trailing-whitespace"]
    assert hooks["check-added-large-files"]["exclude"] == (
        r"^bench/assets/calder2020/protocol_inputs/graph/mnist-vae-knn30\.npz$"
    )


def test_precommit_forwards_parallel_pytest_arguments() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = {hook["id"]: hook for repository in config["repos"] for hook in repository["hooks"]}

    pytest_hook = hooks["pytest"]
    assert 'pytest "$@"' in pytest_hook["entry"]
    assert pytest_hook["args"] == [
        "-n",
        "4",
        "--dist",
        "loadgroup",
        "--cov-report=xml:coverage.xml",
    ]


def test_large_file_exemption_is_independently_authenticated() -> None:
    manifest = json.loads((CALDER_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    relative_path = "graph/mnist-vae-knn30.npz"
    record = manifest["files"][relative_path]
    artifact = CALDER_ROOT / relative_path

    assert artifact.stat().st_size == record["size_bytes"]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == record["sha256"]


def test_notices_distinguish_source_dataset_and_derived_artifact_licenses() -> None:
    calder_notice = (CALDER_ROOT.parent / "NOTICE.md").read_text(encoding="utf-8")
    match_notice = (REPO_ROOT / "bench/assets/cifar10_paper_splits/LICENSES.md").read_text(
        encoding="utf-8"
    )

    assert "source code" in calder_notice
    assert "MNIST" in calder_notice
    assert "derived VAE" in calder_notice
    assert "TorchSSL" in match_notice
    assert "Semi-supervised-learning (USB)" in match_notice
    assert (MATCH_ROOT / "LICENSE.torchssl.mit.txt").is_file()
    assert (MATCH_ROOT / "LICENSE.usb.mit.txt").is_file()
    assert not (CALDER_ROOT.parent / "GraphLearningOld-04bece45").exists()
    assert not (MATCH_ROOT / "sources").exists()
    assert not (REPO_ROOT / "bench/assets/match_reference").exists()


def test_library_has_no_article_profile_identifiers() -> None:
    marker = "paper" + ":"
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "src/modssc").rglob("*.py")
        if marker in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_scientific_assets_contain_no_executable_source_or_jvm_runtime() -> None:
    forbidden_suffixes = {".class", ".dll", ".dylib", ".exe", ".jar", ".java", ".py", ".sh", ".so"}
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "bench/assets").rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]

    assert offenders == []
    assert not (REPO_ROOT / "src/modssc/supervised/backends/weka").exists()
    assert not any("GraphLearningOld" in path.parts for path in CALDER_ROOT.parent.rglob("*"))


def test_dependency_direction_keeps_library_and_bench_independent_of_operational_code() -> None:
    forbidden_roots = {
        REPO_ROOT / "src/modssc": {"bench", "tools", "provenance"},
        REPO_ROOT / "bench": {"tools", "provenance"},
    }
    offenders: list[str] = []
    for root, forbidden in forbidden_roots.items():
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules: tuple[str, ...]
                if isinstance(node, ast.Import):
                    modules = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = (node.module,)
                else:
                    continue
                if any(module.split(".", 1)[0] in forbidden for module in modules):
                    offenders.append(path.relative_to(REPO_ROOT).as_posix())
                    break

    assert offenders == []
