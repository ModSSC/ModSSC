from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench import main as bench_main
from bench.utils import runtime as runtime_mod


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_git_provenance_fingerprints_tracked_and_untracked_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "tests@example.invalid")
    _run_git(repo, "config", "user.name", "ModSSC tests")
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _run_git(repo, "add", ".gitignore", "tracked.txt")
    _run_git(repo, "commit", "-m", "initial")

    sha, dirty, clean_fingerprint = runtime_mod._git_provenance(repo)
    assert sha is not None
    assert dirty is False
    assert clean_fingerprint is not None and len(clean_fingerprint) == 64

    (repo / "ignored.txt").write_text("ignored secret\n", encoding="utf-8")
    assert runtime_mod._git_provenance(repo)[1:] == (False, clean_fingerprint)

    secret = "untracked-sensitive-value"
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    (repo / "untracked.txt").write_text(secret, encoding="utf-8")
    first = runtime_mod._git_provenance(repo)
    second = runtime_mod._git_provenance(repo)
    assert first == second
    assert first[1] is True
    assert first[2] != clean_fingerprint
    assert secret not in repr(first)

    (repo / "untracked.txt").write_text(f"{secret}-changed", encoding="utf-8")
    assert runtime_mod._git_provenance(repo)[2] != first[2]

    before_staging = runtime_mod._git_provenance(repo)[2]
    _run_git(repo, "add", "tracked.txt")
    assert runtime_mod._git_provenance(repo)[2] != before_staging


def test_collect_runtime_versions_marks_non_git_provenance_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _missing_optional_dependency(_name: str) -> None:
        raise ImportError

    monkeypatch.setattr(runtime_mod, "import_module", _missing_optional_dependency)
    versions = runtime_mod.collect_runtime_versions(repo_root=tmp_path)

    assert versions["git_sha"] is None
    assert versions["git_dirty"] is None
    assert versions["git_diff_sha256"] is None
    assert "distribution_sha256" in versions


def test_distribution_fingerprint_authenticates_installed_scientific_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "bench" / "main.py"
    package.parent.mkdir()
    package.write_text("payload\n", encoding="utf-8")
    fake_distribution = SimpleNamespace(
        files=[Path("bench/main.py"), Path("bench/__pycache__/ignored.pyc")],
        locate_file=lambda entry: tmp_path / entry,
    )
    runtime_mod._distribution_fingerprint.cache_clear()
    monkeypatch.setattr(runtime_mod.metadata, "distribution", lambda _name: fake_distribution)

    first = runtime_mod._distribution_fingerprint("modssc-test")
    runtime_mod._distribution_fingerprint.cache_clear()
    package.write_text("changed\n", encoding="utf-8")
    second = runtime_mod._distribution_fingerprint("modssc-test")

    assert first is not None and len(first) == 64
    assert second is not None and second != first
    runtime_mod._distribution_fingerprint.cache_clear()


def test_bench_collects_provenance_from_its_code_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _capture(
        *,
        repo_root: Path | None = None,
        required_distributions: tuple[str, ...] = (),
        require_complete_manifest: bool = False,
    ) -> dict[str, str]:
        captured["repo_root"] = repo_root
        captured["required_distributions"] = required_distributions
        captured["require_complete_manifest"] = require_complete_manifest
        return {"git_sha": "test-sha"}

    monkeypatch.setattr(bench_main, "collect_runtime_versions", _capture)

    assert bench_main._collect_code_runtime_versions() == {"git_sha": "test-sha"}
    assert captured["repo_root"] == Path(bench_main.__file__).resolve().parent
    assert captured["required_distributions"] == ()
    assert captured["require_complete_manifest"] is False
