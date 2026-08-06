from __future__ import annotations

import copy
import subprocess

import pytest

from bench.campaign.build_manifest import (
    _installed_distributions,
    build_manifest,
    environment_identity_sha256,
    validate_build_manifest,
)


def _repo_with_tracked_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("content\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "test"], cwd=repo, check=True)
    return repo


def test_build_manifest_hashes_tracked_files(tmp_path) -> None:
    repo = _repo_with_tracked_file(tmp_path)

    payload = build_manifest(repo)

    assert payload["git"]["dirty"] is False
    assert payload["tracked_file_count"] == 1
    assert payload["files"][0]["path"] == "tracked.txt"
    assert len(payload["files"][0]["sha256"]) == 64
    assert len(payload["tracked_tree_sha256"]) == 64
    assert len(payload["environment_lock_sha256"]) == 64
    assert payload["environment_lock_sha256"] == environment_identity_sha256(
        payload["environment_lock"]
    )
    assert payload["environment_lock"]["schema_version"] == 2
    assert payload["environment_lock"]["distributions"] == sorted(
        payload["environment_lock"]["distributions"],
        key=lambda item: __import__("json").dumps(item, sort_keys=True),
    )
    assert payload["environment_lock"]["model_artifacts"]["models"] == []
    assert payload["required_model_ids"] == []

    verified = validate_build_manifest(
        payload,
        repo_root=repo,
        expected_git_sha=payload["git"]["sha"],
        expected_git_diff_sha256=payload["git"]["diff_sha256"],
    )
    assert verified["tracked_tree_sha256"] == payload["tracked_tree_sha256"]


def test_validate_build_manifest_rejects_schema_commit_tree_and_manifest_tampering(
    tmp_path,
) -> None:
    repo = _repo_with_tracked_file(tmp_path)
    payload = build_manifest(repo)
    expected_sha = payload["git"]["sha"]
    expected_diff = payload["git"]["diff_sha256"]

    wrong_schema = {**payload, "schema_version": 1}
    with pytest.raises(ValueError, match="schema_version"):
        validate_build_manifest(
            wrong_schema,
            repo_root=repo,
            expected_git_sha=expected_sha,
            expected_git_diff_sha256=expected_diff,
        )

    tampered = copy.deepcopy(payload)
    tampered["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="tracked_tree_sha256"):
        validate_build_manifest(
            tampered,
            repo_root=repo,
            expected_git_sha=expected_sha,
            expected_git_diff_sha256=expected_diff,
        )

    subprocess.run(["git", "commit", "--allow-empty", "-qm", "other"], cwd=repo, check=True)
    with pytest.raises(ValueError, match="active Git revision"):
        validate_build_manifest(
            payload,
            repo_root=repo,
            expected_git_sha=expected_sha,
            expected_git_diff_sha256=expected_diff,
        )

    other = _repo_with_tracked_file(tmp_path / "other")
    other_payload = build_manifest(other)
    (other / "tracked.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size|SHA-256"):
        validate_build_manifest(
            other_payload,
            repo_root=other,
            expected_git_sha=other_payload["git"]["sha"],
            expected_git_diff_sha256=other_payload["git"]["diff_sha256"],
        )


def test_build_manifest_refuses_dirty_repository(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="clean Git"):
        build_manifest(repo)


def test_build_manifest_discovers_and_embeds_configured_stub_model(tmp_path) -> None:
    repo = tmp_path / "repo"
    config_root = repo / "configs"
    config_root.mkdir(parents=True)
    (config_root / "cell.yaml").write_text(
        "preprocess:\n  plan:\n    steps:\n      - params:\n          model_id: stub:text\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "configs/cell.yaml"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "test"], cwd=repo, check=True)

    payload = build_manifest(repo, config_roots=[config_root])

    assert payload["required_model_ids"] == ["stub:text"]
    assert payload["environment_lock"]["model_artifacts"]["models"][0]["artifact_free"] is True


def test_distribution_inventory_includes_all_visible_distributions(monkeypatch) -> None:
    class FakeDistribution:
        def __init__(self, name: str, version: str, files: dict[str, str | None]) -> None:
            self.metadata = {"Name": name}
            self.version = version
            self._files = files

        def read_text(self, name: str) -> str | None:
            return self._files.get(name)

    distributions = [
        FakeDistribution(
            "Second_Package",
            "2",
            {
                "METADATA": "second metadata",
                "RECORD": "b.py,hash-b,2\na.py,hash-a,1\n",
                "direct_url.json": None,
            },
        ),
        FakeDistribution(
            "first.package",
            "1",
            {
                "METADATA": "first metadata",
                "RECORD": None,
                "direct_url.json": '{"url":"file:///different/install/path","dir_info":{}}',
            },
        ),
    ]
    monkeypatch.setattr(
        "bench.campaign.build_manifest.importlib.metadata.distributions", lambda: distributions
    )

    inventory = _installed_distributions()

    assert {record["name"] for record in inventory} == {"first-package", "second-package"}
    first = next(record for record in inventory if record["name"] == "first-package")
    assert first["direct_url"]["url"] == "file:<local>"
    assert all(len(record["metadata_sha256"]) == 64 for record in inventory)
