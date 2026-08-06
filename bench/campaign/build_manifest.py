from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.utils.io import atomic_write_json
from bench.utils.runtime import collect_runtime_versions

from .model_artifacts import (
    build_model_artifact_lock,
    discover_model_ids,
    model_artifact_lock_sha256,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return sorted(path for path in result.stdout.decode("utf-8").split("\0") if path)


def _tracked_file_records(repo_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in _tracked_files(repo_root):
        path = repo_root / relative
        if not path.is_file():
            raise RuntimeError(f"tracked file is missing or is not a regular file: {relative}")
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _tracked_tree_sha256(files: Sequence[Mapping[str, Any]]) -> str:
    tree = hashlib.sha256()
    for record in files:
        tree.update(str(record["path"]).encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(record["sha256"]).encode("ascii"))
        tree.update(b"\n")
    return tree.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_build_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_git_sha: str,
    expected_git_diff_sha256: str | None,
) -> dict[str, Any]:
    """Verify a build manifest against the campaign and active Git checkout."""

    if manifest.get("schema_version") != 2:
        raise ValueError("build manifest schema_version must equal 2")
    git = manifest.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("build manifest git payload is missing")
    if git.get("sha") != expected_git_sha:
        raise ValueError("build manifest Git revision differs from the campaign")
    if git.get("dirty") is not False:
        raise ValueError("build manifest was not created from a clean worktree")
    if git.get("diff_sha256") != expected_git_diff_sha256:
        raise ValueError("build manifest worktree fingerprint differs from the campaign")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("build manifest files must be a list")
    declared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in raw_files:
        if not isinstance(record, Mapping):
            raise ValueError("build manifest file record must be a mapping")
        relative = record.get("path")
        size = record.get("size")
        digest = record.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
        ):
            raise ValueError("build manifest contains an invalid tracked path")
        if relative in seen:
            raise ValueError(f"build manifest contains duplicate tracked path: {relative}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"build manifest contains an invalid size for {relative}")
        if not _is_sha256(digest):
            raise ValueError(f"build manifest contains an invalid SHA-256 for {relative}")
        seen.add(relative)
        declared.append({"path": relative, "size": size, "sha256": digest})
    if [record["path"] for record in declared] != sorted(seen):
        raise ValueError("build manifest tracked files are not canonically ordered")
    if manifest.get("tracked_file_count") != len(declared):
        raise ValueError("build manifest tracked_file_count is invalid")
    declared_tree = _tracked_tree_sha256(declared)
    if manifest.get("tracked_tree_sha256") != declared_tree:
        raise ValueError("build manifest tracked_tree_sha256 is invalid")

    repo_root = repo_root.resolve()
    try:
        active_files = _tracked_file_records(repo_root)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot inventory the active tracked tree: {exc}") from exc
    if [record["path"] for record in active_files] != [record["path"] for record in declared]:
        raise ValueError("active tracked file set differs from the build manifest")
    for expected, actual in zip(declared, active_files, strict=True):
        if expected["size"] != actual["size"]:
            raise ValueError(f"active tracked file size differs: {expected['path']}")
        if expected["sha256"] != actual["sha256"]:
            raise ValueError(f"active tracked file SHA-256 differs: {expected['path']}")
    if _tracked_tree_sha256(active_files) != declared_tree:  # defensive consistency check
        raise ValueError("active tracked tree digest differs from the build manifest")

    versions = collect_runtime_versions(repo_root=repo_root)
    if versions.get("git_sha") != expected_git_sha:
        raise ValueError("active Git revision differs from the campaign")
    if versions.get("git_dirty") is not False:
        raise ValueError("active Git worktree is dirty")
    if versions.get("git_diff_sha256") != expected_git_diff_sha256:
        raise ValueError("active Git worktree fingerprint differs from the campaign")
    return {
        "git_sha": expected_git_sha,
        "git_diff_sha256": expected_git_diff_sha256,
        "tracked_file_count": len(active_files),
        "tracked_tree_sha256": declared_tree,
    }


def _digest_metadata_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_direct_url(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return {"invalid_sha256": _digest_metadata_text(value)}
    if not isinstance(raw, dict):
        return {"invalid_sha256": _digest_metadata_text(value)}
    url = raw.get("url")
    if isinstance(url, str) and url.startswith("file:"):
        # Local editable paths are deployment locations, not environment
        # identities.  Git and the tracked-tree digest lock ModSSC's code.
        raw["url"] = "file:<local>"
    return raw


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _installed_distributions() -> list[dict[str, Any]]:
    """Inventory every visible Python distribution in deterministic order."""

    records: list[dict[str, Any]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or "UNKNOWN"
        record = distribution.read_text("RECORD")
        if record is not None:
            # RECORD ordering is not semantically relevant.  Sorting its rows
            # preserves the declared per-file hashes while remaining stable.
            record = "\n".join(sorted(line for line in record.splitlines() if line))
        records.append(
            {
                "name": _canonical_distribution_name(str(name)),
                "version": str(distribution.version),
                "metadata_sha256": _digest_metadata_text(distribution.read_text("METADATA")),
                "record_sha256": _digest_metadata_text(record),
                "direct_url": _normalized_direct_url(distribution.read_text("direct_url.json")),
            }
        )
    return sorted(records, key=lambda item: json.dumps(item, sort_keys=True))


def _empty_model_lock() -> dict[str, Any]:
    return {"schema_version": 1, "models": []}


def collect_environment_identity(
    *, model_artifact_lock: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the stable runtime fields covered by the environment lock.

    Operational execution identifiers, loaded modules, kernel releases and
    hostnames are intentionally recorded elsewhere: they may change between
    staging and execution without changing the prepared Python environment.
    """

    models = dict(model_artifact_lock or _empty_model_lock())
    return {
        "schema_version": 2,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "distributions": _installed_distributions(),
        "model_artifacts": models,
        "model_artifacts_sha256": model_artifact_lock_sha256(models),
    }


def environment_identity_sha256(identity: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def python_environment_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the environment fields that can be checked without model I/O."""

    return {
        key: identity.get(key)
        for key in ("schema_version", "python", "implementation", "machine", "distributions")
    }


def validate_environment_lock(identity: Mapping[str, Any]) -> None:
    if identity.get("schema_version") != 2:
        raise ValueError("environment_lock schema_version must equal 2")
    model_lock = identity.get("model_artifacts")
    if not isinstance(model_lock, Mapping):
        raise ValueError("environment_lock has no model_artifacts payload")
    if identity.get("model_artifacts_sha256") != model_artifact_lock_sha256(model_lock):
        raise ValueError("environment_lock model artifact digest is invalid")


def build_manifest(
    repo_root: Path,
    *,
    require_clean: bool = True,
    config_roots: Sequence[Path] | None = None,
    model_ids: Sequence[str] = (),
    model_cache_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    versions = collect_runtime_versions(repo_root=repo_root)
    if require_clean and versions.get("git_dirty") is not False:
        raise RuntimeError("build manifest requires a clean Git worktree")

    files = _tracked_file_records(repo_root)

    selected_config_roots = list(config_roots or ())
    if config_roots is None:
        default_config_root = repo_root / "bench" / "configs"
        if default_config_root.is_dir():
            selected_config_roots.append(default_config_root)
    required_model_ids = sorted(set(model_ids).union(discover_model_ids(selected_config_roots)))
    model_lock = build_model_artifact_lock(
        required_model_ids,
        model_cache_root=model_cache_root,
    )
    environment_lock = collect_environment_identity(model_artifact_lock=model_lock)
    runtime = {
        **environment_lock,
        "executable": sys.executable,
        "python_prefix": sys.prefix,
        "platform": platform.platform(),
        "loaded_modules": os.environ.get("LOADEDMODULES"),
    }
    environment_lock_sha256 = environment_identity_sha256(environment_lock)
    return {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "git": {
            "sha": versions.get("git_sha"),
            "dirty": versions.get("git_dirty"),
            "diff_sha256": versions.get("git_diff_sha256"),
        },
        "runtime": runtime,
        "environment_lock": environment_lock,
        "environment_lock_sha256": environment_lock_sha256,
        "model_artifacts_sha256": environment_lock["model_artifacts_sha256"],
        "required_model_ids": required_model_ids,
        "tracked_file_count": len(files),
        "tracked_tree_sha256": _tracked_tree_sha256(files),
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create an authenticated ModSSC build manifest")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--config-root",
        type=Path,
        action="append",
        default=None,
        help="YAML file/directory used to discover required model_id values",
    )
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--model-cache-root", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = build_manifest(
        args.repo_root,
        require_clean=not args.allow_dirty,
        config_roots=args.config_root,
        model_ids=args.model_id,
        model_cache_root=args.model_cache_root,
    )
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "git_sha": payload["git"]["sha"],
                "environment_lock_sha256": payload["environment_lock_sha256"],
                "tracked_tree_sha256": payload["tracked_tree_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
