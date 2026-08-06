from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from bench.campaign.build_manifest import (
    environment_identity_sha256,
    validate_build_manifest,
    validate_environment_lock,
)
from tools.replication_audit.calder.artifacts import (
    CALDER_CONFIGS,
    EFFECTIVE_CONFIG_KIND,
    CalderArtifactError,
    load_calder_config_family,
    materialized_calder_graph_spec,
    verify_calder_artifact_lock,
)
from tools.replication_audit.calder.replay import (
    SOURCE_REPLAY_ORACLE_RELATIVE,
    SOURCE_REPLAY_ORACLE_SHA256,
)

CANARY_CAMPAIGN_ID = "article10-calder-paper-canary-local-v1"
PRODUCTION_CAMPAIGN_ID = "article10-calder-paper-local-v1"
CANARY_FILE_NAME = f"{CANARY_CAMPAIGN_ID}.yaml"
PRODUCTION_FILE_NAME = f"{PRODUCTION_CAMPAIGN_ID}.yaml"

_EXPECTED_SEEDS = list(range(100))
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "artifact_lock_sha256",
    "artifact_builder_git_sha",
    "source_configs",
    "pins",
    "configs",
    "lock_sha256",
}
_CONFIG_RECORD_KEYS = {"path", "repo_path", "sha256"}


class CalderCampaignError(RuntimeError):
    """Raised when immutable local Calder campaign specifications cannot be built."""


@dataclass(frozen=True)
class CalderCampaignSpecs:
    canary_path: str
    production_path: str
    canary_task_count: int
    production_task_count: int
    git_sha: str
    environment_lock_sha256: str
    artifact_lock_sha256: str
    dataset_fingerprint: str
    dataset_content_sha256: str


@dataclass(frozen=True)
class _EffectiveCell:
    method_id: str
    budget: int
    config_path: str
    config_sha256: str

    @property
    def protocol_id(self) -> str:
        short_method = self.method_id.removesuffix("_learning")
        return f"calder-2020-mnist-table1-{short_method}-{self.budget}-label-per-class"


@dataclass(frozen=True)
class _EffectiveBundle:
    cells: tuple[_EffectiveCell, ...]
    manifest_path: str
    manifest_sha256: str
    manifest_lock_sha256: str
    config_sha256: dict[str, str]


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalderCampaignError(f"{label} must be a mapping")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise CalderCampaignError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise CalderCampaignError(f"{label} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalderCampaignError(f"cannot read {label}: {candidate}") from exc
    if not isinstance(raw, dict):
        raise CalderCampaignError(f"{label} root must be a mapping")
    return raw


def _under_generated(path: Path, *, repo_root: Path, label: str) -> Path:
    generated_root = repo_root / "bench" / "generated"
    try:
        resolved_root = generated_root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise CalderCampaignError(
            f"{label} must be below {generated_root.resolve(strict=False)}"
        ) from exc
    if path.is_symlink() or not resolved.is_file():
        raise CalderCampaignError(f"{label} must be a regular non-symlink file: {path}")
    return resolved


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return value == "unlocked" or value.startswith("REPLACE_WITH_")
    if isinstance(value, Mapping):
        return any(_contains_placeholder(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(child) for child in value)
    return False


def _verify_manifest_seal(manifest: Mapping[str, Any]) -> None:
    if set(manifest) != _MANIFEST_KEYS:
        raise CalderCampaignError("effective configuration MANIFEST has unexpected fields")
    digest = _require_sha256(
        manifest.get("lock_sha256"),
        label="effective configuration MANIFEST lock_sha256",
    )
    unsigned = dict(manifest)
    unsigned.pop("lock_sha256")
    if _canonical_sha256(unsigned) != digest:
        raise CalderCampaignError("effective configuration MANIFEST SHA-256 differs")


def _validate_effective_config(
    *,
    repo_root: Path,
    manifest_dir: Path,
    relative: Path,
    record: Mapping[str, Any],
    source_raw: Mapping[str, Any],
    pins: Mapping[str, Any],
) -> _EffectiveCell:
    if set(record) != _CONFIG_RECORD_KEYS:
        raise CalderCampaignError(f"effective config record has unexpected fields: {relative}")
    if record.get("path") != relative.as_posix():
        raise CalderCampaignError(f"effective config record path differs: {relative}")
    repo_path = record.get("repo_path")
    if not isinstance(repo_path, str) or not repo_path:
        raise CalderCampaignError(f"effective config repo_path is missing: {relative}")

    expected_path = manifest_dir / relative
    actual_path = _under_generated(
        repo_root / repo_path,
        repo_root=repo_root,
        label=f"effective config {relative}",
    )
    try:
        expected_resolved = expected_path.resolve(strict=True)
    except OSError as exc:
        raise CalderCampaignError(f"effective config is missing: {expected_path}") from exc
    if actual_path != expected_resolved:
        raise CalderCampaignError(f"effective config is not stored beside its MANIFEST: {relative}")
    expected_digest = _require_sha256(
        record.get("sha256"),
        label=f"effective config SHA-256 for {relative}",
    )
    if _sha256_file(actual_path) != expected_digest:
        raise CalderCampaignError(f"effective config SHA-256 differs: {relative}")
    try:
        raw = yaml.safe_load(actual_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CalderCampaignError(f"cannot read effective config: {relative}") from exc
    if not isinstance(raw, dict):
        raise CalderCampaignError(f"effective config root is not a mapping: {relative}")

    expected_raw = dict(source_raw)
    expected_graph = dict(_mapping(expected_raw.get("graph"), label=f"{relative}.graph"))
    try:
        expected_graph["spec"] = materialized_calder_graph_spec(
            _mapping(expected_graph.get("spec"), label=f"{relative}.graph.spec")
        )
    except CalderArtifactError as exc:
        raise CalderCampaignError(str(exc)) from exc
    expected_graph["expected_fingerprint"] = pins["graph_fingerprint"]
    expected_graph["expected_preprocess_fingerprint"] = pins["preprocess_fingerprint"]
    expected_graph["require_cache_hit"] = True
    expected_raw["graph"] = expected_graph
    if raw != expected_raw:
        raise CalderCampaignError(
            f"effective config differs outside the lock-derived graph pins: {relative}"
        )
    if _contains_placeholder(raw):
        raise CalderCampaignError(f"effective config contains a template placeholder: {relative}")

    method_id = relative.parts[0]
    if method_id not in {"laplace_learning", "poisson_learning"}:
        raise CalderCampaignError(f"unexpected Calder method directory: {relative}")
    match = re.fullmatch(r"mnist-table1-([1-5])-label-per-class\.yaml", relative.name)
    if match is None:
        raise CalderCampaignError(f"unexpected Calder effective config name: {relative}")
    budget = int(match.group(1))
    method = _mapping(raw.get("method"), label=f"{relative}.method")
    dataset = _mapping(raw.get("dataset"), label=f"{relative}.dataset")
    run = _mapping(raw.get("run"), label=f"{relative}.run")
    sampling = _mapping(raw.get("sampling"), label=f"{relative}.sampling")
    plan = _mapping(sampling.get("plan"), label=f"{relative}.sampling.plan")
    labeling = _mapping(plan.get("labeling"), label=f"{relative}.sampling.plan.labeling")
    artifact = _mapping(
        labeling.get("fixed_indices_artifact"),
        label=f"{relative}.fixed_indices_artifact",
    )
    short_method = method_id.removesuffix("_learning")
    expected_profile = f"paper:calder2020-mnist-table1-{short_method}-{budget}-label-per-class"
    if (
        method.get("id") != method_id
        or method.get("profile") != expected_profile
        or dataset.get("id") != "mnist"
        or dataset.get("download") is not False
        or run.get("seed") != 0
        or run.get("seeds") != _EXPECTED_SEEDS
        or labeling.get("mode") != "per_class"
        or labeling.get("value") != budget
        or artifact.get("index_stride") != 5
        or artifact.get("index_offset") != budget - 1
        or artifact.get("expected_size") != budget * 10
        or artifact.get("expected_per_class") != budget
    ):
        raise CalderCampaignError(f"effective config protocol identity differs: {relative}")
    return _EffectiveCell(
        method_id=method_id,
        budget=budget,
        config_path=actual_path.relative_to(repo_root).as_posix(),
        config_sha256=expected_digest,
    )


def _validate_effective_manifest(
    *,
    repo_root: Path,
    manifest_path: Path,
    lock: Mapping[str, Any],
    build_git_sha: str,
) -> _EffectiveBundle:
    resolved_manifest = _under_generated(
        manifest_path,
        repo_root=repo_root,
        label="effective configuration MANIFEST",
    )
    if resolved_manifest.name != "MANIFEST.json":
        raise CalderCampaignError("effective configuration MANIFEST must be named MANIFEST.json")
    manifest = _read_json(resolved_manifest, label="effective configuration MANIFEST")
    _verify_manifest_seal(manifest)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != EFFECTIVE_CONFIG_KIND:
        raise CalderCampaignError("effective configuration MANIFEST schema differs")

    lock_sha256 = _require_sha256(
        lock.get("lock_sha256"),
        label="Calder artifact lock SHA-256",
    )
    if manifest.get("artifact_lock_sha256") != lock_sha256:
        raise CalderCampaignError(
            "effective configuration MANIFEST references a different artifact lock"
        )
    builder = _mapping(lock.get("builder"), label="Calder artifact lock builder")
    lock_git_sha = builder.get("git_sha")
    if (
        not isinstance(lock_git_sha, str)
        or not lock_git_sha
        or manifest.get("artifact_builder_git_sha") != lock_git_sha
        or build_git_sha != lock_git_sha
    ):
        raise CalderCampaignError(
            "build manifest, artifact lock, and effective configs use different commits"
        )

    family = load_calder_config_family(repo_root)
    source_configs = list(family.files)
    if builder.get("config_files") != source_configs:
        raise CalderCampaignError(
            "artifact lock source configurations differ from the active repository"
        )
    if manifest.get("source_configs") != source_configs:
        raise CalderCampaignError(
            "effective configuration MANIFEST source configs differ from the lock"
        )
    lock_pins = _mapping(lock.get("pins"), label="Calder artifact lock pins")
    expected_pins = {
        "preprocess_fingerprint": lock_pins.get("preprocess_fingerprint"),
        "graph_fingerprint": lock_pins.get("graph_fingerprint"),
    }
    pins = _mapping(manifest.get("pins"), label="effective configuration MANIFEST pins")
    if dict(pins) != expected_pins or any(
        not isinstance(value, str) or not value for value in expected_pins.values()
    ):
        raise CalderCampaignError(
            "effective configuration MANIFEST graph pins differ from the artifact lock"
        )

    records = manifest.get("configs")
    if not isinstance(records, list) or len(records) != len(CALDER_CONFIGS):
        raise CalderCampaignError("effective configuration MANIFEST must contain ten configs")
    if [record.get("path") if isinstance(record, Mapping) else None for record in records] != [
        relative.as_posix() for relative in CALDER_CONFIGS
    ]:
        raise CalderCampaignError(
            "effective configuration MANIFEST config order or membership differs"
        )

    cells: list[_EffectiveCell] = []
    reproduction_root = repo_root / "bench" / "configs" / "reproductions"
    for relative, record in zip(CALDER_CONFIGS, records, strict=True):
        if not isinstance(record, Mapping):
            raise CalderCampaignError(f"effective config record is not a mapping: {relative}")
        try:
            source_raw = yaml.safe_load((reproduction_root / relative).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise CalderCampaignError(f"cannot read Calder source config: {relative}") from exc
        if not isinstance(source_raw, dict):
            raise CalderCampaignError(f"Calder source config root is invalid: {relative}")
        cells.append(
            _validate_effective_config(
                repo_root=repo_root,
                manifest_dir=resolved_manifest.parent,
                relative=relative,
                record=record,
                source_raw=source_raw,
                pins=pins,
            )
        )
    return _EffectiveBundle(
        cells=tuple(cells),
        manifest_path=resolved_manifest.relative_to(repo_root).as_posix(),
        manifest_sha256=_sha256_file(resolved_manifest),
        manifest_lock_sha256=str(manifest["lock_sha256"]),
        config_sha256={cell.config_path: cell.config_sha256 for cell in cells},
    )


def _validate_local_build_manifest(
    *, repo_root: Path, manifest: Mapping[str, Any]
) -> tuple[str, str, str]:
    git = _mapping(manifest.get("git"), label="build manifest git")
    git_sha = git.get("sha")
    git_diff_sha256 = git.get("diff_sha256")
    if not isinstance(git_sha, str) or not git_sha:
        raise CalderCampaignError("build manifest has no Git commit")
    if git.get("dirty") is not False:
        raise CalderCampaignError("build manifest was not created from a clean worktree")
    git_diff_sha256 = _require_sha256(
        git_diff_sha256,
        label="build manifest Git worktree fingerprint",
    )
    runtime = _mapping(manifest.get("runtime"), label="build manifest runtime")
    if runtime.get("scheduler_cluster_name") not in {None, ""}:
        raise CalderCampaignError("build manifest must describe a local unscheduled runtime")
    environment = _mapping(
        manifest.get("environment_lock"),
        label="build manifest environment_lock",
    )
    try:
        validate_environment_lock(environment)
    except ValueError as exc:
        raise CalderCampaignError(f"invalid build environment lock: {exc}") from exc
    environment_sha256 = _require_sha256(
        manifest.get("environment_lock_sha256"),
        label="build manifest environment lock SHA-256",
    )
    if environment_identity_sha256(dict(environment)) != environment_sha256:
        raise CalderCampaignError("build manifest environment lock SHA-256 differs")
    try:
        validate_build_manifest(
            manifest,
            repo_root=repo_root,
            expected_git_sha=git_sha,
            expected_git_diff_sha256=git_diff_sha256,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CalderCampaignError(f"build manifest verification failed: {exc}") from exc
    return git_sha, git_diff_sha256, environment_sha256


def _dataset_pins(lock: Mapping[str, Any]) -> tuple[str, str]:
    dataset = _mapping(lock.get("dataset"), label="Calder artifact lock dataset")
    prepared_fingerprint = _require_sha256(
        dataset.get("prepared_fingerprint"),
        label="Calder prepared dataset fingerprint",
    )
    content = _mapping(
        dataset.get("content_evidence"),
        label="Calder source dataset content evidence",
    )
    content_sha256 = _require_sha256(
        content.get("content_sha256"),
        label="Calder source dataset content SHA-256",
    )
    return prepared_fingerprint, content_sha256


def _artifact_evidence(
    *,
    repo_root: Path,
    lock: Mapping[str, Any],
    bundle: _EffectiveBundle,
    dataset_fingerprint: str,
    dataset_content_sha256: str,
) -> dict[str, Any]:
    oracle_path = (repo_root / SOURCE_REPLAY_ORACLE_RELATIVE).resolve(strict=True)
    if (
        not oracle_path.is_relative_to(repo_root)
        or _sha256_file(oracle_path) != SOURCE_REPLAY_ORACLE_SHA256
    ):
        raise CalderCampaignError("Calder source-replay oracle SHA-256 differs")
    pins = _mapping(lock.get("pins"), label="Calder artifact lock pins")
    official = _mapping(
        lock.get("official_evidence"),
        label="Calder artifact lock official_evidence",
    )
    official_commit = str(pins.get("official_commit", ""))
    official_knn_sha256 = _require_sha256(
        pins.get("official_knn_sha256"),
        label="official GraphLearning kNN SHA-256",
    )
    official_permutations_sha256 = _require_sha256(
        pins.get("official_permutations_sha256"),
        label="official GraphLearning permutations SHA-256",
    )
    official_labels_sha256 = _require_sha256(
        official.get("labels_sha256"),
        label="official GraphLearning labels SHA-256",
    )
    if (
        not official_commit
        or official.get("commit") != official_commit
        or official.get("knn_sha256") != official_knn_sha256
        or official.get("permutations_sha256") != official_permutations_sha256
    ):
        raise CalderCampaignError(
            "official GraphLearning evidence differs from the artifact lock pins"
        )
    graph_fingerprint = pins.get("graph_fingerprint")
    preprocess_fingerprint = pins.get("preprocess_fingerprint")
    if (
        not isinstance(graph_fingerprint, str)
        or not graph_fingerprint
        or not isinstance(preprocess_fingerprint, str)
        or not preprocess_fingerprint
    ):
        raise CalderCampaignError("Calder graph/preprocess fingerprints are missing")
    return {
        "artifact_lock_sha256": _require_sha256(
            lock.get("lock_sha256"),
            label="Calder artifact lock SHA-256",
        ),
        "effective_manifest": {
            "path": bundle.manifest_path,
            "sha256": bundle.manifest_sha256,
            "lock_sha256": bundle.manifest_lock_sha256,
        },
        "effective_configs": dict(bundle.config_sha256),
        "source_replay_oracle": {
            "path": SOURCE_REPLAY_ORACLE_RELATIVE.as_posix(),
            "sha256": SOURCE_REPLAY_ORACLE_SHA256,
        },
        "official": {
            "commit": official_commit,
            "labels_sha256": official_labels_sha256,
            "permutations_sha256": official_permutations_sha256,
            "knn_sha256": official_knn_sha256,
        },
        "dataset": {
            "prepared_fingerprint": dataset_fingerprint,
            "content_sha256": dataset_content_sha256,
        },
        "graph": {
            "fingerprint": graph_fingerprint,
            "preprocess_fingerprint": preprocess_fingerprint,
        },
    }


def _campaign_spec(
    *,
    campaign_id: str,
    cells: Sequence[_EffectiveCell],
    selected: Sequence[_EffectiveCell],
    seeds: str | list[int],
    fidelity_status: str,
    task_count: int,
    tasks_per_method: Mapping[str, int],
    git_sha: str,
    git_diff_sha256: str,
    environment_lock_sha256: str,
    dataset_fingerprint: str,
    dataset_content_sha256: str,
    artifact_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    selected_identities = {(cell.method_id, cell.budget) for cell in selected}
    ordered = [cell for cell in cells if (cell.method_id, cell.budget) in selected_identities]
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "track": "paper",
        "default_site": "local-cpu",
        "calder_artifacts": dict(artifact_evidence),
        "code": {
            "git_sha": git_sha,
            "git_diff_sha256": git_diff_sha256,
            "require_clean": True,
            "environment_lock_sha256": environment_lock_sha256,
        },
        "expect": {
            "config_count": len(ordered),
            "task_count": task_count,
            "tasks_per_method": dict(tasks_per_method),
            "tasks_by_profile": {"cpu_graph": task_count},
            "tasks_by_site": {"local-cpu": task_count},
        },
        "cells": [
            {
                "protocol_id": cell.protocol_id,
                "config": cell.config_path,
                "effective_config_sha256": cell.config_sha256,
                "seeds": seeds,
                "resource_profile": "cpu_graph",
                "site": "local-cpu",
                "fidelity_status": fidelity_status,
                "expected_dataset_fingerprint": dataset_fingerprint,
                "expected_dataset_content_sha256": dataset_content_sha256,
            }
            for cell in ordered
        ],
    }


def _render_yaml(payload: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(payload), sort_keys=False)


def _publish_immutable_directory(output_dir: Path, files: Mapping[str, str]) -> None:
    destination = output_dir.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.calder-campaigns.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    lock_fd: int | None = None
    staging: Path | None = None
    try:
        lock_fd = os.open(lock_path, flags, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if os.path.lexists(destination):
            if destination.is_symlink() or not destination.is_dir():
                raise CalderCampaignError(
                    f"immutable campaign output is not a regular directory: {destination}"
                )
            existing = {path.name for path in destination.iterdir()}
            if existing != set(files):
                raise CalderCampaignError(
                    f"refusing to modify existing immutable campaign output: {destination}"
                )
            for name, rendered in files.items():
                path = destination / name
                if path.is_symlink() or not path.is_file():
                    raise CalderCampaignError(
                        f"immutable campaign output contains an invalid file: {path}"
                    )
                if path.read_text(encoding="utf-8") != rendered:
                    raise CalderCampaignError(
                        f"refusing to replace different immutable campaign spec: {path}"
                    )
            return

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-",
                dir=destination.parent,
            )
        )
        for name, rendered in files.items():
            path = staging / name
            with path.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())
        os.rename(staging, destination)
        staging = None
    except CalderCampaignError:
        raise
    except (OSError, UnicodeError) as exc:
        raise CalderCampaignError(
            f"cannot publish immutable Calder campaign specs: {destination}"
        ) from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)


def generate_calder_campaign_specs(
    *,
    repo_root: Path,
    artifact_lock_path: Path,
    effective_manifest_path: Path,
    build_manifest_path: Path,
    output_dir: Path,
) -> CalderCampaignSpecs:
    """Verify all Calder identities and publish local canary/production specs once."""

    root = repo_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise CalderCampaignError(f"repository root is not a directory: {root}")
    lock = _read_json(artifact_lock_path, label="Calder artifact lock")
    try:
        verify_calder_artifact_lock(lock)
    except (CalderArtifactError, OSError, ValueError) as exc:
        raise CalderCampaignError(f"Calder artifact lock verification failed: {exc}") from exc
    build_manifest = _read_json(build_manifest_path, label="local build manifest")
    git_sha, git_diff_sha256, environment_sha256 = _validate_local_build_manifest(
        repo_root=root,
        manifest=build_manifest,
    )
    bundle = _validate_effective_manifest(
        repo_root=root,
        manifest_path=effective_manifest_path,
        lock=lock,
        build_git_sha=git_sha,
    )
    dataset_fingerprint, dataset_content_sha256 = _dataset_pins(lock)
    artifact_evidence = _artifact_evidence(
        repo_root=root,
        lock=lock,
        bundle=bundle,
        dataset_fingerprint=dataset_fingerprint,
        dataset_content_sha256=dataset_content_sha256,
    )

    canary_cells = [cell for cell in bundle.cells if cell.budget in {1, 5}]
    if len(canary_cells) != 4:
        raise CalderCampaignError("Calder canary selection must contain exactly four configs")
    canary = _campaign_spec(
        campaign_id=CANARY_CAMPAIGN_ID,
        cells=bundle.cells,
        selected=canary_cells,
        seeds=[0],
        fidelity_status="not_claimable",
        task_count=4,
        tasks_per_method={"laplace_learning": 2, "poisson_learning": 2},
        git_sha=git_sha,
        git_diff_sha256=git_diff_sha256,
        environment_lock_sha256=environment_sha256,
        dataset_fingerprint=dataset_fingerprint,
        dataset_content_sha256=dataset_content_sha256,
        artifact_evidence=artifact_evidence,
    )
    production = _campaign_spec(
        campaign_id=PRODUCTION_CAMPAIGN_ID,
        cells=bundle.cells,
        selected=bundle.cells,
        seeds="from_config",
        fidelity_status="paper_matched",
        task_count=1000,
        tasks_per_method={"laplace_learning": 500, "poisson_learning": 500},
        git_sha=git_sha,
        git_diff_sha256=git_diff_sha256,
        environment_lock_sha256=environment_sha256,
        dataset_fingerprint=dataset_fingerprint,
        dataset_content_sha256=dataset_content_sha256,
        artifact_evidence=artifact_evidence,
    )
    rendered = {
        CANARY_FILE_NAME: _render_yaml(canary),
        PRODUCTION_FILE_NAME: _render_yaml(production),
    }
    _publish_immutable_directory(output_dir, rendered)
    destination = output_dir.expanduser().resolve(strict=False)
    return CalderCampaignSpecs(
        canary_path=str(destination / CANARY_FILE_NAME),
        production_path=str(destination / PRODUCTION_FILE_NAME),
        canary_task_count=4,
        production_task_count=1000,
        git_sha=git_sha,
        environment_lock_sha256=environment_sha256,
        artifact_lock_sha256=str(lock["lock_sha256"]),
        dataset_fingerprint=dataset_fingerprint,
        dataset_content_sha256=dataset_content_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.replication_audit.calder.campaigns",
        description=(
            "Verify Calder Table 1 locks and emit immutable local canary/production specs."
        ),
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-lock", type=Path, required=True)
    parser.add_argument("--effective-manifest", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = generate_calder_campaign_specs(
            repo_root=args.repo_root,
            artifact_lock_path=args.artifact_lock,
            effective_manifest_path=args.effective_manifest,
            build_manifest_path=args.build_manifest,
            output_dir=args.output_dir,
        )
    except CalderCampaignError as exc:
        parser.exit(2, f"calder-campaigns: {exc}\n")
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANARY_CAMPAIGN_ID",
    "PRODUCTION_CAMPAIGN_ID",
    "CalderCampaignError",
    "CalderCampaignSpecs",
    "generate_calder_campaign_specs",
]
