from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bench.campaign import generate as generate_module
from bench.campaign.dcl_partition_lock import (
    load_dcl_partition_selection,
    verify_dcl_partition_replay,
)
from bench.campaign.errors import CampaignError
from bench.campaign.executor import (
    _inject_and_verify_partition_replay,
    _validate_sampling_replay,
)
from bench.campaign.generate import generate_campaign
from bench.campaign.governance import _check_frozen_partition_replays
from bench.campaign.manifest import derive_task_id, load_manifest, sha256_file
from bench.schema import ExperimentConfig
from bench.seed_sweep import apply_global_seed
from bench.utils.io import load_yaml

from .helpers import write_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_RELATIVE = Path("bench/campaigns/locks/dcl-vote-zhou-goldman-2004-v1/selected-partitions.json")
REPLAY_ROOT_RELATIVE = Path("bench/campaigns/locks/dcl-vote-zhou-goldman-2004-v1/splits")
LOCK_PATH = REPO_ROOT / LOCK_RELATIVE
VOTE_CONFIG = REPO_ROOT / "bench/configs/reproductions/democratic_co_learning/vote.yaml"
LOCK_SHA256 = "5f586b2ab21bd6c2b0e058ab9d588ec1fc04b41b7d93e5a125d0a5f2ea1b36fb"
DATASET_FINGERPRINT = "98f2cf80ea8e8fb8f3f546dc87d3a231a0ec10fe6d26b5dfe490fc832079b0dd"
DATASET_CONTENT_SHA256 = "5b95c771651aa62b985332026f63b423d7fe7dff2f0bc90ef2c336d4d2b70130"
SOURCE_ARTIFACT_SHA256 = {
    "selected-partitions.json": "efa80d397d70dd6d9679d6414a99069a1ef7578a7a28ab865a65eccd9e075043",
    "source/manifest.jsonl": "08c2d658c8dd3ba821439bb3f2694dcb8ea46dec316c331fafb92e6d6b3be123",
    "source/manifest.meta.json": (
        "7ced19045325778bb1db6121b582aada81cb315443e4adb151f854d4cdbd8a6e"
    ),
    "source/reconcile.json": "c13e2c65a353e5c530e748076c30cb671065312459dd3e665f0ef2e8ba3cf7a1",
}
PARTITION_SELECTION_KEYS = {
    "kind",
    "selection_path",
    "selection_sha256",
    "selection_rank",
    "source_task_id",
    "source_task_row_sha256",
    "replay_path",
    "split_fingerprint",
    "split_manifest_sha256",
    "split_json_sha256",
    "split_arrays_sha256",
}


def _clean_placeholder_runtime(**_kwargs: object) -> dict[str, object]:
    return {
        "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
        "git_dirty": False,
        "git_diff_sha256": "0" * 64,
    }


def _write_spec(
    path: Path,
    *,
    selection_sha256: str = LOCK_SHA256,
    replay_root: str = REPLAY_ROOT_RELATIVE.as_posix(),
    claim_eligible: bool = False,
) -> None:
    write_yaml(
        path,
        {
            "schema_version": 1,
            "campaign_id": "article10-dcl-vote-lock-test",
            "track": "paper",
            "scientific_scope": {
                "claim_scope_id": "article10",
                "stage": "production",
                "claim_eligible": claim_eligible,
                "gate_policy_id": "modssc-scientific-gates-v2",
                "gate_policy_sha256": "from_registry",
            },
            "default_site": "regional",
            "code": {
                "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
                "require_clean": True,
                "environment_lock_sha256": "REPLACE_WITH_ENVIRONMENT_LOCK_SHA256",
            },
            "expect": {
                "config_count": 1,
                "task_count": 20,
                "tasks_per_method": {"democratic_co_learning": 20},
                "tasks_by_profile": {"cpu_tabular": 20},
                "tasks_by_site": {"regional": 20},
            },
            "cells": [
                {
                    "protocol_id": "zhou-goldman-2004-vote-table3",
                    "config": ("bench/configs/reproductions/democratic_co_learning/vote.yaml"),
                    "seeds": "from_partition_selection",
                    "partition_selection": {
                        "path": LOCK_RELATIVE.as_posix(),
                        "sha256": selection_sha256,
                        "replay_root": replay_root,
                    },
                    "resource_profile": "cpu_tabular",
                    "site": "regional",
                    "fidelity_status": "not_claimable",
                    "expected_dataset_fingerprint": DATASET_FINGERPRINT,
                    "expected_dataset_content_sha256": DATASET_CONTENT_SHA256,
                }
            ],
        },
    )


def test_claim_eligible_generation_requires_private_source_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )
    spec = tmp_path / "claim-eligible.yaml"
    _write_spec(spec, claim_eligible=True)

    with pytest.raises(CampaignError, match="PRIVATE_REQUIRED"):
        generate_campaign(
            spec,
            repo_root=REPO_ROOT,
            output_dir=tmp_path / "campaign",
            _allow_template_placeholders=True,
        )


def _selected_rows() -> dict[int, dict[str, Any]]:
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return {int(row["seed"]): row for row in payload["selected"]}


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _generate_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> list[Any]:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )
    spec = tmp_path / "paper.yaml"
    _write_spec(spec)
    generated = generate_campaign(
        spec,
        repo_root=REPO_ROOT,
        output_dir=tmp_path / "campaign",
        _allow_template_placeholders=True,
    )
    return load_manifest(Path(generated.manifest_path))[1]


def test_paper_generation_consumes_locked_dcl_vote_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )
    spec = tmp_path / "paper.yaml"
    _write_spec(spec)

    generated = generate_campaign(
        spec,
        repo_root=REPO_ROOT,
        output_dir=tmp_path / "campaign",
        _allow_template_placeholders=True,
    )
    _, tasks = load_manifest(Path(generated.manifest_path))
    selected = _selected_rows()

    assert generated.task_count == 20
    assert [task.seed for task in tasks] == list(range(1, 21))
    assert {task.schema_version for task in tasks} == {4}
    assert {task.expected_dataset_fingerprint for task in tasks} == {DATASET_FINGERPRINT}
    assert {task.expected_dataset_content_sha256 for task in tasks} == {DATASET_CONTENT_SHA256}

    for task in tasks:
        evidence = task.partition_selection
        source = selected[task.seed]

        assert evidence is not None
        assert set(evidence) == PARTITION_SELECTION_KEYS
        assert evidence["selection_sha256"] == LOCK_SHA256
        assert Path(str(evidence["selection_path"])).as_posix() == LOCK_RELATIVE.as_posix()
        assert evidence["selection_rank"] == task.seed
        assert evidence["source_task_id"] == source["task_id"]
        assert evidence["source_task_row_sha256"] == source["task_row_sha256"]
        assert evidence["split_fingerprint"] == source["split_fingerprint"]
        assert evidence["split_manifest_sha256"] == source["split_manifest_sha256"]
        assert evidence["split_json_sha256"] == source["split_json_sha256"]
        assert evidence["split_arrays_sha256"] == source["split_arrays_sha256"]
        assert task.expected_split_fingerprint == source["split_fingerprint"]

        replay_path = _resolve_repo_path(str(evidence["replay_path"]))
        assert replay_path == REPO_ROOT / REPLAY_ROOT_RELATIVE / f"seed-{task.seed:03d}"
        assert replay_path.is_dir()
        assert sha256_file(replay_path / "MANIFEST.json") == evidence["split_manifest_sha256"]
        assert sha256_file(replay_path / "split.json") == evidence["split_json_sha256"]
        assert sha256_file(replay_path / "arrays.npz") == evidence["split_arrays_sha256"]
        replay_manifest = json.loads((replay_path / "MANIFEST.json").read_text(encoding="utf-8"))
        assert replay_manifest["split_fingerprint"] == evidence["split_fingerprint"]
        assert replay_manifest["files"]["split.json"]["sha256"] == evidence["split_json_sha256"]
        assert replay_manifest["files"]["arrays.npz"]["sha256"] == evidence["split_arrays_sha256"]

    first_payload = tasks[0].to_dict()
    changed_lock = deepcopy(first_payload)
    assert changed_lock["partition_selection"] is not None
    changed_lock["partition_selection"]["selection_sha256"] = "f" * 64
    assert derive_task_id(changed_lock) != tasks[0].task_id


def test_paper_generation_rejects_wrong_selection_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )
    spec = tmp_path / "wrong-digest.yaml"
    _write_spec(spec, selection_sha256="0" * 64)

    with pytest.raises(CampaignError):
        generate_campaign(
            spec,
            repo_root=REPO_ROOT,
            output_dir=tmp_path / "campaign",
            _allow_template_placeholders=True,
        )


def test_selection_loader_exposes_non_claimable_public_descriptor(tmp_path: Path) -> None:
    lock = load_dcl_partition_selection(
        LOCK_PATH,
        expected_sha256=LOCK_SHA256,
        expected_dataset_fingerprint=DATASET_FINGERPRINT,
        expected_dataset_content_sha256=DATASET_CONTENT_SHA256,
    )
    assert lock.source_artifact_sha256 == SOURCE_ARTIFACT_SHA256
    assert lock.claim_eligible is False
    assert lock.source_uri == "evidence://historical/dcl-vote-zhou-goldman-2004-v1/raw-v1"
    assert not (LOCK_PATH.parent / "source").exists()

    copied = tmp_path / "lock"
    copied.mkdir()
    shutil.copy2(LOCK_PATH, copied / "selected-partitions.json")
    copied_lock = copied / "selected-partitions.json"
    payload = json.loads(copied_lock.read_text(encoding="utf-8"))
    payload["provenance"]["artifact_sha256"]["source/manifest.meta.json"] = "0" * 64
    copied_lock.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="source artifact digests differ"):
        load_dcl_partition_selection(
            copied_lock,
            expected_sha256=sha256_file(copied_lock),
            expected_dataset_fingerprint=DATASET_FINGERPRINT,
            expected_dataset_content_sha256=DATASET_CONTENT_SHA256,
        )


def test_selection_loader_rejects_a_rewritten_selection_payload(tmp_path: Path) -> None:
    copied_lock = tmp_path / "selected-partitions.json"
    shutil.copy2(LOCK_PATH, copied_lock)
    payload = json.loads(copied_lock.read_text(encoding="utf-8"))
    payload["selected"][0]["pseudo_labels_added_total"] += 1
    payload["evaluated_candidates"] = deepcopy(payload["selected"])
    copied_lock.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="logical content address"):
        load_dcl_partition_selection(
            copied_lock,
            expected_sha256=sha256_file(copied_lock),
            expected_dataset_fingerprint=DATASET_FINGERPRINT,
            expected_dataset_content_sha256=DATASET_CONTENT_SHA256,
        )


def test_paper_generation_rejects_missing_partition_replay_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )
    spec = tmp_path / "missing-replay.yaml"
    _write_spec(spec, replay_root="bench/campaigns/locks/missing-dcl-replays")

    with pytest.raises(CampaignError):
        generate_campaign(
            spec,
            repo_root=REPO_ROOT,
            output_dir=tmp_path / "campaign",
            _allow_template_placeholders=True,
        )


def test_partition_replay_verification_rejects_tampered_locked_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )
    spec = tmp_path / "paper.yaml"
    _write_spec(spec)
    generated = generate_campaign(
        spec,
        repo_root=REPO_ROOT,
        output_dir=tmp_path / "campaign",
        _allow_template_placeholders=True,
    )
    _, tasks = load_manifest(Path(generated.manifest_path))
    evidence = dict(tasks[0].partition_selection or {})
    replay = tmp_path / "replay"
    shutil.copytree(REPO_ROOT / REPLAY_ROOT_RELATIVE / "seed-001", replay)
    (replay / "arrays.npz").write_bytes(b"tampered")
    evidence["selection_path"] = str(LOCK_PATH)
    evidence["replay_path"] = str(replay)
    plan = load_yaml(VOTE_CONFIG)["sampling"]["plan"]

    with pytest.raises(CampaignError, match="E_CAMPAIGN_PARTITION_REPLAY_MISMATCH"):
        verify_dcl_partition_replay(
            evidence,
            expected_seed=1,
            expected_dataset_fingerprint=DATASET_FINGERPRINT,
            expected_plan=plan,
        )


def test_executor_injects_only_the_manifest_bound_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _generate_tasks(tmp_path, monkeypatch)[0]
    raw = load_yaml(VOTE_CONFIG)
    effective = apply_global_seed(
        raw,
        seed=task.seed,
        seeded_sections=task.seeded_sections,
    )

    _inject_and_verify_partition_replay(task, effective, repo_root=REPO_ROOT)

    replay = effective["sampling"]["replay"]
    assert Path(replay["selection_path"]).is_absolute()
    assert Path(replay["replay_path"]).is_absolute()
    assert replay["selection_sha256"] == LOCK_SHA256
    assert replay["split_fingerprint"] == task.expected_split_fingerprint
    assert ExperimentConfig.from_dict(effective).sampling.replay == replay

    tampered_evidence = {
        **(task.partition_selection or {}),
        "split_arrays_sha256": "0" * 64,
    }
    tampered_task = replace(task, partition_selection=tampered_evidence)
    second_effective = apply_global_seed(
        raw,
        seed=task.seed,
        seeded_sections=task.seeded_sections,
    )
    with pytest.raises(CampaignError, match="PARTITION_SELECTION_MISMATCH"):
        _inject_and_verify_partition_replay(
            tampered_task,
            second_effective,
            repo_root=REPO_ROOT,
        )


def test_preflight_attests_all_replays_and_fails_closed_on_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = _generate_tasks(tmp_path, monkeypatch)
    configs = {tasks[0].config_path: load_yaml(VOTE_CONFIG)}

    errors, attestations = _check_frozen_partition_replays(
        tasks,
        configs,
        repo_root=REPO_ROOT,
    )

    assert errors == []
    assert len(attestations) == 20
    assert {record["selection_rank"] for record in attestations} == set(range(1, 21))
    assert {record["selection_sha256"] for record in attestations} == {LOCK_SHA256}

    tampered = replace(
        tasks[0],
        partition_selection={
            **(tasks[0].partition_selection or {}),
            "split_json_sha256": "0" * 64,
        },
    )
    errors, attestations = _check_frozen_partition_replays(
        [tampered],
        configs,
        repo_root=REPO_ROOT,
    )

    assert attestations == []
    assert "PARTITION_SELECTION_MISMATCH" in "\n".join(errors)


def test_result_validation_requires_byte_identical_selected_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _generate_tasks(tmp_path, monkeypatch)[0]
    evidence = task.partition_selection
    assert evidence is not None
    run_dir = tmp_path / "run"
    replay_dir = run_dir / "sampling_split"
    shutil.copytree(REPO_ROOT / evidence["replay_path"], replay_dir)
    replay = {
        "format": "modssc.sampling.storage.v1",
        "path": "sampling_split",
        "manifest": "MANIFEST.json",
        "manifest_sha256": evidence["split_manifest_sha256"],
        "selection": {
            "kind": evidence["kind"],
            "selection_sha256": evidence["selection_sha256"],
            "selection_rank": evidence["selection_rank"],
            "source_task_id": evidence["source_task_id"],
            "source_task_row_sha256": evidence["source_task_row_sha256"],
        },
    }

    _validate_sampling_replay(
        run_json_path=run_dir / "run.json",
        replay=replay,
        dataset_fingerprint=DATASET_FINGERPRINT,
        split_fingerprint=evidence["split_fingerprint"],
        task_id=task.task_id,
        partition_selection=evidence,
    )

    (replay_dir / "arrays.npz").write_bytes(b"tampered")
    with pytest.raises(CampaignError, match="RESULT_INVALID"):
        _validate_sampling_replay(
            run_json_path=run_dir / "run.json",
            replay=replay,
            dataset_fingerprint=DATASET_FINGERPRINT,
            split_fingerprint=evidence["split_fingerprint"],
            task_id=task.task_id,
            partition_selection=evidence,
        )
