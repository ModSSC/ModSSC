from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from bench.campaign import generate as generate_module
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from bench.campaign.scientific_gates import evaluate_gate, load_gate_registry
from tools.hpc.slurm_renderer import render_slurm_sites

from .helpers import write_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_ROOT = REPO_ROOT / "tools" / "hpc" / "specs"
SITE_PATH = REPO_ROOT / "tools/hpc/config/profiles/slurm.example.yaml"
GATE_PATH = REPO_ROOT / "bench" / "campaigns" / "scientific-gates.yaml"

CANARY_PATH = SPEC_ROOT / "article10-grand-paper-canary-full.example.yaml"
PRODUCTION_PATH = SPEC_ROOT / "article10-grand-paper-production.example.yaml"
CANARY_ID = "article10-grand-paper-canary-full-v1"
PRODUCTION_ID = "article10-grand-paper-production-v1"
DATASET_FINGERPRINT = "774b6f9406cdd219aabdea52e411cfc28931fc2007878145b472e58e469e1471"
DATASET_CONTENT_SHA256 = "0db92e9492bbe2f32d6d70018fa205b351cb5a44c3f742b2585547e8160922a6"


def _clean_placeholder_runtime(**_kwargs: object) -> dict[str, object]:
    return {
        "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
        "git_dirty": False,
        "git_diff_sha256": "0" * 64,
    }


@pytest.mark.parametrize(
    ("path", "campaign_id", "task_count", "seeds", "profile", "fidelity"),
    [
        (
            CANARY_PATH,
            CANARY_ID,
            3,
            set(range(3)),
            "v100_dev",
            "not_claimable",
        ),
        (
            PRODUCTION_PATH,
            PRODUCTION_ID,
            100,
            set(range(100)),
            "v100_gpu_grand10",
            "paper_matched",
        ),
    ],
)
def test_grand_templates_generate_pinned_literal_seed_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    campaign_id: str,
    task_count: int,
    seeds: set[int],
    profile: str,
    fidelity: str,
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )
    text = path.read_text(encoding="utf-8")
    placeholders = re.findall(r"REPLACE_WITH_[A-Z0-9_]+", text)
    assert placeholders == [
        "REPLACE_WITH_CLEAN_COMMIT",
        "REPLACE_WITH_ENVIRONMENT_LOCK_SHA256",
    ]

    output_dir = tmp_path / campaign_id
    generated = generate_campaign(
        path,
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        _allow_template_placeholders=True,
    )
    meta, tasks = load_manifest(Path(generated.manifest_path))
    render_slurm_sites(site_paths=(SITE_PATH,), campaign_dir=output_dir)

    assert generated.campaign_id == campaign_id
    assert generated.task_count == task_count
    assert meta["counts_by_method"] == {"grand": task_count}
    assert meta["counts_by_profile"] == {profile: task_count}
    assert meta["counts_by_site"] == {"slurm-gpu": task_count}
    assert {task.seed for task in tasks} == seeds
    assert {task.model_seed for task in tasks} == seeds
    assert all(task.model_seed == task.seed for task in tasks)
    assert {task.required_seed_count for task in tasks} == {task_count}
    assert {task.method_id for task in tasks} == {"grand"}
    assert {task.method_profile for task in tasks} == {"paper:feng2020-cora-table1"}
    assert {task.protocol_id for task in tasks} == {"feng-2020-cora-table1-planetoid"}
    assert {task.dataset_id for task in tasks} == {"cora"}
    assert {task.expected_dataset_fingerprint for task in tasks} == {DATASET_FINGERPRINT}
    assert {task.expected_dataset_content_sha256 for task in tasks} == {DATASET_CONTENT_SHA256}
    assert {task.resource_profile for task in tasks} == {profile}
    assert {task.assigned_site for task in tasks} == {"slurm-gpu"}
    assert {task.fidelity_status for task in tasks} == {fidelity}
    wrappers = {
        str(wrapper.relative_to(output_dir / "submit"))
        for wrapper in (output_dir / "submit").glob("*/*.slurm")
    }
    assert wrappers == {f"slurm-gpu/{profile}.slurm"}
    if campaign_id == PRODUCTION_ID:
        wrapper = output_dir / "submit" / "slurm-gpu" / "v100_gpu_grand10.slurm"
        assert "#SBATCH --array=0-99%10" in wrapper.read_text(encoding="utf-8")
        resources = json.loads(
            (output_dir / "profiles" / "resources.json").read_text(encoding="utf-8")
        )
        assert resources["resources"] == [
            {
                "accelerators_per_task": 1,
                "architecture": "V100",
                "configured_walltime_seconds": 72_000,
                "initial_concurrency": 10,
                "max_walltime_seconds": 72_000,
                "profile_id": "v100_gpu_grand10",
                "promoted_concurrency": 10,
                "promotion_max_failure_rate": 0.02,
                "promotion_min_successes": 200,
                "site_id": "slurm-gpu",
            }
        ]


def test_grand_production_profile_is_v100_gpu_with_a_fixed_ten_way_throttle() -> None:
    site = yaml.safe_load(SITE_PATH.read_text(encoding="utf-8"))
    standard = site["profiles"]["v100_gpu"]
    grand = site["profiles"]["v100_gpu_grand10"]

    assert grand["concurrency"] == 10
    assert grand["initial_concurrency"] == 10
    assert grand["promoted_concurrency"] == 10
    for field in (
        "architecture",
        "accelerators_per_task",
        "promotion_min_successes",
        "promotion_max_failure_rate",
        "max_walltime",
        "array_block_size",
        "setup",
        "directives",
    ):
        assert grand[field] == standard[field]


def test_grand_canary_is_nonclaimable_and_production_uses_the_passed_gate(
    tmp_path: Path,
) -> None:
    registry = load_gate_registry(GATE_PATH)

    assert evaluate_gate(
        registry,
        campaign_id=CANARY_ID,
        track="paper",
        method_id="grand",
        campaign_stage="canary",
        claim_eligible=False,
    ).allowed
    production = evaluate_gate(
        registry,
        campaign_id=PRODUCTION_ID,
        track="paper",
        method_id="grand",
    )
    assert production.allowed
    assert production.blockers == ()

    pending_payload = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    pending_payload["methods"]["grand"] = {
        "algorithmic_conformity": "pending",
        "evidence": [],
    }
    pending_path = tmp_path / "pending-grand-gates.yaml"
    write_yaml(pending_path, pending_payload)
    pending_registry = load_gate_registry(pending_path)

    assert evaluate_gate(
        pending_registry,
        campaign_id=CANARY_ID,
        track="paper",
        method_id="grand",
        campaign_stage="canary",
        claim_eligible=False,
    ).allowed
    blocked_production = evaluate_gate(
        pending_registry,
        campaign_id=PRODUCTION_ID,
        track="paper",
        method_id="grand",
    )
    assert not blocked_production.allowed
    assert blocked_production.blockers == ("method_conformity:grand=pending",)
