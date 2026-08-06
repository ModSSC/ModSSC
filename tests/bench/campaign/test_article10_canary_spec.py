from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from bench.campaign import generate as generate_module
from bench.campaign.errors import CampaignError
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest, sha256_file
from bench.campaign.scientific_gates import evaluate_gate, load_gate_registry
from tools.hpc.slurm_renderer import render_slurm_sites

from .helpers import build_test_campaign, minimal_config, write_yaml


def _clean_placeholder_runtime(**_kwargs: object) -> dict[str, object]:
    return {
        "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
        "git_dirty": False,
        "git_diff_sha256": "0" * 64,
    }


def test_article10_canary_dataset_lock_is_complete_and_concrete() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    lock_path = repo_root / "bench" / "campaigns" / "locks" / "article10-canary-wave1-datasets.yaml"
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert set(payload["datasets"]) == {
        "adult",
        "ag_news",
        "cifar10",
        "cora",
        "speechcommands",
    }
    for identity in payload["datasets"].values():
        for field in ("fingerprint", "content_sha256"):
            value = identity[field]
            assert isinstance(value, str)
            assert len(value) == 64
            int(value, 16)
            assert "REPLACE_WITH_" not in value


def test_article10_canary_waves_are_separate_submit_ready_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )
    spec_root = repo_root / "tools" / "hpc" / "specs"
    lock_path = repo_root / "bench" / "campaigns" / "locks" / "article10-canary-wave1-datasets.yaml"
    locked_datasets = yaml.safe_load(lock_path.read_text(encoding="utf-8"))["datasets"]
    lock_sha256 = sha256_file(lock_path)
    site_paths = (
        repo_root / "tools/hpc/config/profiles/slurm.example.yaml",
        repo_root / "tools/hpc/config/profiles/regional.example.yaml",
    )
    cases = {
        "wave1": {
            "filename": "article10-canary-wave1.example.yaml",
            "campaign_id": "article10-canary-r3-wave1-v1",
            "task_count": 35,
            "methods": {
                "pseudo_label",
                "tri_training",
                "democratic_co_learning",
                "fixmatch",
                "laplace_learning",
                "poisson_learning",
                "grand",
            },
            "profiles": {"a100_dev": 18, "cpu_graph": 5, "v100_dev": 12},
            "sites": {"slurm-gpu": 30, "regional": 5},
            "wrappers": {
                "slurm-gpu/a100_dev.block000.slurm",
                "slurm-gpu/a100_dev.block001.slurm",
                "slurm-gpu/v100_dev.block000.slurm",
                "slurm-gpu/v100_dev.block001.slurm",
                "regional/cpu_graph.slurm",
            },
        },
        "wave2": {
            "filename": "article10-canary-wave2.example.yaml",
            "campaign_id": "article10-canary-r3-wave2-v1",
            "task_count": 15,
            "methods": {"flexmatch", "free_match", "softmatch"},
            "profiles": {"a100_dev": 9, "v100_dev": 6},
            "sites": {"slurm-gpu": 15},
            "wrappers": {
                "slurm-gpu/a100_dev.slurm",
                "slurm-gpu/v100_dev.slurm",
            },
        },
    }
    tasks_by_wave = {}
    all_tasks = []
    for wave, expected in cases.items():
        output_dir = tmp_path / wave
        generated = generate_campaign(
            spec_root / str(expected["filename"]),
            repo_root=repo_root,
            output_dir=output_dir,
            _allow_template_placeholders=True,
        )
        meta, tasks = load_manifest(Path(generated.manifest_path))
        render_slurm_sites(
            site_paths=site_paths,
            campaign_dir=output_dir,
            allow_template_placeholders=True,
        )
        tasks_by_wave[wave] = tasks
        all_tasks.extend(tasks)

        assert generated.task_count == expected["task_count"]
        assert meta["campaign_id"] == expected["campaign_id"]
        assert {task.campaign_id for task in tasks} == {expected["campaign_id"]}
        assert Counter(task.method_id for task in tasks) == {
            method_id: 5 for method_id in expected["methods"]
        }
        assert meta["counts_by_profile"] == expected["profiles"]
        assert meta["counts_by_site"] == expected["sites"]
        wrappers = {
            str(path.relative_to(output_dir / "submit"))
            for path in (output_dir / "submit").glob("*/*.slurm")
        }
        assert wrappers == expected["wrappers"]
        resources = json.loads(
            (output_dir / "profiles" / "resources.json").read_text(encoding="utf-8")
        )
        assert sum(item["task_count"] for item in resources["array_indices"]) == len(tasks)
        assert max(item["task_count"] for item in resources["array_indices"]) <= 10

    method_counts = Counter(task.method_id for task in all_tasks)
    dataset_counts = Counter(task.dataset_id for task in all_tasks)
    modality_counts = Counter(task.modality for task in all_tasks)
    assert len(all_tasks) == 50
    assert set(method_counts.values()) == {5}
    assert set(method_counts) == {
        "pseudo_label",
        "tri_training",
        "democratic_co_learning",
        "fixmatch",
        "flexmatch",
        "free_match",
        "softmatch",
        "laplace_learning",
        "poisson_learning",
        "grand",
    }
    assert dataset_counts == {
        "adult": 10,
        "cora": 10,
        "cifar10": 10,
        "ag_news": 10,
        "speechcommands": 10,
    }
    assert modality_counts == {
        "tabular": 10,
        "graph": 10,
        "vision": 10,
        "text": 10,
        "audio": 10,
    }
    assert {task.regime for task in all_tasks} == {"R3"}
    assert {task.seed for task in all_tasks} == {0}
    assert {task.required_seed_count for task in all_tasks} == {1}
    assert len({task.config_path for task in all_tasks}) == 50
    assert {task.dataset_lock_sha256 for task in all_tasks} == {lock_sha256}
    for task in all_tasks:
        expected_identity = locked_datasets[task.dataset_id]
        assert task.expected_dataset_fingerprint == expected_identity["fingerprint"]
        assert task.expected_dataset_content_sha256 == expected_identity["content_sha256"]

    registry = load_gate_registry(repo_root / "bench" / "campaigns" / "scientific-gates.yaml")
    for task in tasks_by_wave["wave1"]:
        decision = evaluate_gate(
            registry,
            campaign_id=task.campaign_id,
            track=task.track,
            method_id=task.method_id,
            claim_scope_id=task.claim_scope_id,
            campaign_stage=task.campaign_stage,
            claim_eligible=task.claim_eligible,
        )
        assert decision.allowed is True
        assert decision.blockers == ()
    for task in tasks_by_wave["wave2"]:
        decision = evaluate_gate(
            registry,
            campaign_id=task.campaign_id,
            track=task.track,
            method_id=task.method_id,
            claim_scope_id=task.claim_scope_id,
            campaign_stage=task.campaign_stage,
            claim_eligible=task.claim_eligible,
        )
        assert decision.allowed is True
        assert decision.blockers == ()

    registry_payload = yaml.safe_load(
        (repo_root / "bench" / "campaigns" / "scientific-gates.yaml").read_text(encoding="utf-8")
    )
    registry_payload["methods"]["fixmatch"]["algorithmic_conformity"] = "pending"
    pending_registry_path = tmp_path / "scientific-gates-wave-two-pending.yaml"
    write_yaml(pending_registry_path, registry_payload)
    pending_registry = load_gate_registry(pending_registry_path)
    for task in tasks_by_wave["wave2"]:
        decision = evaluate_gate(
            pending_registry,
            campaign_id=task.campaign_id,
            track=task.track,
            method_id=task.method_id,
            claim_scope_id=task.claim_scope_id,
            campaign_stage=task.campaign_stage,
            claim_eligible=task.claim_eligible,
        )
        assert decision.allowed is False
        assert decision.blockers == ("dependency_conformity:fixmatch=pending",)

    registry_payload["methods"]["fixmatch"] = {
        "algorithmic_conformity": "passed",
        "conformity_basis": "pinned_official_implementation",
        "evidence": ["canary/fixmatch-parity.json"],
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-07-23T10:00:00+02:00",
    }
    wave_two_registry_path = tmp_path / "scientific-gates-wave-two.yaml"
    write_yaml(wave_two_registry_path, registry_payload)
    wave_two_registry = load_gate_registry(wave_two_registry_path)
    for task in tasks_by_wave["wave2"]:
        decision = evaluate_gate(
            wave_two_registry,
            campaign_id=task.campaign_id,
            track=task.track,
            method_id=task.method_id,
            claim_scope_id=task.claim_scope_id,
            campaign_stage=task.campaign_stage,
            claim_eligible=task.claim_eligible,
        )
        assert decision.allowed is True
        assert decision.blockers == ()

    for task in all_tasks:
        if task.method_id == "poisson_learning":
            assert (task.resource_profile, task.assigned_site) == ("cpu_graph", "regional")
        elif task.modality in {"vision", "text", "audio"}:
            assert (task.resource_profile, task.assigned_site) == ("a100_dev", "slurm-gpu")
        else:
            assert task.modality in {"tabular", "graph"}
            assert (task.resource_profile, task.assigned_site) == ("v100_dev", "slurm-gpu")


def test_standardized_selection_filters_and_seed_override(tmp_path: Path) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base")
    config_root = repo / "bench" / "configs" / "best"

    cases = (
        ("R1", "vision", "cifar10"),
        ("R3", "tabular", "cifar10"),
        ("R3", "vision", "adult"),
        ("R3", "vision", "cifar10"),
    )
    for regime, modality, dataset_id in cases:
        config = minimal_config(
            output_dir=tmp_path / "outputs" / regime / modality / dataset_id,
            cache_dir=tmp_path / "cache",
        )
        config["run"]["name"] = f"{regime}_{modality}_{dataset_id}"
        config["run"]["seed"] = 7
        config["run"]["seeds"] = [7, 8]
        config["dataset"]["id"] = dataset_id
        write_yaml(
            config_root / regime / "inductive" / "pseudo_label" / modality / f"{dataset_id}.yaml",
            config,
        )

    spec_path = repo / "campaign.yaml"
    raw_spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    raw_spec["selection"].update(
        {
            "regimes": ["R3"],
            "modalities": ["vision"],
            "datasets": ["cifar10"],
            "seeds": [0],
        }
    )
    raw_spec["expect"] = {"config_count": 1, "task_count": 1, "tasks_per_method": 1}
    write_yaml(spec_path, raw_spec)

    generated = generate_campaign(
        spec_path,
        repo_root=repo,
        output_dir=tmp_path / "filtered",
    )
    _, tasks = load_manifest(Path(generated.manifest_path))

    assert len(tasks) == 1
    task = tasks[0]
    assert task.regime == "R3"
    assert task.modality == "vision"
    assert task.dataset_id == "cifar10"
    assert task.seed == 0
    assert task.config_path.endswith("R3/inductive/pseudo_label/vision/cifar10.yaml")


@pytest.mark.parametrize("seeds", [[], [0, 0], [True], [0, "1"], "invalid"])
def test_standardized_selection_rejects_invalid_explicit_seeds(
    tmp_path: Path, seeds: object
) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base")
    spec_path = repo / "campaign.yaml"
    raw_spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    raw_spec["selection"]["seeds"] = seeds
    write_yaml(spec_path, raw_spec)

    with pytest.raises(CampaignError, match="E_CAMPAIGN_SPEC_INVALID"):
        generate_campaign(spec_path, repo_root=repo, output_dir=tmp_path / "invalid")


@pytest.mark.parametrize("field", ["regimes", "modalities", "datasets"])
def test_standardized_selection_rejects_duplicate_filter_values(tmp_path: Path, field: str) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base")
    spec_path = repo / "campaign.yaml"
    raw_spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    raw_spec["selection"][field] = ["duplicate", "duplicate"]
    write_yaml(spec_path, raw_spec)

    with pytest.raises(CampaignError, match=rf"selection\.{field} must contain unique values"):
        generate_campaign(spec_path, repo_root=repo, output_dir=tmp_path / "invalid")
