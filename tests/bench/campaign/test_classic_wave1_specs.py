from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from bench.campaign import generate as generate_module
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from bench.campaign.scientific_gates import evaluate_gate, load_gate_registry
from tools.hpc.slurm_renderer import render_slurm_sites

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_ROOT = REPO_ROOT / "tools" / "hpc" / "specs"
SITE_PATH = REPO_ROOT / "tools/hpc/config/profiles/slurm.example.yaml"
GATE_PATH = REPO_ROOT / "bench" / "campaigns" / "scientific-gates.yaml"
CANARY_EVIDENCE_PATH = REPO_ROOT / "provenance/article10/evidence/easy-wave1-v5-canaries.json"

VOTE_FINGERPRINT = "98f2cf80ea8e8fb8f3f546dc87d3a231a0ec10fe6d26b5dfe490fc832079b0dd"
VOTE_CONTENT = "5b95c771651aa62b985332026f63b423d7fe7dff2f0bc90ef2c336d4d2b70130"
CASES: dict[str, dict[str, Any]] = {
    "article10-paper-tri-vote-canary-v1.example.yaml": {
        "campaign_id": "article10-paper-tri-vote-canary-v1",
        "method_id": "tri_training",
        "protocol_id": "zhou-li-2005-vote-table3-j48-80pct-unlabeled",
        "config": "bench/configs/reproductions/tri_training/vote_table3_j48.yaml",
        "seeds": [1],
        "profile": "v100_dev",
        "fidelity": "not_claimable",
        "fingerprint": VOTE_FINGERPRINT,
        "content": VOTE_CONTENT,
        "canary": True,
    },
    "article10-paper-tri-vote-v1.example.yaml": {
        "campaign_id": "article10-paper-tri-vote-v1",
        "method_id": "tri_training",
        "protocol_id": "zhou-li-2005-vote-table3-j48-80pct-unlabeled",
        "config": "bench/configs/reproductions/tri_training/vote_table3_j48.yaml",
        "seeds": [1, 2, 3],
        "profile": "v100_gpu",
        "fidelity": "paper_approx",
        "fingerprint": VOTE_FINGERPRINT,
        "content": VOTE_CONTENT,
        "canary": False,
    },
}


def _clean_placeholder_runtime(**_kwargs: object) -> dict[str, object]:
    return {
        "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
        "git_dirty": False,
        "git_diff_sha256": "0" * 64,
    }


def _placeholder_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [
            placeholder for child in value.values() for placeholder in _placeholder_values(child)
        ]
    if isinstance(value, list):
        return [placeholder for child in value for placeholder in _placeholder_values(child)]
    if isinstance(value, str) and value.startswith("REPLACE_WITH_"):
        return [value]
    return []


def test_classic_wave1_templates_pin_everything_except_commit_and_environment() -> None:
    for spec_name, expected in CASES.items():
        path = SPEC_ROOT / spec_name
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["schema_version"] == 1
        assert raw["campaign_id"] == expected["campaign_id"]
        assert raw["track"] == "paper"
        assert raw["default_site"] == "slurm-gpu"
        assert _placeholder_values(raw) == [
            "REPLACE_WITH_CLEAN_COMMIT",
            "REPLACE_WITH_ENVIRONMENT_LOCK_SHA256",
        ]
        assert raw["expect"] == {
            "config_count": 1,
            "task_count": len(expected["seeds"]),
            "tasks_per_method": {expected["method_id"]: len(expected["seeds"])},
            "tasks_by_profile": {expected["profile"]: len(expected["seeds"])},
            "tasks_by_site": {"slurm-gpu": len(expected["seeds"])},
        }
        assert raw["cells"] == [
            {
                "protocol_id": expected["protocol_id"],
                "config": expected["config"],
                "seeds": expected["seeds"],
                "resource_profile": expected["profile"],
                "site": "slurm-gpu",
                "fidelity_status": expected["fidelity"],
                "expected_dataset_fingerprint": expected["fingerprint"],
                "expected_dataset_content_sha256": expected["content"],
            }
        ]


@pytest.mark.parametrize(("spec_name", "expected"), CASES.items())
def test_classic_wave1_templates_generate_exact_slurm_gpu_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spec_name: str,
    expected: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )
    output_dir = tmp_path / expected["campaign_id"]
    generated = generate_campaign(
        SPEC_ROOT / spec_name,
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        _allow_template_placeholders=True,
    )
    meta, tasks = load_manifest(Path(generated.manifest_path))
    render_slurm_sites(site_paths=(SITE_PATH,), campaign_dir=output_dir)

    assert generated.campaign_id == expected["campaign_id"]
    assert generated.task_count == len(expected["seeds"])
    assert meta["counts_by_method"] == {expected["method_id"]: len(expected["seeds"])}
    assert meta["counts_by_profile"] == {expected["profile"]: len(expected["seeds"])}
    assert meta["counts_by_site"] == {"slurm-gpu": len(expected["seeds"])}
    assert {task.seed for task in tasks} == set(expected["seeds"])
    assert {task.required_seed_count for task in tasks} == {len(expected["seeds"])}
    assert {task.method_id for task in tasks} == {expected["method_id"]}
    assert {task.protocol_id for task in tasks} == {expected["protocol_id"]}
    assert {task.config_path for task in tasks} == {expected["config"]}
    assert {task.resource_profile for task in tasks} == {expected["profile"]}
    assert {task.assigned_site for task in tasks} == {"slurm-gpu"}
    assert {task.fidelity_status for task in tasks} == {expected["fidelity"]}
    assert {task.expected_dataset_fingerprint for task in tasks} == {expected["fingerprint"]}
    assert {task.expected_dataset_content_sha256 for task in tasks} == {expected["content"]}
    wrapper = output_dir / "submit" / "slurm-gpu" / f"{expected['profile']}.slurm"
    assert wrapper.is_file()


def test_classic_wave1_canaries_and_productions_use_reviewed_passed_gates() -> None:
    registry = load_gate_registry(GATE_PATH)
    assert registry.status("tri_training") == "passed"
    assert registry.status("pseudo_label") == "passed"

    for expected in CASES.values():
        decision = evaluate_gate(
            registry,
            campaign_id=expected["campaign_id"],
            track="paper",
            method_id=expected["method_id"],
        )
        assert decision.allowed is True
        assert decision.blockers == ()


def test_classic_wave1_gate_evidence_freezes_full_profile_canary_invariants() -> None:
    evidence = json.loads(CANARY_EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert evidence["schema_version"] == 1
    assert evidence["release"] == {
        "git_sha": "8cbee2e53a029c39d53b0f1557b68dbbd9653e77",
        "tag": "replication-10m-easy-wave1-v5",
        "environment_manifest": ("evidence://modssc/build-manifests/8cbee2e53a02.json"),
    }
    tri = evidence["tri_training"]
    assert (tri["successes"], tri["expected"], tri["promotion"]) == (1, 1, "passed")
    assert tri["weka_jar_sha256"] == (
        "b034ab0d5b8d9edf7a81d10d9cd9fcf1e3b4e9db970980a9e1aaf94c515caaee"
    )
    assert tri["diagnostics"] == {
        "converged": True,
        "initial_ensemble_retained": True,
        "n_iter": 2,
        "prediction_rule": "soft_average",
        "pseudo_labels_selected_total": 59,
    }

    pseudo = evidence["pseudo_label"]
    assert (pseudo["successes"], pseudo["expected"], pseudo["promotion"]) == (1, 1, "passed")
    assert pseudo["diagnostics"] == {
        "epochs_completed": 601,
        "steps_per_epoch": 229,
        "parameter_updates": 137629,
        "alpha_reached_final": True,
        "confidence_threshold_applied": False,
        "final_pseudo_labels_assigned": 58_400,
    }
    assert pseudo["projected_ten_run_a100_hours"] < 100
    assert pseudo["elapsed_seconds"] < 20 * 60 * 60

    grand = evidence["grand"]
    assert (grand["successes"], grand["expected"], grand["promotion"]) == (3, 3, "passed")
    assert grand["mean_test_accuracy"] == 0.854
    assert grand["required_diagnostics_passed"] == 10
    assert grand["production_walltime_seconds"] == 180
