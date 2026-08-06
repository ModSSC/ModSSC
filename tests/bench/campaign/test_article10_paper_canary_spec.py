from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from bench.campaign import generate as generate_module
from bench.campaign.errors import CampaignError
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from bench.campaign.paper_acceptance import evaluate_paper_campaign
from bench.campaign.scientific_gates import evaluate_gate, load_gate_registry
from tools.hpc.slurm_renderer import render_slurm_sites

from .helpers import write_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_ROOT = REPO_ROOT / "tools" / "hpc" / "specs"
SITE_PATH = REPO_ROOT / "tools/hpc/config/profiles/slurm.example.yaml"
GATE_PATH = REPO_ROOT / "bench" / "campaigns" / "scientific-gates.yaml"

CASES: dict[str, dict[str, Any]] = {
    "wave1": {
        "filename": "article10-paper-canary-wave1.example.yaml",
        "campaign_id": "article10-paper-canary-wave1-v2",
        "methods": {
            "fixmatch": {
                "protocol_id": "sohn-2020-cifar10-table2-250",
                "config_path": (
                    "bench/configs/diagnostics/paper_canaries/fixmatch/cifar10-250-dev.yaml"
                ),
                "paper_config_path": "bench/configs/reproductions/fixmatch/cifar10-250.yaml",
                "dataset_id": "cifar10",
                "method_profile": "paper:sohn2020-cifar10-table2-250:diagnostic-dev",
                "resource_profile": "a100_dev",
                "seed": 1,
            },
            "grand": {
                "protocol_id": "feng-2020-cora-table1-planetoid",
                "config_path": "bench/configs/diagnostics/paper_canaries/grand/cora-dev.yaml",
                "paper_config_path": "bench/configs/reproductions/grand/cora.yaml",
                "dataset_id": "cora",
                "method_profile": "paper:feng2020-cora-table1:diagnostic-dev",
                "resource_profile": "v100_dev",
                "seed": 0,
            },
        },
        "profiles": {"a100_dev": 1, "v100_dev": 1},
        "wrappers": {
            "slurm-gpu/a100_dev.slurm",
            "slurm-gpu/v100_dev.slurm",
        },
    },
    "wave2": {
        "filename": "article10-paper-canary-wave2.example.yaml",
        "campaign_id": "article10-paper-canary-wave2-v1",
        "methods": {
            "flexmatch": {
                "protocol_id": "zhang-2021-cifar10-table1-250",
                "config_path": (
                    "bench/configs/diagnostics/paper_canaries/flexmatch/cifar10-250-dev.yaml"
                ),
                "paper_config_path": "bench/configs/reproductions/flexmatch/cifar10-250.yaml",
                "dataset_id": "cifar10",
                "method_profile": "paper:zhang2021-cifar10-table1-250:diagnostic-dev",
                "resource_profile": "a100_dev",
                "seed": 0,
            },
            "free_match": {
                "protocol_id": "wang-2023-cifar10-table1-40",
                "config_path": (
                    "bench/configs/diagnostics/paper_canaries/free_match/cifar10-40-dev.yaml"
                ),
                "paper_config_path": "bench/configs/reproductions/free_match/cifar10-40.yaml",
                "dataset_id": "cifar10",
                "method_profile": "paper:wang2023-cifar10-table1-40:diagnostic-dev",
                "resource_profile": "a100_dev",
                "seed": 0,
            },
            "softmatch": {
                "protocol_id": "chen-2023-cifar10-table2-250",
                "config_path": (
                    "bench/configs/diagnostics/paper_canaries/softmatch/cifar10-250-dev.yaml"
                ),
                "paper_config_path": "bench/configs/reproductions/softmatch/cifar10-250.yaml",
                "dataset_id": "cifar10",
                "method_profile": "paper:chen2023-cifar10-table2-250:diagnostic-dev",
                "resource_profile": "a100_dev",
                "seed": 0,
            },
        },
        "profiles": {"a100_dev": 3},
        "wrappers": {"slurm-gpu/a100_dev.slurm"},
    },
}


def _clean_placeholder_runtime(**_kwargs: object) -> dict[str, object]:
    return {
        "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
        "git_dirty": False,
        "git_diff_sha256": "0" * 64,
    }


def test_paper_canary_specs_are_explicit_templates_with_exact_expectations() -> None:
    for expected in CASES.values():
        path = SPEC_ROOT / expected["filename"]
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        methods = expected["methods"]

        assert path.name.endswith(".example.yaml")
        assert raw["schema_version"] == 1
        assert raw["campaign_id"] == expected["campaign_id"]
        assert raw["track"] == "paper"
        assert raw["default_site"] == "slurm-gpu"
        assert "selection" not in raw
        assert "profile_rules" not in raw
        assert raw["expect"] == {
            "config_count": len(methods),
            "task_count": len(methods),
            "tasks_per_method": {method_id: 1 for method_id in methods},
            "tasks_by_profile": expected["profiles"],
            "tasks_by_site": {"slurm-gpu": len(methods)},
        }
        assert len(raw["cells"]) == len(methods)
        assert {cell["config"] for cell in raw["cells"]} == {
            values["config_path"] for values in methods.values()
        }
        assert all(
            cell["seeds"] == [methods[Path(cell["config"]).parent.name]["seed"]]
            for cell in raw["cells"]
        )
        assert all(cell["site"] == "slurm-gpu" for cell in raw["cells"])
        assert all(cell["fidelity_status"] == "not_claimable" for cell in raw["cells"])
        grand_cells = [cell for cell in raw["cells"] if "/grand/" in str(cell["config"])]
        if grand_cells:
            assert len(grand_cells) == 1
            assert grand_cells[0]["model_seed_policy"] == "literal"

    for path in SPEC_ROOT.glob("*.yaml"):
        if "REPLACE_WITH_" in path.read_text(encoding="utf-8"):
            assert path.name.endswith(".example.yaml")


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ("invalid", "invalid model_seed_policy"),
        (None, "requires model_seed_policy=literal"),
    ],
)
def test_grand_paper_cell_requires_literal_model_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str | None,
    message: str,
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )
    raw = yaml.safe_load(
        (SPEC_ROOT / "article10-paper-canary-wave1.example.yaml").read_text(encoding="utf-8")
    )
    grand_cell = next(cell for cell in raw["cells"] if "/grand/" in cell["config"])
    if policy is None:
        grand_cell.pop("model_seed_policy")
    else:
        grand_cell["model_seed_policy"] = policy
    spec_path = tmp_path / "grand-seed-policy.yaml"
    write_yaml(spec_path, raw)

    with pytest.raises(CampaignError, match=message):
        generate_campaign(
            spec_path,
            repo_root=REPO_ROOT,
            output_dir=tmp_path / "out",
            _allow_template_placeholders=True,
        )


def test_paper_canary_manifests_have_only_the_five_requested_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )

    all_tasks = []
    for wave, expected in CASES.items():
        output_dir = tmp_path / wave
        generated = generate_campaign(
            SPEC_ROOT / expected["filename"],
            repo_root=REPO_ROOT,
            output_dir=output_dir,
            _allow_template_placeholders=True,
        )
        meta, tasks = load_manifest(Path(generated.manifest_path))
        render_slurm_sites(site_paths=(SITE_PATH,), campaign_dir=output_dir)
        methods = expected["methods"]
        all_tasks.extend(tasks)

        assert generated.campaign_id == expected["campaign_id"]
        assert generated.task_count == len(methods)
        assert len(tasks) == len(methods)
        assert len({task.config_path for task in tasks}) == len(methods)
        assert meta["campaign_id"] == expected["campaign_id"]
        assert meta["counts_by_method"] == {method_id: 1 for method_id in methods}
        assert meta["counts_by_profile"] == expected["profiles"]
        assert meta["counts_by_site"] == {"slurm-gpu": len(methods)}
        assert Counter(task.seed for task in tasks) == Counter(
            values["seed"] for values in methods.values()
        )
        assert {task.required_seed_count for task in tasks} == {1}
        assert {task.track for task in tasks} == {"paper"}
        assert {task.assigned_site for task in tasks} == {"slurm-gpu"}
        assert {task.fidelity_status for task in tasks} == {"not_claimable"}
        assert all(task.method_profile.endswith(":diagnostic-dev") for task in tasks)
        assert all(
            task.expected_dataset_fingerprint is not None
            and task.expected_dataset_fingerprint.startswith("REPLACE_WITH_")
            for task in tasks
        )
        assert all(
            task.expected_dataset_content_sha256 is not None
            and task.expected_dataset_content_sha256.startswith("REPLACE_WITH_")
            for task in tasks
        )

        tasks_by_method = {task.method_id: task for task in tasks}
        assert set(tasks_by_method) == set(methods)
        for method_id, values in methods.items():
            task = tasks_by_method[method_id]
            assert task.protocol_id == values["protocol_id"]
            assert task.config_path == values["config_path"]
            assert task.dataset_id == values["dataset_id"]
            assert task.method_profile == values["method_profile"]
            assert task.resource_profile == values["resource_profile"]
            assert task.seed == values["seed"]
        if "grand" in tasks_by_method:
            grand_task = tasks_by_method["grand"]
            assert grand_task.seed == 0
            assert grand_task.model_seed == 0

        wrappers = {
            str(path.relative_to(output_dir / "submit"))
            for path in (output_dir / "submit").glob("*/*.slurm")
        }
        assert wrappers == expected["wrappers"]
        resources = json.loads(
            (output_dir / "profiles" / "resources.json").read_text(encoding="utf-8")
        )
        assert sum(item["task_count"] for item in resources["array_indices"]) == len(tasks)
        assert all(1 <= item["task_count"] <= len(tasks) for item in resources["array_indices"])

    assert len(all_tasks) == 5
    assert {task.method_id for task in all_tasks} == {
        "fixmatch",
        "grand",
        "flexmatch",
        "free_match",
        "softmatch",
    }
    assert len({task.task_id for task in all_tasks}) == 5


def _scientific_core(raw: dict[str, Any]) -> dict[str, Any]:
    core = json.loads(
        json.dumps(
            {
                key: raw[key]
                for key in (
                    "dataset",
                    "sampling",
                    "preprocess",
                    "augmentation",
                    "method",
                    "evaluation",
                )
                if key in raw
            }
        )
    )
    for section in ("sampling", "preprocess", "augmentation"):
        if isinstance(core.get(section), dict):
            core[section].pop("seed", None)
    method = core["method"]
    method.pop("profile")
    params = method["params"]
    for key in ("max_epochs", "max_steps", "patience", "allow_short_run"):
        params.pop(key, None)
    classifier_params = method.get("model", {}).get("classifier_params", {})
    classifier_params.pop("max_steps", None)
    return core


def test_diagnostic_cards_only_truncate_seed_and_training_horizon() -> None:
    for expected in CASES.values():
        for method_id, values in expected["methods"].items():
            diagnostic_path = REPO_ROOT / values["config_path"]
            paper_path = REPO_ROOT / values["paper_config_path"]
            diagnostic = yaml.safe_load(diagnostic_path.read_text(encoding="utf-8"))
            paper = yaml.safe_load(paper_path.read_text(encoding="utf-8"))
            text = diagnostic_path.read_text(encoding="utf-8")

            assert "DIAGNOSTIC DEV CARD — NON-REPORTABLE" in text
            assert diagnostic["run"]["seeds"] == [values["seed"]]
            assert diagnostic["method"]["profile"].endswith(":diagnostic-dev")
            assert _scientific_core(diagnostic) == _scientific_core(paper)
            if method_id == "grand":
                assert diagnostic["method"]["params"]["max_epochs"] == 20
                assert diagnostic["method"]["params"]["patience"] == 10
                assert paper["method"]["params"]["max_epochs"] == 5000
            else:
                assert diagnostic["method"]["params"]["max_epochs"] == 1024
                assert diagnostic["method"]["params"]["max_steps"] == 4096
                assert diagnostic["method"]["model"]["classifier_params"]["max_steps"] == 1048576
                assert paper["method"]["params"]["max_steps"] == 1048576


def test_paper_acceptance_refuses_diagnostic_canary_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )
    for wave, expected in CASES.items():
        generated = generate_campaign(
            SPEC_ROOT / expected["filename"],
            repo_root=REPO_ROOT,
            output_dir=tmp_path / wave,
            _allow_template_placeholders=True,
        )
        with pytest.raises(CampaignError, match="E_PAPER_ACCEPTANCE_DIAGNOSTIC"):
            evaluate_paper_campaign(
                Path(generated.manifest_path),
                reconcile_path=tmp_path / "unused-reconcile.json",
                acceptance_path=REPO_ROOT
                / "bench"
                / "campaigns"
                / "article10-paper-acceptance.yaml",
                gate_registry_path=GATE_PATH,
                output_dir=tmp_path / f"{wave}-acceptance",
            )


def test_nonclaimable_paper_canaries_keep_fixmatch_dependency_active(tmp_path: Path) -> None:
    registry = load_gate_registry(GATE_PATH)
    wave1_id = CASES["wave1"]["campaign_id"]
    wave2_id = CASES["wave2"]["campaign_id"]

    assert registry.exempt_campaign_ids == frozenset()
    for method_id in CASES["wave1"]["methods"]:
        decision = evaluate_gate(
            registry,
            campaign_id=wave1_id,
            track="paper",
            method_id=method_id,
            campaign_stage="canary",
            claim_eligible=False,
        )
        assert decision.allowed is True
        assert decision.blockers == ()

    for method_id in CASES["wave2"]["methods"]:
        decision = evaluate_gate(
            registry,
            campaign_id=wave2_id,
            track="paper",
            method_id=method_id,
            campaign_stage="canary",
            claim_eligible=False,
        )
        assert decision.allowed is True
        assert decision.blockers == ()

    registry_payload = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    registry_payload["methods"]["fixmatch"]["algorithmic_conformity"] = "pending"
    pending_path = tmp_path / "scientific-gates-fixmatch-pending.yaml"
    write_yaml(pending_path, registry_payload)
    pending_registry = load_gate_registry(pending_path)
    for method_id in CASES["wave2"]["methods"]:
        decision = evaluate_gate(
            pending_registry,
            campaign_id=wave2_id,
            track="paper",
            method_id=method_id,
            campaign_stage="canary",
            claim_eligible=False,
        )
        assert decision.allowed is False
        assert decision.blockers == ("dependency_conformity:fixmatch=pending",)

    registry_payload["methods"]["fixmatch"] = {
        "algorithmic_conformity": "passed",
        "conformity_basis": "pinned_official_implementation",
        "evidence": ["canary/fixmatch-parity.json"],
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-07-24T10:00:00+02:00",
    }
    passed_path = tmp_path / "scientific-gates-fixmatch-passed.yaml"
    write_yaml(passed_path, registry_payload)
    passed_registry = load_gate_registry(passed_path)

    for method_id in CASES["wave2"]["methods"]:
        decision = evaluate_gate(
            passed_registry,
            campaign_id=wave2_id,
            track="paper",
            method_id=method_id,
            campaign_stage="canary",
            claim_eligible=False,
        )
        assert decision.allowed is True
        assert decision.blockers == ()


@pytest.mark.parametrize("case", CASES.values(), ids=CASES)
def test_paper_canary_examples_cannot_be_used_as_pinned_production_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        _clean_placeholder_runtime,
    )

    with pytest.raises(CampaignError, match="E_CAMPAIGN_TEMPLATE_PLACEHOLDER"):
        generate_campaign(
            SPEC_ROOT / case["filename"],
            repo_root=REPO_ROOT,
            output_dir=tmp_path / str(case["campaign_id"]),
        )
