from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from bench.campaign.cli import main
from bench.campaign.errors import CampaignError
from bench.campaign.executor import execute_task
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from bench.campaign.paper_acceptance import _load_acceptance_cards, evaluate_paper_campaign
from bench.campaign.preflight_coverage import build_task_coverage
from bench.campaign.reconcile import materialize_reconcile_paths, reconcile_campaign
from bench.campaign.scientific_gates import ARTICLE10_METHODS
from bench.utils.io import atomic_write_json

from .helpers import (
    FakeRunner,
    fake_versions,
    minimal_config,
    preflight_governance,
    rewrite_success_digest,
    write_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE_PATH = REPO_ROOT / "bench" / "campaigns" / "article10-paper-acceptance.yaml"


def _passed_gate_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "registry_id": "modssc-scientific-gates-v2",
        "methods": {
            method_id: {
                "algorithmic_conformity": "passed",
                "conformity_basis": "pinned_official_implementation",
                "evidence": [f"evidence/{method_id}.json"],
                "reviewed_by": "reviewer",
                "reviewed_at": "2026-07-23T12:00:00Z",
            }
            for method_id in ARTICLE10_METHODS
        },
        "dependencies": {
            "flexmatch": ["fixmatch"],
            "free_match": ["fixmatch"],
            "softmatch": ["fixmatch"],
        },
        "protected_campaign_prefixes": ["article10-"],
        "exempt_campaign_ids": [
            "article10-canary-r3-wave1-v1",
            "article10-canary-r3-wave2-v1",
        ],
    }


def _build_case(
    tmp_path: Path,
    *,
    fidelity_status: str = "paper_matched",
    critical_unknowns: list[str] | None = None,
    diagnostic_values: tuple[bool, bool] = (True, True),
    second_success: bool = True,
    secondary_values: tuple[float, float] | None = None,
    require_secondary: bool = False,
    method_id: str = "pseudo_label",
    gate_payload: dict[str, Any] | None = None,
    published_mean: float = 0.5,
    protocol_conformity: str | None = None,
    published_std_ddof: int | None = None,
    environment_differences: list[str] | None = None,
    method_profile: str = "paper:test-protocol",
    documented_equivalences: list[str] | None = None,
    target_path: str | None = None,
    require_informational: bool = False,
) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    gates_path = repo / "bench" / "campaigns" / "scientific-gates.yaml"
    write_yaml(gates_path, gate_payload or _passed_gate_payload())
    config_path = repo / "paper.yaml"
    config = minimal_config(output_dir=tmp_path / "source-output", cache_dir=tmp_path / "cache")
    config["run"]["seeds"] = [1, 2]
    config["method"]["id"] = method_id
    config["method"]["profile"] = method_profile
    write_yaml(config_path, config)
    spec_path = repo / "campaign.yaml"
    write_yaml(
        spec_path,
        {
            "schema_version": 1,
            "campaign_id": "paper-test",
            "track": "paper",
            "code": {
                "git_sha": "test-sha",
                "require_clean": False,
                "git_diff_sha256": "0" * 64,
                "environment_lock_sha256": "test-env",
            },
            "scientific_scope": {
                "claim_scope_id": "article10",
                "stage": "production",
                "claim_eligible": True,
            },
            "expect": {"config_count": 1, "task_count": 2, "tasks_per_method": 2},
            "cells": [
                {
                    "protocol_id": "test-protocol",
                    "config": "paper.yaml",
                    "seeds": "from_config",
                    "resource_profile": "cpu_test",
                    "site": "local",
                    "fidelity_status": fidelity_status,
                    "expected_dataset_fingerprint": "dataset-fp",
                }
            ],
        },
    )
    campaign = tmp_path / "campaign"
    generate_campaign(spec_path, repo_root=repo, output_dir=campaign)
    meta, tasks = load_manifest(campaign / "manifest.jsonl")

    preflight_path = tmp_path / "preflight.json"
    atomic_write_json(
        preflight_path,
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "max_authorization_age_hours": 24.0,
            "status": "pass",
            "campaign_id": "paper-test",
            "manifest_sha256": meta["manifest_sha256"],
            "required_architecture": "CPU",
            "task_coverage": build_task_coverage(
                [task.task_id for task in tasks], architecture="CPU"
            ),
            **preflight_governance(tasks),
        },
    )
    result_root = tmp_path / "results"
    for index, task in enumerate(tasks):
        is_success = index == 0 or second_success
        if not is_success:
            continue
        execution = execute_task(
            campaign / "manifest.jsonl",
            repo_root=repo,
            result_root=result_root,
            work_root=tmp_path / "work",
            site_id="local",
            index=task.task_index,
            environment_lock_sha256="test-env",
            preflight_report_path=preflight_path,
            runner=FakeRunner(),
            version_collector=fake_versions,
        )
        result_dir = Path(execution.result_dir)
        run_path = result_dir / "run" / "run.json"
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        metrics: dict[str, Any] = {"test": {"accuracy": 0.49 if index == 0 else 0.51}}
        if secondary_values is not None:
            metrics["test_initial"] = {"accuracy": secondary_values[index]}
        payload["metrics"] = metrics
        payload["artifacts"]["method"]["diagnostics"] = {
            "coherent": diagnostic_values[index],
            "paper_metrics": {
                "historical_paper_metric": {
                    "test_accuracy": 0.49 if index == 0 else 0.51,
                },
                "fixed_terminal_metric": {
                    "test_accuracy": 0.10 if index == 0 else 0.20,
                },
            },
        }
        atomic_write_json(run_path, payload)
        rewrite_success_digest(result_dir)
    reconcile_report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[result_root],
        output_dir=tmp_path / "reconcile",
        emit_retry=False,
    )
    reconcile_path = Path(reconcile_report.report_path)
    acceptance_path = tmp_path / "acceptance.yaml"
    card: dict[str, Any] = {
        "method_id": method_id,
        "repetitions": 2,
        "target": {
            "transform": "identity",
            "published_mean": published_mean,
            "published_std": 0.02,
            "margin_absolute": 0.01,
        },
        "required_diagnostics": [
            {
                "path": "artifacts.method.diagnostics.coherent",
                "op": "eq",
                "value": True,
            }
        ],
        "known_deviations": [],
        "documented_equivalences": documented_equivalences or [],
        "critical_unknowns": critical_unknowns or [],
        "environment_differences": environment_differences or [],
    }
    if target_path is None:
        card["target"].update({"split": "test", "metric": "accuracy"})
    else:
        card["target"]["path"] = target_path
    if published_std_ddof is not None:
        card["target"]["published_std_ddof"] = published_std_ddof
    if protocol_conformity is not None:
        card["protocol_conformity"] = protocol_conformity
    if require_informational:
        card["informational_targets"] = [
            {
                "id": "terminal",
                "path": (
                    "artifacts.method.diagnostics.paper_metrics.fixed_terminal_metric.test_accuracy"
                ),
                "transform": "identity",
                "published_mean": 0.9,
                "margin_absolute": 0.01,
            }
        ]
    if require_secondary:
        card["secondary_targets"] = [
            {
                "id": "round0",
                "split": "test_initial",
                "metric": "accuracy",
                "transform": "one_minus",
                "published_mean": 0.5,
                "margin_absolute": 0.01,
            }
        ]
    write_yaml(
        acceptance_path,
        {
            "schema_version": 1,
            "registry_id": "test-acceptance",
            "protocols": {"test-protocol": card},
        },
    )
    return campaign, reconcile_path, acceptance_path, gates_path


def _evaluate(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    campaign, reconcile, acceptance, gates = _build_case(tmp_path, **kwargs)
    result = evaluate_paper_campaign(
        campaign / "manifest.jsonl",
        reconcile_path=reconcile,
        acceptance_path=acceptance,
        gate_registry_path=gates,
        output_dir=tmp_path / "out",
    )
    assert result.protocol_count == 1
    return json.loads(Path(result.report_path).read_text(encoding="utf-8"))["protocols"][0]


def test_paper_acceptance_requires_ci_margin_diagnostics_and_passed_gate(tmp_path: Path) -> None:
    row = _evaluate(tmp_path)

    assert row["status"] == "paper_matched"
    assert row["protocol_status"] == "paper_matched"
    assert row["result_status"] == "matched"
    assert row["equation_conformity"] == "passed"
    assert row["protocol_conformity"] == "passed"
    assert row["algorithmic_conformity"] == row["equation_conformity"]
    assert row["target_in_ci95"] is True
    assert row["within_margin"] is True
    assert row["diagnostics_ok"] is True
    assert row["absolute_difference"] == pytest.approx(0.0)
    assert row["published_std"] == pytest.approx(0.02)
    assert row["published_std_ddof"] == 1
    assert row["environment_differences"] == []
    assert row["std_absolute_difference"] == pytest.approx(abs(row["replication_std"] - 0.02))
    assert (tmp_path / "out" / "paper-acceptance.csv").is_file()


def test_paper_acceptance_uses_published_ddof_without_narrowing_ci95(tmp_path: Path) -> None:
    sample = _evaluate(tmp_path / "sample")
    population = _evaluate(tmp_path / "population", published_std_ddof=0)

    assert sample["published_std_ddof"] == 1
    assert population["published_std_ddof"] == 0
    assert sample["replication_std"] == pytest.approx(2**0.5 * 0.01)
    assert population["replication_std"] == pytest.approx(0.01)
    assert population["ci95_low"] == pytest.approx(sample["ci95_low"])
    assert population["ci95_high"] == pytest.approx(sample["ci95_high"])


def test_paper_acceptance_reads_authenticated_nested_historical_metric(tmp_path: Path) -> None:
    row = _evaluate(
        tmp_path,
        target_path=(
            "artifacts.method.diagnostics.paper_metrics.historical_paper_metric.test_accuracy"
        ),
    )

    assert row["result_status"] == "matched"
    assert row["metric"].startswith("path:artifacts.method.diagnostics.paper_metrics.")


def test_informational_target_is_reported_but_never_gates_claim(tmp_path: Path) -> None:
    row = _evaluate(tmp_path, require_informational=True)

    assert row["status"] == "paper_matched"
    assert row["informational_targets"][0]["replication_mean"] == pytest.approx(0.15)
    assert row["informational_targets"][0]["within_margin"] is False
    assert (tmp_path / "out" / "paper-informational-targets.csv").is_file()


def test_environment_difference_is_reported_without_lowering_protocol_fidelity(
    tmp_path: Path,
) -> None:
    row = _evaluate(
        tmp_path,
        environment_differences=["PyTorch and accelerator differ from the paper"],
    )

    assert row["status"] == "paper_matched"
    assert row["environment_differences"] == ["PyTorch and accelerator differ from the paper"]


def test_documented_equivalence_is_reported_without_lowering_protocol_fidelity(
    tmp_path: Path,
) -> None:
    row = _evaluate(
        tmp_path,
        documented_equivalences=[
            "The independent sampler has the same authenticated replacement law."
        ],
    )

    assert row["status"] == "paper_matched"
    assert row["documented_equivalences"] == [
        "The independent sampler has the same authenticated replacement law."
    ]


@pytest.mark.parametrize("published_std_ddof", [True, -1, 2, "0"])
def test_paper_acceptance_rejects_invalid_published_std_ddof(
    tmp_path: Path, published_std_ddof: object
) -> None:
    acceptance = tmp_path / "acceptance.yaml"
    write_yaml(
        acceptance,
        {
            "schema_version": 1,
            "protocols": {
                "test-protocol": {
                    "method_id": "pseudo_label",
                    "repetitions": 2,
                    "target": {
                        "split": "test",
                        "metric": "accuracy",
                        "published_mean": 0.5,
                        "published_std": 0.02,
                        "published_std_ddof": published_std_ddof,
                        "margin_absolute": 0.01,
                    },
                    "known_deviations": [],
                    "critical_unknowns": [],
                }
            },
        },
    )

    with pytest.raises(CampaignError, match="published_std_ddof must equal 0 or 1"):
        _load_acceptance_cards(acceptance)


def test_result_and_protocol_conformity_statuses_are_independent(tmp_path: Path) -> None:
    failed_result = _evaluate(tmp_path / "failed-result", published_mean=0.8)
    pending_protocol = _evaluate(
        tmp_path / "pending-protocol",
        protocol_conformity="pending",
    )
    failed_protocol = _evaluate(
        tmp_path / "failed-protocol",
        protocol_conformity="failed",
    )

    assert failed_result["protocol_status"] == "paper_approx"
    assert failed_result["result_status"] == "failed_margin"
    assert failed_result["equation_conformity"] == "passed"
    assert failed_result["protocol_conformity"] == "passed"
    assert "absolute_margin_exceeded" in failed_result["reasons"]

    assert pending_protocol["protocol_status"] == "paper_approx"
    assert pending_protocol["result_status"] == "matched"
    assert pending_protocol["equation_conformity"] == "passed"
    assert pending_protocol["protocol_conformity"] == "pending"
    assert "protocol_conformity=pending" in pending_protocol["reasons"]

    assert failed_protocol["protocol_status"] == "paper_approx"
    assert failed_protocol["result_status"] == "matched"
    assert failed_protocol["equation_conformity"] == "passed"
    assert failed_protocol["protocol_conformity"] == "failed"
    assert "protocol_conformity=failed" in failed_protocol["reasons"]


def test_secondary_target_is_aggregated_and_required_for_exact_claim(tmp_path: Path) -> None:
    matched = _evaluate(
        tmp_path / "matched",
        require_secondary=True,
        secondary_values=(0.49, 0.51),
    )
    missing = _evaluate(tmp_path / "missing", require_secondary=True)
    incompatible = _evaluate(
        tmp_path / "incompatible",
        require_secondary=True,
        secondary_values=(0.79, 0.81),
    )

    assert matched["status"] == "paper_matched"
    assert matched["secondary_targets_ok"] is True
    assert matched["secondary_targets"][0]["replication_mean"] == pytest.approx(0.5)
    assert (tmp_path / "matched" / "out" / "paper-secondary-targets.csv").is_file()
    assert missing["status"] == "paper_approx"
    assert "secondary_target_missing" in missing["reasons"]
    assert incompatible["status"] == "paper_approx"
    assert "secondary_target_outside_replication_ci95" in incompatible["reasons"]
    assert "secondary_target_margin_exceeded" in incompatible["reasons"]


@pytest.mark.parametrize(
    ("kwargs", "expected_status", "reason"),
    [
        ({"fidelity_status": "paper_approx"}, "paper_approx", "manifest_ceiling=paper_approx"),
        ({"critical_unknowns": ["learner unknown"]}, "not_claimable", "critical_protocol_unknowns"),
        ({"diagnostic_values": (True, False)}, "paper_approx", "secondary_diagnostics_failed"),
        ({"second_success": False}, "not_claimable", "repetitions_incomplete"),
    ],
)
def test_paper_acceptance_never_inflates_claims(
    tmp_path: Path, kwargs: dict[str, Any], expected_status: str, reason: str
) -> None:
    row = _evaluate(tmp_path, **kwargs)

    assert row["status"] == expected_status
    assert reason in row["reasons"]


def test_article10_acceptance_registry_has_all_protocols_and_preregistered_margins() -> None:
    cards = _load_acceptance_cards(ACCEPTANCE_PATH)

    assert len(cards) == 21
    assert {card["method_id"] for card in cards.values()} == set(ARTICLE10_METHODS) | {
        "co_training"
    }
    nigam = cards["nigam-ghani2000-webkb-table2"]
    assert nigam["repetitions"] == 10
    assert nigam["target"] == {
        "split": "test",
        "metric": "accuracy",
        "transform": "one_minus",
        "published_mean": 0.054,
        "margin_absolute": 0.02,
    }
    assert [target["published_mean"] for target in nigam["secondary_targets"]] == [
        0.130,
        0.033,
    ]
    nigam_rules = {
        diagnostic["path"]: {key: value for key, value in diagnostic.items() if key != "path"}
        for diagnostic in nigam["required_diagnostics"]
    }
    assert nigam_rules["artifacts.method.diagnostics.n_iter"] == {
        "op": "between",
        "value": [97, 194],
    }
    assert nigam_rules["artifacts.method.diagnostics.views_select_from_same_pre_round_pool"] == {
        "op": "eq",
        "value": True,
    }
    assert nigam_rules["artifacts.method.diagnostics.addition_policy"] == {
        "op": "eq",
        "value": "ordered_multiset_view1_then_view2",
    }
    assert nigam_rules["artifacts.method.diagnostics.overlap_policy"] == {
        "op": "eq",
        "value": "ordered_multiset_view1_then_view2",
    }
    assert nigam_rules["artifacts.method.diagnostics.pseudo_label_proposals_view1"] == {
        "op": "between",
        "value": [388, 776],
    }
    assert nigam_rules["artifacts.method.diagnostics.pseudo_labels_added_to_shared_l"] == {
        "op": "between",
        "value": [776, 1552],
    }
    assert nigam_rules["artifacts.method.diagnostics.final_labeled_size"] == {
        "op": "between",
        "value": [788, 1564],
    }
    assert nigam_rules["artifacts.method.diagnostics.overlap_count"] == {
        "op": "between",
        "value": [0, 776],
    }
    assert nigam_rules["artifacts.method.diagnostics.duplicate_multiset_additions"] == {
        "op": "between",
        "value": [0, 776],
    }
    for protocol_id in (
        "sohn-2020-cifar10-table2-250",
        "zhang-2021-cifar10-table1-250",
        "wang-2023-cifar10-table1-40",
        "chen-2023-cifar10-table2-250",
    ):
        card = cards[protocol_id]
        assert card["target"]["path"].endswith(
            "paper_metrics.historical_paper_metric.test_accuracy"
        )
        assert card["informational_targets"][0]["path"].endswith(
            "paper_metrics.fixed_terminal_metric.test_accuracy"
        )
    softmatch = cards["chen-2023-cifar10-table2-250"]
    assert softmatch["critical_unknowns"] == []
    assert softmatch["known_deviations"] == []
    assert any(
        "produced with TorchSSL" in deviation for deviation in softmatch["documented_equivalences"]
    )
    tri_secondary = cards["zhou-li-2005-wdbc-table3-j48-80pct-unlabeled"]["secondary_targets"]
    assert tri_secondary == [
        {
            "id": "round0-initial-ensemble-error",
            "split": "test_initial",
            "metric": "accuracy",
            "transform": "one_minus",
            "published_mean": 0.094,
            "margin_absolute": 0.02,
        }
    ]
    vote_tri_secondary = cards["zhou-li-2005-vote-table3-j48-80pct-unlabeled"]["secondary_targets"]
    assert vote_tri_secondary == [
        {
            "id": "round0-initial-ensemble-error",
            "split": "test_initial",
            "metric": "accuracy",
            "transform": "one_minus",
            "published_mean": 0.076,
            "margin_absolute": 0.02,
        }
    ]
    dcl_vote = cards["zhou-goldman-2004-vote-table3"]
    assert dcl_vote["target"] == {
        "split": "test",
        "metric": "accuracy",
        "transform": "identity",
        "published_mean": 0.944,
        "published_std": 0.012,
        "margin_absolute": 0.02,
    }
    assert dcl_vote["critical_unknowns"] == []
    assert {diagnostic["path"] for diagnostic in dcl_vote["required_diagnostics"]} == {
        "artifacts.method.diagnostics.converged",
        "artifacts.method.diagnostics.n_iter",
        "artifacts.method.diagnostics.changed_rounds",
        "artifacts.method.diagnostics.pseudo_labels_added_per_learner",
        "artifacts.method.diagnostics.pseudo_labels_added_total",
    }
    assert any("embedded NumPy" in item for item in dcl_vote["known_deviations"])
    assert any("normal/Wald" in item for item in dcl_vote["known_deviations"])
    assert any("equation oracle" in item for item in dcl_vote["known_deviations"])
    assert dcl_vote["protocol_conformity"] == "failed"
    assert dcl_vote["diagnostic_targets"] == [
        {
            "id": "table2-mean-iterations-including-terminal-pass",
            "path": "artifacts.method.diagnostics.n_iter",
            "published_mean": 2.2,
            "margin_absolute": 1.0,
        },
        {
            "id": "table2-naive-bayes-unlabeled-examples-assigned",
            "path": "artifacts.method.diagnostics.pseudo_labels_added_per_learner.0",
            "published_mean": 66,
            "margin_absolute": 10,
        },
        {
            "id": "table2-c45-unlabeled-examples-assigned",
            "path": "artifacts.method.diagnostics.pseudo_labels_added_per_learner.1",
            "published_mean": 40,
            "margin_absolute": 5,
        },
        {
            "id": "table2-3nn-unlabeled-examples-assigned",
            "path": "artifacts.method.diagnostics.pseudo_labels_added_per_learner.2",
            "published_mean": 40,
            "margin_absolute": 5,
        },
    ]
    assert {
        target["control_mode"]: target["published_mean"] for target in dcl_vote["control_targets"]
    } == {
        "learner_0": 0.861,
        "learner_1": 0.942,
        "learner_2": 0.902,
        "combining_only": 0.938,
    }
    assert [
        (
            candidate["estimator"],
            candidate["interval"],
            candidate["role"],
            candidate["protocol_conformity"],
            candidate["test_information_used"],
        )
        for candidate in dcl_vote["confidence_candidates"]
    ] == [
        ("training_accuracy", "wald", "v1_control", "pending", False),
        ("kfold_oof", "wald", "primary_reconstruction", "pending", False),
        ("kfold_oof", "wilson", "conditional", "pending", False),
        ("kfold_oof", "clopper_pearson", "conditional", "pending", False),
    ]
    for protocol_id, card in cards.items():
        expected_margin = (
            0.02
            if protocol_id.startswith(("zhou-li-2005", "zhou-goldman-2004", "nigam-ghani2000"))
            else 0.01
        )
        assert card["target"]["margin_absolute"] == expected_margin


def test_evaluate_paper_cli_writes_machine_readable_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    campaign, reconcile, acceptance, gates = _build_case(tmp_path)

    assert (
        main(
            [
                "evaluate-paper",
                "--manifest",
                str(campaign / "manifest.jsonl"),
                "--reconcile",
                str(reconcile),
                "--acceptance",
                str(acceptance),
                "--scientific-gates",
                str(gates),
                "--output-dir",
                str(tmp_path / "cli-out"),
            ]
        )
        == 0
    )
    assert '"paper_matched": 1' in capsys.readouterr().out
    assert (tmp_path / "cli-out" / "paper-acceptance.json").is_file()


def test_paper_acceptance_rejects_diagnostic_dev_profiles_fail_closed(tmp_path: Path) -> None:
    campaign, reconcile, acceptance, gates = _build_case(
        tmp_path,
        method_profile="paper:li-zhou-2005-wine-nn-l-confirmation-v2:diagnostic-dev",
    )

    with pytest.raises(CampaignError, match="diagnostic profile metrics") as exc_info:
        evaluate_paper_campaign(
            campaign / "manifest.jsonl",
            reconcile_path=reconcile,
            acceptance_path=acceptance,
            gate_registry_path=gates,
            output_dir=tmp_path / "out",
        )
    assert exc_info.value.code == "E_PAPER_ACCEPTANCE_DIAGNOSTIC"


def test_paper_acceptance_revalidates_results_instead_of_trusting_reconcile(tmp_path) -> None:
    campaign, reconcile, acceptance, gates = _build_case(tmp_path)
    reconcile_payload = materialize_reconcile_paths(
        reconcile,
        json.loads(reconcile.read_text(encoding="utf-8")),
    )
    run_path = Path(reconcile_payload["tasks"][0]["run_json_paths"][0])
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    run_payload["metrics"]["test"]["accuracy"] = 1.0
    atomic_write_json(run_path, run_payload)

    with pytest.raises(CampaignError, match="run.json digest differs"):
        evaluate_paper_campaign(
            campaign / "manifest.jsonl",
            reconcile_path=reconcile,
            acceptance_path=acceptance,
            gate_registry_path=gates,
            output_dir=tmp_path / "out",
        )


def test_paper_acceptance_rejects_duplicate_reconcile_rows(tmp_path) -> None:
    campaign, reconcile, acceptance, gates = _build_case(tmp_path)
    payload = materialize_reconcile_paths(
        reconcile,
        json.loads(reconcile.read_text(encoding="utf-8")),
    )
    payload["tasks"].append(dict(payload["tasks"][0]))
    duplicate_reconcile = tmp_path / "duplicate-reconcile.json"
    atomic_write_json(duplicate_reconcile, payload)

    with pytest.raises(CampaignError, match="duplicate task rows"):
        evaluate_paper_campaign(
            campaign / "manifest.jsonl",
            reconcile_path=duplicate_reconcile,
            acceptance_path=acceptance,
            gate_registry_path=gates,
            output_dir=tmp_path / "out",
        )


def test_match_stack_dependency_blocks_execution_before_paper_acceptance(tmp_path) -> None:
    gates = _passed_gate_payload()
    gates["methods"]["fixmatch"]["algorithmic_conformity"] = "pending"
    gates["protected_campaign_prefixes"] = ["paper-"]

    with pytest.raises(CampaignError, match="dependency_conformity:fixmatch=pending"):
        _build_case(
            tmp_path,
            method_id="flexmatch",
            gate_payload=gates,
        )
