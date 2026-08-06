from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATH = REPO_ROOT / "provenance/article10/evidence/article10-replication-summary.json"
MARKDOWN_PATH = REPO_ROOT / "docs" / "development" / "article10-replication-summary.md"
REFERENCE_PATH = REPO_ROOT / "docs" / "reference" / "paper-methods.md"

METHOD_IDS = (
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
)
ALLOWED_STATUSES = {
    "algorithmic_conformity": ["passed", "pending"],
    "protocol_conformity": ["passed", "pending", "failed", "not_assessed"],
    "campaign": ["complete", "pending", "blocked"],
    "result": ["matched", "failed_margin", "pending", "not_evaluated"],
    "replication": ["paper_matched", "paper_approx", "not_claimable", "pending"],
}
FINAL_METHODS = {
    "pseudo_label": ("paper_approx", "failed_margin", 10),
    "tri_training": ("paper_approx", "matched", 3),
    "democratic_co_learning": ("paper_approx", "failed_margin", 20),
    "fixmatch": ("paper_matched", "matched", 5),
    "flexmatch": ("paper_matched", "matched", 3),
    "free_match": ("paper_matched", "matched", 3),
    "softmatch": ("paper_matched", "matched", 3),
    "laplace_learning": ("paper_matched", "matched", 500),
    "poisson_learning": ("paper_matched", "matched", 500),
    "grand": ("paper_matched", "matched", 100),
}
ACCEPTANCE_SHAS = {
    "pseudo_label": "6e656b42d8d24edf10c2b62fe7afbc4c72bd57f3ef12d709a6373f896d83b34e",
    "tri_training": "495cf0abafb7575db39c835dc3d162b12e93b5324767a83cca2b0261bd2999a6",
    "democratic_co_learning": ("d66677565b5968daade77bfa252c2178859aaad8390732f6acbae9364dc61dcc"),
    "fixmatch": "f80a675aa3a0463021577614c0334dde9ab3904ab7b9669e923cf0e8f8c4a9d9",
    "flexmatch": "b80e95ac13941b8805fcf629bdd833c825ff576bf8b1163496e7e326bfa26475",
    "free_match": "b80e95ac13941b8805fcf629bdd833c825ff576bf8b1163496e7e326bfa26475",
    "softmatch": "b80e95ac13941b8805fcf629bdd833c825ff576bf8b1163496e7e326bfa26475",
    "laplace_learning": "5c5a149db34c531a6a87dbe99f6e9b6152b4c87ee6d1a773cab4381f15b2673a",
    "poisson_learning": "5c5a149db34c531a6a87dbe99f6e9b6152b4c87ee6d1a773cab4381f15b2673a",
    "grand": "0c8a48a217ae4da243eac3a8ccbe117742212c9345beab615733986a2da73ddc",
}
DAILY_USAGE = {
    "grand": (
        "V100",
        1.5933824413888888,
        "9ede52ac93490b6790f432954471662654e1b575a7c9996d9bc6bd830de59bcd",
    ),
    "tri_training": (
        "V100",
        0.009924006666666667,
        "a22f86b8e5d5be665e8694a3818e72eae3e0d4d7677206d0e2b0554930d4188a",
    ),
    "pseudo_label": (
        "A100",
        0.6074879725,
        "1424a5595388d5b20e58ea3d0ce32cc30eadd9850391f1961088741b80401eb8",
    ),
}
TRI_EXTENDED100 = {
    "manifest_sha256": ("c2b335df4dd694ed00ff54ebf8c7318230ffc7fdb9b3a7979b8ff5e0e7336757"),
    "acceptance_sha256": ("e6d007ab44742c2b24b3a2707fb9f00aa8331867470ea05725d1b95007c179fe"),
    "reconciliation_sha256": ("9e3dc726dc924cb543366da60fc89b23fd5d83effe3fb7ec8b8dda9f67f7844c"),
    "analysis_sha256": ("4c1cdf1cb24c5a5b1d63b10d7eb9d986d5ff0aefa69f51e7bd6b465bebaaf5c5"),
    "daily_sha256": ("ee4348907e68e133eec473228d4bcdcab52eb1464862d1433a1df66657f58e71"),
}
DCL_CONFIDENCE_DIAGNOSTICS = {
    "release_sha": "e6e509e0840e6ee18ef55d1b5b99255798364f5a",
    "environment_lock_sha256": ("040ef191238a49230ed3b3e035ce03ee51dc949da9aa0b662c26f305289e37c7"),
    "primary_manifest_sha256": ("ad8e4787c5b9ce48282c0ed105813b12775746717acb2258038b676e51e286ad"),
    "primary_reconciliation_sha256": (
        "ad1a2e4d5a6710df1f4f712c697c60f2e61e7e8b2b9e5358882e6e9015d182e0"
    ),
    "primary_gates_sha256": ("42e6805a991ce4230d6158c7fdf20382884b53c86601d9a4437590faeb362990"),
    "attribution_sha256": ("565bd6f5c453bdfc7cb13bd8f04b4e75c3f8db45d08f3eef816ecf8e1221069e"),
    "conditional_manifest_sha256": (
        "0267792444c0280b9fdeb8106b94b4782a491bb502557e1f8c1459cfa5272e39"
    ),
    "conditional_canary_sha256": (
        "7744de15cb23d23b987fbd53bfe437a86790a141fe17287508192461a298f4a9"
    ),
    "conditional_reconciliation_sha256": (
        "1409d8daccaeb5356cbb7ad37d895c45931f4179c16db3ca57d911c0b8ebadc9"
    ),
    "conditional_gates_sha256": (
        "8d79278dfb3a3b40718d48ad24b81d5d036dd0b16a8a3b54ef844db8924c37dc"
    ),
    "resource_usage_descriptor_sha256": (
        "f4fafb0e5a2fb83a38bea7bb888eca0a8444d4fd6422a4c67441112aa437aa3d"
    ),
    "resource_usage_source_sha256": (
        "22a20d62130fae3c2e2e5fef5615b87f3190c2e4fc5decc424592067a0a0be5e"
    ),
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")

MATCH_EXPECTATIONS = {
    "fixmatch": {
        "campaign_id": "article10-match-fix-production-v3",
        "manifest_sha256": "1c0f8eae6e1f1dd44b974fb2687181b9ed576ced1fbe3440831576df8c473473",
        "replication_mean": 0.051180000000000024,
        "terminal_mean": 0.051239999999999994,
        "reconciliation_sha256": "42b0a8d3f77159badc440df81133c0b5cb47ece6deb59b1c5c6b4edf6418a9ed",
    },
    "flexmatch": {
        "campaign_id": "article10-match-adaptive-production-v2",
        "manifest_sha256": "c4bf7a13ffbc40d9fa69731ad583c03971e79576f22f6907074d43443a8e04a5",
        "replication_mean": 0.05106666666666667,
        "terminal_mean": 0.05266666666666664,
        "reconciliation_sha256": "07902f4567574b21ecf6fff780386d23cd8592dd12d47f505cdb50440d221673",
    },
    "free_match": {
        "campaign_id": "article10-match-adaptive-production-v2",
        "manifest_sha256": "c4bf7a13ffbc40d9fa69731ad583c03971e79576f22f6907074d43443a8e04a5",
        "replication_mean": 0.053766666666666664,
        "terminal_mean": 0.05623333333333332,
        "reconciliation_sha256": "07902f4567574b21ecf6fff780386d23cd8592dd12d47f505cdb50440d221673",
    },
    "softmatch": {
        "campaign_id": "article10-match-adaptive-production-v2",
        "manifest_sha256": "c4bf7a13ffbc40d9fa69731ad583c03971e79576f22f6907074d43443a8e04a5",
        "replication_mean": 0.0487,
        "terminal_mean": 0.05083333333333332,
        "reconciliation_sha256": "07902f4567574b21ecf6fff780386d23cd8592dd12d47f505cdb50440d221673",
    },
}


def _load_summary() -> dict[str, Any]:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _methods_by_id(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    methods = summary["methods"]
    return {str(method["method_id"]): method for method in methods}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_replication_summary_has_exact_scope_statuses_repetitions_and_proofs() -> None:
    summary = _load_summary()

    assert summary["schema_version"] == 1
    assert summary["draft"] is False
    assert summary["report_id"] == "article10-replication-summary-v2"
    assert summary["snapshot_date"] == "2026-08-05"
    assert summary["publication_redaction"] == {
        "execution_site_identity": "private",
        "historical_aliases_are_source_ids": False,
        "source_identities_are_bound_by_sha256": True,
        "provenance_descriptor": "provenance/article10/evidence/execution-history-bundle.json",
    }
    dcl_release = summary["execution_releases"]["dcl_confidence_diagnostics"]
    assert dcl_release == {
        "git_sha": DCL_CONFIDENCE_DIAGNOSTICS["release_sha"],
        "tag": "replication-10m-easy-wave1-v8",
        "environment_lock_sha256": DCL_CONFIDENCE_DIAGNOSTICS["environment_lock_sha256"],
        "accelerator_architecture": "V100",
        "claim_eligible": False,
    }
    assert summary["execution_releases"]["match_fix_production"] == {
        "git_sha": "c024a03fce7d93a1a1ac29fc0fd31bde3e6b780f",
        "tag": "replication-10m-match-wave1-v3",
        "environment_lock_sha256": DCL_CONFIDENCE_DIAGNOSTICS["environment_lock_sha256"],
    }
    assert summary["execution_releases"]["match_adaptive_production"] == {
        "git_sha": "f0fb2b7c3834f54dfd03b9aba72a32901f649ca2",
        "tag": "replication-10m-match-wave2-v1",
        "environment_lock_sha256": DCL_CONFIDENCE_DIAGNOSTICS["environment_lock_sha256"],
    }
    assert summary["allowed_statuses"] == ALLOWED_STATUSES
    assert summary["scope"] == {
        "method_count": 10,
        "method_ids": list(METHOD_IDS),
    }
    assert summary["track_statuses"] == {
        "paper": "passed",
        "standardized": "pending",
    }
    assert len(summary["methods"]) == 10
    assert [method["method_id"] for method in summary["methods"]] == list(METHOD_IDS)

    for method in summary["methods"]:
        assert method["coded"] is True
        assert method["algorithmic_conformity"] in ALLOWED_STATUSES["algorithmic_conformity"]
        assert method["protocol_conformity"] in ALLOWED_STATUSES["protocol_conformity"]
        assert method["campaign"]["status"] in ALLOWED_STATUSES["campaign"]
        assert method["result"]["status"] in ALLOWED_STATUSES["result"]
        assert method["replication_status"] in ALLOWED_STATUSES["replication"]

        repetitions = method["repetitions"]
        assert repetitions["cells"] > 0
        assert repetitions["required_per_cell"] > 0
        assert repetitions["required_total"] == (
            repetitions["cells"] * repetitions["required_per_cell"]
        )
        if method["campaign"]["status"] == "complete":
            assert repetitions["successful_total"] == repetitions["required_total"]
            assert SHA256_RE.fullmatch(method["campaign"]["manifest_sha256"])
        else:
            assert repetitions["successful_total"] == 0
            assert method["campaign"]["manifest_sha256"] is None

        assert method["evidence"]
        for proof in method["evidence"]:
            assert proof["kind"]
            assert proof["path"]
            assert SHA256_RE.fullmatch(proof["sha256"])
            proof_location = str(proof["path"])
            if "://" not in proof_location and not proof_location.startswith("/"):
                proof_path = REPO_ROOT / proof_location
                assert proof_path.is_file()
                assert _sha256(proof_path) == proof["sha256"]
            elif proof_location.startswith("modssc-artifact://"):
                assert proof_location.rsplit("/", maxsplit=1)[-1] == proof["sha256"]


def test_replication_summary_freezes_final_and_pending_decisions() -> None:
    methods = _methods_by_id(_load_summary())

    for method_id, (status, result_status, repetitions) in FINAL_METHODS.items():
        method = methods[method_id]
        assert method["campaign"]["status"] == "complete"
        assert method["repetitions"]["successful_total"] == repetitions
        assert method["replication_status"] == status
        assert method["result"]["status"] == result_status
        acceptance = next(
            proof for proof in method["evidence"] if proof["kind"] == "paper_acceptance"
        )
        assert acceptance["sha256"] == ACCEPTANCE_SHAS[method_id]

    assert methods["grand"]["result"]["replication_mean"] == pytest.approx(0.85366)
    assert methods["grand"]["protocol_conformity"] == "passed"
    assert methods["tri_training"]["result"]["secondary_targets_ok"] is False
    tri_extension = methods["tri_training"]["robustness_extension"]
    assert tri_extension["status"] == "complete"
    assert tri_extension["manifest_sha256"] == TRI_EXTENDED100["manifest_sha256"]
    assert tri_extension["required_total"] == 100
    assert tri_extension["successful_total"] == 100
    assert tri_extension["acceptance_result_status"] == "failed_ci95"
    assert tri_extension["equivalence_claim"] == ("replicated_within_preregistered_margin")
    assert tri_extension["equivalence_interval"] == [0.035, 0.075]
    assert tri_extension["final_test_error"]["replication_mean"] == pytest.approx(
        0.06715596330275227
    )
    assert tri_extension["final_test_error"]["within_margin"] is True
    assert tri_extension["final_test_error"]["target_in_ci95"] is False
    assert tri_extension["initial_ensemble_test_error"]["within_margin"] is True
    assert tri_extension["initial_ensemble_test_error"]["target_in_ci95"] is False
    assert tri_extension["paired_final_minus_initial_error"]["improved_count"] == 28
    assert tri_extension["paired_final_minus_initial_error"]["worsened_count"] == 36
    assert tri_extension["paired_final_minus_initial_error"]["unchanged_count"] == 36
    assert tri_extension["integrity"] == {
        "fixed_test_partition_count": 1,
        "unique_labeled_unlabeled_partitions": 100,
        "test_based_selection": False,
        "original_seed_replay_exact": True,
    }
    tri_evidence = {proof["kind"]: proof["sha256"] for proof in methods["tri_training"]["evidence"]}
    assert tri_evidence["paper_acceptance_extended100"] == (TRI_EXTENDED100["acceptance_sha256"])
    assert tri_evidence["reconciliation_extended100"] == (TRI_EXTENDED100["reconciliation_sha256"])
    assert (
        tri_evidence["paired_robustness_analysis_extended100"]
        == (TRI_EXTENDED100["analysis_sha256"])
    )
    assert methods["pseudo_label"]["result"]["within_margin"] is False
    assert methods["democratic_co_learning"]["protocol_conformity"] == "failed"
    assert methods["democratic_co_learning"]["result"]["protocol_diagnostics_ok"] is False
    dcl = methods["democratic_co_learning"]
    assert dcl["campaign"] == {
        "public_campaign_alias": "article10-paper-dcl-vote-v1",
        "source_campaign_id_sha256": (
            "61410e61b4f3d1e7434ff592b3e15e8fe00dded8bccaab61fae9d549cee9283e"
        ),
        "status": "complete",
        "manifest_sha256": "2a9f24c759442c26715c94348017eed5452f64cc53c44825b90a937bc08195f2",
    }
    diagnostic = dcl["confidence_diagnostic_extension"]
    assert diagnostic["purpose"].endswith("not a new paper replication")
    assert diagnostic["paper_claim_allowed"] is False
    assert diagnostic["selection_basis"] == "table2_trajectory_only"
    assert diagnostic["test_information_used"] is False
    assert diagnostic["primary_campaign"] == {
        "public_campaign_alias": "dcl-vote-confidence-primary-v2-v8",
        "source_campaign_id_sha256": (
            "da2b29125ec25f6690bb02a1b4bd3c9848c5666a15eba055910410395015ae67"
        ),
        "status": "complete",
        "manifest_sha256": DCL_CONFIDENCE_DIAGNOSTICS["primary_manifest_sha256"],
        "required_total": 40,
        "successful_total": 40,
        "reconciliation_sha256": DCL_CONFIDENCE_DIAGNOSTICS["primary_reconciliation_sha256"],
    }
    assert diagnostic["conditional_campaign"] == {
        "public_campaign_alias": "dcl-vote-confidence-conditional-v2-v8",
        "source_campaign_id_sha256": (
            "cc6adbf83c380e46f37f64e19ac3afbbc2c9c903d80b0eae891cb18a49f9ee69"
        ),
        "status": "complete",
        "manifest_sha256": DCL_CONFIDENCE_DIAGNOSTICS["conditional_manifest_sha256"],
        "required_total": 40,
        "successful_total": 40,
        "reconciliation_sha256": DCL_CONFIDENCE_DIAGNOSTICS["conditional_reconciliation_sha256"],
    }
    assert diagnostic["conditional_decision"]["triggered"] is True
    assert diagnostic["conditional_decision"]["status"] == "completed"
    assert diagnostic["final_diagnostic_status"] == ("tested_confidence_constructions_insufficient")
    candidates = diagnostic["primary_candidates"] + diagnostic["conditional_candidates"]
    assert len(candidates) == 4
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "resub-wald-v1-control",
        "10fold-wald-primary-reconstruction",
        "10fold-wilson-conditional",
        "10fold-clopper-pearson-conditional",
    ]
    assert all(candidate["n_success"] == 20 for candidate in candidates)
    assert all(candidate["protocol_conformity"] == "pending" for candidate in candidates)
    assert all(candidate["dynamics_status"] == "failed" for candidate in candidates)
    assert all(candidate["nb_receives_most"] is False for candidate in candidates)
    assert [candidate["mean_iterations_including_terminal_pass"] for candidate in candidates] == [
        5.55,
        5.35,
        5.1,
        4.95,
    ]
    assert [candidate["pseudo_labels_added_mean_per_learner"] for candidate in candidates] == [
        [4.45, 42.8, 5.6],
        [4.05, 40.85, 4.6],
        [3.9, 38.9, 3.65],
        [3.95, 35.2, 3.6],
    ]
    assert diagnostic["attribution"]["first_round_raw_disagreement_mean_per_learner"] == [
        3.45,
        13.4,
        2.3,
    ]
    dcl_evidence = {proof["kind"]: proof["sha256"] for proof in dcl["evidence"]}
    assert (
        dcl_evidence["confidence_primary_reconciliation"]
        == (DCL_CONFIDENCE_DIAGNOSTICS["primary_reconciliation_sha256"])
    )
    assert (
        dcl_evidence["confidence_primary_diagnostic_gates"]
        == (DCL_CONFIDENCE_DIAGNOSTICS["primary_gates_sha256"])
    )
    assert (
        dcl_evidence["confidence_full_attribution"]
        == (DCL_CONFIDENCE_DIAGNOSTICS["attribution_sha256"])
    )
    assert (
        dcl_evidence["confidence_conditional_canary_audit"]
        == (DCL_CONFIDENCE_DIAGNOSTICS["conditional_canary_sha256"])
    )
    assert (
        dcl_evidence["confidence_conditional_reconciliation"]
        == (DCL_CONFIDENCE_DIAGNOSTICS["conditional_reconciliation_sha256"])
    )
    assert (
        dcl_evidence["confidence_conditional_diagnostic_gates"]
        == (DCL_CONFIDENCE_DIAGNOSTICS["conditional_gates_sha256"])
    )
    assert (
        dcl_evidence["confidence_resource_usage_public_descriptor"]
        == (DCL_CONFIDENCE_DIAGNOSTICS["resource_usage_descriptor_sha256"])
    )

    for method_id in ("laplace_learning", "poisson_learning"):
        method = methods[method_id]
        assert method["algorithmic_conformity"] == "passed"
        assert method["protocol_conformity"] == "passed"
        assert method["campaign"]["status"] == "complete"
        assert method["campaign"]["manifest_sha256"] == (
            "26210cf77d0a1b27b12c9dce0fb1036f2ff2d54eb756d56c7c968beeec8888c8"
        )
        assert method["repetitions"] == {
            "cells": 5,
            "required_per_cell": 100,
            "required_total": 500,
            "successful_total": 500,
        }
        assert method["result"]["status"] == "matched"
        assert method["result"]["replication_mean"] is None
        assert method["result"]["std_ddof"] == 0
        assert method["result"]["all_cells_matched"] is True
        assert len(method["result"]["cells"]) == 5
        assert all(cell["n_success"] == 100 for cell in method["result"]["cells"])
        assert all(cell["target_in_ci95"] for cell in method["result"]["cells"])
        assert all(cell["within_margin"] for cell in method["result"]["cells"])
        assert all(cell["status"] == "paper_matched" for cell in method["result"]["cells"])
        assert method["replication_status"] == "paper_matched"

    for method_id in ("fixmatch", "flexmatch", "free_match", "softmatch"):
        method = methods[method_id]
        expected = MATCH_EXPECTATIONS[method_id]
        assert method["algorithmic_conformity"] == "passed"
        assert method["protocol_conformity"] == "passed"
        assert method["campaign"] == {
            "campaign_id": expected["campaign_id"],
            "status": "complete",
            "manifest_sha256": expected["manifest_sha256"],
        }
        assert method["repetitions"]["successful_total"] == method["repetitions"]["required_total"]
        assert method["result"]["status"] == "matched"
        assert method["result"]["replication_mean"] == pytest.approx(expected["replication_mean"])
        assert method["result"]["informational_terminal"]["replication_mean"] == pytest.approx(
            expected["terminal_mean"]
        )
        assert method["result"]["target_in_ci95"] is True
        assert method["result"]["within_margin"] is True
        assert method["replication_status"] == "paper_matched"
        evidence = {proof["kind"]: proof["sha256"] for proof in method["evidence"]}
        assert evidence["paper_acceptance"] == ACCEPTANCE_SHAS[method_id]
        assert evidence["reconciliation"] == expected["reconciliation_sha256"]


def test_replication_summary_freezes_daily_production_usage() -> None:
    summary = _load_summary()
    usage = summary["production_resource_usage"]

    assert usage["reserve_fraction"] == 0.15
    assert usage["reserve_status"] == "pass"
    assert usage["calder_local_cpu"] == {
        "wall_time_seconds": 11370,
        "max_processes": 2,
        "successful_tasks": 1000,
        "failed_tasks": 0,
    }
    assert set(usage["methods"]) == set(DAILY_USAGE)
    for method_id, (architecture, hours, report_sha256) in DAILY_USAGE.items():
        row = usage["methods"][method_id]
        assert row["architecture"] == architecture
        assert row["accelerator_hours"] == pytest.approx(hours)
        assert row["daily_report_path"].endswith("/daily-usage.json")
        assert row["daily_report_sha256"] == report_sha256
        assert SHA256_RE.fullmatch(row["daily_report_sha256"])

    tri_extension = usage["additional_analyses"]["tri_training_extended100"]
    assert tri_extension == {
        "architecture": "V100",
        "accelerator_hours": 0.26006854,
        "successful_tasks": 100,
        "failed_tasks": 0,
        "daily_report_path": (
            "evidence://modssc/daily/article10-paper-tri-vote-extended100-v1-002/daily-usage.json"
        ),
        "daily_report_sha256": TRI_EXTENDED100["daily_sha256"],
    }
    dcl_diagnostics = usage["additional_analyses"]["dcl_confidence_diagnostics"]
    assert dcl_diagnostics == {
        "architecture": "V100",
        "accelerator_hours": pytest.approx(2.1883333333333335),
        "scientific_tasks": 80,
        "successful_scientific_tasks": 80,
        "failed_scientific_tasks": 0,
        "allocation_count_including_postprocessing": 95,
        "budget_hours": 5.0,
        "within_budget": True,
        "usage_descriptor_path": (
            "provenance/article10/evidence/dcl-confidence-v8-resource-usage.json"
        ),
        "usage_descriptor_sha256": DCL_CONFIDENCE_DIAGNOSTICS["resource_usage_descriptor_sha256"],
        "source_usage_report_sha256": DCL_CONFIDENCE_DIAGNOSTICS["resource_usage_source_sha256"],
    }

    descriptor_path = REPO_ROOT / dcl_diagnostics["usage_descriptor_path"]
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    assert _sha256(descriptor_path) == dcl_diagnostics["usage_descriptor_sha256"]
    assert descriptor["artifact_kind"] == "redacted_historical_descriptor"
    assert descriptor["source_artifact_sha256"] == (dcl_diagnostics["source_usage_report_sha256"])
    assert descriptor["execution_site"] == {
        "scheduler": "slurm",
        "accelerator_architecture": "V100",
        "identity_visibility": "private",
    }
    assert descriptor["redacted_source_fields"] == [
        "account",
        "cluster",
        "submission_job_ids",
    ]
    assert "array_job_ids" not in descriptor["scientific_tasks"]

    bundle = summary["calder_final_bundle"]
    assert bundle["result_file_count"] == 9000
    assert bundle["artifact_manifest_sha256"] == (
        "ed70f03ce70023438466a60d9c14a21e27ffb7e5e711eb2ed04b0bfb374bf143"
    )
    assert bundle["result_files_manifest_sha256"] == (
        "47ca7c878b86bd9a1812f0ab91dbffd11e3aa9bc335343478dd0f4814431668c"
    )
    for key, value in bundle.items():
        if key.endswith("_sha256"):
            assert SHA256_RE.fullmatch(value)


def test_markdown_and_reference_statuses_match_machine_summary() -> None:
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    reference = REFERENCE_PATH.read_text(encoding="utf-8")
    overview = markdown.split("## Vue d’ensemble", maxsplit=1)[1].split(
        "## Consommation des productions", maxsplit=1
    )[0]

    display_names = (
        "Pseudo-Label",
        "Tri-Training",
        "Democratic Co-Learning",
        "FixMatch",
        "FlexMatch",
        "FreeMatch",
        "SoftMatch",
        "Laplace Learning",
        "Poisson Learning",
        "GRAND",
    )
    method_rows = [
        line
        for line in overview.splitlines()
        if any(line.startswith(f"| {display_name} |") for display_name in display_names)
    ]
    assert len(method_rows) == 10
    assert "`1.5933824413888888`" in markdown
    assert "`0.009924006666666667`" in markdown
    assert "`0.26006854`" in markdown
    assert "`0.6074879725`" in markdown
    assert "`2.1883333333333335`" in markdown
    assert "La réserve de `15 %` est respectée" in markdown
    assert "répliqué selon la marge d’équivalence préenregistrée" in markdown
    assert "résultat n’est donc pas `paper_matched`" in markdown
    assert "quatre constructions de confiance insuffisantes" in markdown
    assert "aucune ne reproduit la dynamique de la Table 2" in markdown
    assert "Vague Match terminée" in markdown
    assert "14/14 répétitions" in markdown
    assert "5,118 %" in markdown
    assert "needs a fresh confirmation" in reference
    assert "confidence diagnostics retained inverted learner dynamics" in reference

    assert "No current method is marked `paper_matched`." not in reference
    expected_reference_rows = {
        "fixmatch": "paper_matched",
        "flexmatch": "paper_matched",
        "free_match": "paper_matched",
        "softmatch": "paper_matched",
        "laplace_learning": "paper_matched",
        "poisson_learning": "paper_matched",
        "grand": "paper_matched",
    }
    for method_id, status in expected_reference_rows.items():
        assert f"| `{method_id}` |" in reference
        assert re.search(
            rf"^\| `{re.escape(method_id)}` \| [^|]+ \| `{status}` \|",
            reference,
            flags=re.MULTILINE,
        )
