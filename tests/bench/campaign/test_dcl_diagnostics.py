from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bench.campaign import cli
from bench.campaign import dcl_diagnostics as diagnostics_module
from bench.campaign.dcl_diagnostics import (
    DCLDiagnosticReport,
    _collect_results,
    _confidence_evaluation,
    _control_evaluation,
    _isolated_control_failures,
    _round_trace_failures,
)
from bench.campaign.dcl_partition_lock import (
    DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS,
    DCL_DIAGNOSTIC_CONTROL_PROTOCOLS,
    DCL_PAPER_PROTOCOL_ID,
)
from bench.campaign.errors import CampaignError
from bench.campaign.paper_acceptance import _load_acceptance_cards

REPO_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE_PATH = REPO_ROOT / "bench" / "campaigns" / "article10-paper-acceptance.yaml"


def _task(protocol_id: str, seed: int) -> Any:
    return SimpleNamespace(
        protocol_id=protocol_id,
        seed=seed,
        task_id=f"{protocol_id}-{seed:02d}",
    )


def _valid_trace_diagnostics(
    *,
    n_iter: int = 2,
    additions: tuple[int, int, int] = (66, 40, 40),
) -> dict[str, Any]:
    changed_rounds = n_iter - 1
    assert changed_rounds > 0
    learner_ids = ("gaussian_nb", "decision_tree", "knn")
    sizes = [40, 40, 40]
    trace: list[dict[str, Any]] = []
    for round_number in range(1, n_iter + 1):
        is_terminal = round_number == n_iter
        learners: list[dict[str, Any]] = []
        proposals: list[int] = []
        for learner_index, total in enumerate(additions):
            if is_terminal:
                proposal_count = 0
            else:
                base, remainder = divmod(total, changed_rounds)
                proposal_count = base + int(round_number <= remainder)
            proposals.append(proposal_count)
            size_before = sizes[learner_index]
            size_after = size_before + proposal_count
            learners.append(
                {
                    "learner_index": learner_index,
                    "classifier_id": learner_ids[learner_index],
                    "original_interval": {"lower": 0.8, "upper": 1.0},
                    "weight": 0.9,
                    "evolving_interval": {"lower": 0.75, "upper": 1.0},
                    "training_size_before": size_before,
                    "training_size_after": size_after,
                    "disagreement_count": proposal_count,
                    "proposal_count": proposal_count,
                    "error_estimate_before": 0.0,
                    "proposal_error": 0.0,
                    "error_estimate_after": 0.0,
                    "q": float(size_before),
                    "q_prime": float(size_after),
                    "accepted": proposal_count > 0,
                    "added_count": proposal_count,
                }
            )
            sizes[learner_index] = size_after
        trace.append(
            {
                "round": round_number,
                "majority_eligible_count": max(proposals),
                "learners": learners,
            }
        )
    return {
        "n_iter": n_iter,
        "changed_rounds": changed_rounds,
        "converged": True,
        "pseudo_labels_added_per_learner": list(additions),
        "pseudo_labels_added_total": sum(additions),
        "round_trace": trace,
    }


def _trace_payload(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {"artifacts": {"method": {"diagnostics": diagnostics}}}


def _isolated_control_diagnostics(control_mode: str) -> dict[str, Any]:
    return {
        "n_iter": 0,
        "changed_rounds": 0,
        "converged": True,
        "pseudo_labels_added_per_learner": [0, 0, 0],
        "pseudo_labels_added_total": 0,
        "confidence_protocol": {
            "estimator": "training_accuracy",
            "interval": "wald",
            "folds": 10,
            "seed": 0,
        },
        "control": {
            "mode": control_mode,
            "available_modes": [
                "learner_0",
                "learner_1",
                "learner_2",
                "combining_only",
            ],
            "learner_ids": ["gaussian_nb", "decision_tree", "knn"],
        },
        "round_trace": [],
    }


def _control_payload(control_mode: str, accuracy: float) -> dict[str, Any]:
    diagnostics = _isolated_control_diagnostics(control_mode)
    return {
        "metrics": {"test": {"accuracy": accuracy}},
        **_trace_payload(diagnostics),
    }


def _confidence_payload(
    *,
    estimator: str,
    interval: str,
    seed: int,
    test_leak: bool = False,
) -> dict[str, Any]:
    # Sixteen 2-iteration and four 3-iteration runs give the Table 2 mean 2.2.
    n_iter = 2 if seed <= 16 else 3
    metrics = {"train_labeled": {"accuracy": 1.0}}
    if test_leak:
        metrics["test"] = {"accuracy": 0.944}
    diagnostics = _valid_trace_diagnostics(n_iter=n_iter)
    diagnostics.update(
        {
            "confidence_protocol": {
                "estimator": estimator,
                "interval": interval,
                "folds": 10,
                "seed": 0,
            },
            "control": {"mode": "dcl"},
        }
    )
    return {
        "metrics": metrics,
        **_trace_payload(diagnostics),
    }


def _cards() -> dict[str, Any]:
    return _load_acceptance_cards(ACCEPTANCE_PATH)[DCL_PAPER_PROTOCOL_ID]


def _replace_trace_value(
    diagnostics: dict[str, Any],
    path: tuple[str | int, ...],
    value: Any,
) -> None:
    current: Any = diagnostics
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "expected_failure"),
    [
        (("round_trace", 1, "round"), 1, "round is not sequential"),
        (
            ("round_trace", 0, "learners", 0, "original_interval", "upper"),
            1.1,
            "confidence interval or weight is invalid",
        ),
        (
            ("round_trace", 0, "learners", 0, "evolving_interval", "upper"),
            0.7,
            "confidence interval or weight is invalid",
        ),
        (
            ("round_trace", 0, "learners", 0, "weight"),
            0.8,
            "weight is not the interval midpoint",
        ),
        (
            ("round_trace", 1, "learners", 0, "training_size_before"),
            107,
            "size or error continuity failed",
        ),
        (
            ("round_trace", 0, "learners", 0, "disagreement_count"),
            65,
            "proposal/error bounds failed",
        ),
        (
            ("round_trace", 0, "majority_eligible_count"),
            65,
            "proposal/error bounds failed",
        ),
        (
            ("round_trace", 0, "learners", 0, "proposal_error"),
            67.0,
            "proposal/error bounds failed",
        ),
        (
            ("round_trace", 0, "learners", 0, "q"),
            41.0,
            "q is inconsistent",
        ),
        (
            ("round_trace", 0, "learners", 0, "q_prime"),
            107.0,
            "q_prime is inconsistent",
        ),
        (
            ("round_trace", 0, "learners", 0, "accepted"),
            False,
            "accepted is inconsistent with q/q_prime",
        ),
        (
            ("round_trace", 0, "learners", 0, "added_count"),
            65,
            "added_count/size/error-after failed",
        ),
        (
            ("round_trace", 0, "learners", 0, "training_size_after"),
            107,
            "added_count/size/error-after failed",
        ),
        (
            ("round_trace", 0, "learners", 0, "error_estimate_after"),
            1.0,
            "added_count/size/error-after failed",
        ),
        (
            ("round_trace", 0, "learners", 1, "classifier_id"),
            "gaussian_nb",
            "classifier ids must be unique",
        ),
        (
            ("round_trace", 1, "learners", 0, "classifier_id"),
            "changed_classifier",
            "classifier_id changed between rounds",
        ),
        (
            ("round_trace", 1, "learners", 0, "accepted"),
            True,
            "terminal convergence pass changed a learner",
        ),
        (
            ("round_trace", 0, "learners", 0, "proposal_count"),
            True,
            "numeric or boolean field is invalid",
        ),
        (
            ("pseudo_labels_added_total",),
            147,
            "top-level trace diagnostics are inconsistent",
        ),
        (
            ("changed_rounds",),
            0,
            "changed_rounds does not match",
        ),
        (
            ("round_trace", 0, "majority_eligible_count"),
            None,
            "majority_eligible_count is invalid",
        ),
    ],
)
def test_round_trace_validator_rejects_corrupted_values(
    path: tuple[str | int, ...],
    value: Any,
    expected_failure: str,
) -> None:
    diagnostics = _valid_trace_diagnostics()
    _replace_trace_value(diagnostics, path, value)

    failures = _round_trace_failures(_trace_payload(diagnostics))

    assert any(expected_failure in failure for failure in failures)


def test_round_trace_validator_accepts_complete_two_and_three_round_traces() -> None:
    assert _round_trace_failures(_trace_payload(_valid_trace_diagnostics(n_iter=2))) == []
    assert _round_trace_failures(_trace_payload(_valid_trace_diagnostics(n_iter=3))) == []


def test_round_trace_validator_rejects_structural_corruption() -> None:
    diagnostics = _valid_trace_diagnostics()
    diagnostics["round_trace"][0]["learners"].pop()
    assert any(
        "exactly three learners" in failure
        for failure in _round_trace_failures(_trace_payload(diagnostics))
    )

    diagnostics = _valid_trace_diagnostics()
    diagnostics["round_trace"][0]["learners"][1]["learner_index"] = 0
    assert any(
        "learner indices must be ordered uniquely" in failure
        for failure in _round_trace_failures(_trace_payload(diagnostics))
    )

    diagnostics = _valid_trace_diagnostics()
    first_round = diagnostics["round_trace"][0]["learners"]
    first_round[0], first_round[1] = first_round[1], first_round[0]
    assert any(
        "learner indices must be ordered uniquely" in failure
        for failure in _round_trace_failures(_trace_payload(diagnostics))
    )

    diagnostics = _valid_trace_diagnostics()
    diagnostics["round_trace"][0]["learners"][0]["classifier_id"] = "other_nb"
    assert any(
        "classifier_id does not match the locked learner" in failure
        for failure in _round_trace_failures(_trace_payload(diagnostics))
    )

    diagnostics = _valid_trace_diagnostics()
    diagnostics["round_trace"][0]["learners"][0].pop("q")
    assert any(
        "fields differ" in failure for failure in _round_trace_failures(_trace_payload(diagnostics))
    )

    diagnostics = _valid_trace_diagnostics()
    diagnostics["round_trace"].pop()
    assert _round_trace_failures(_trace_payload(diagnostics)) == [
        "top-level trace diagnostics are inconsistent"
    ]


def test_control_evaluator_compares_all_table3_targets() -> None:
    card = _cards()
    target_by_mode = {
        target["control_mode"]: target["published_mean"] for target in card["control_targets"]
    }
    results = {
        protocol_id: [
            (_task(protocol_id, seed), _control_payload(mode, target_by_mode[mode]))
            for seed in range(1, 21)
        ]
        for protocol_id, mode in DCL_DIAGNOSTIC_CONTROL_PROTOCOLS.items()
    }

    protocols, matrix, gates = _control_evaluation(
        results,
        control_targets=card["control_targets"],
    )

    assert gates == {
        "control_integrity": {
            "status": "passed",
            "required_protocols": sorted(DCL_DIAGNOSTIC_CONTROL_PROTOCOLS),
            "failed_protocols": [],
            "incomplete_protocols": [],
        },
        "numerical_equivalence": {
            "status": "passed",
            "required_protocols": sorted(DCL_DIAGNOSTIC_CONTROL_PROTOCOLS),
            "failed_protocols": [],
            "incomplete_protocols": [],
        },
        "confidence": {"status": "not_applicable"},
        "dynamics": {"status": "not_applicable"},
    }
    assert len(protocols) == 4
    assert len(matrix) == 4
    assert all(protocol["status"] == "passed" for protocol in protocols)
    assert all(protocol["integrity_status"] == "passed" for protocol in protocols)
    assert all(protocol["numerical_equivalence_status"] == "passed" for protocol in protocols)
    assert all(protocol["paper_claim_allowed"] is False for protocol in protocols)


def test_control_evaluator_separates_integrity_from_numerical_equivalence() -> None:
    card = _cards()
    target_by_mode = {
        target["control_mode"]: target["published_mean"] for target in card["control_targets"]
    }
    results = {
        protocol_id: [
            (_task(protocol_id, seed), _control_payload(mode, target_by_mode[mode]))
            for seed in range(1, 21)
        ]
        for protocol_id, mode in DCL_DIAGNOSTIC_CONTROL_PROTOCOLS.items()
    }
    out_of_margin_protocol = next(iter(DCL_DIAGNOSTIC_CONTROL_PROTOCOLS))
    out_of_margin_mode = DCL_DIAGNOSTIC_CONTROL_PROTOCOLS[out_of_margin_protocol]
    for _task_row, payload in results[out_of_margin_protocol]:
        payload["metrics"]["test"]["accuracy"] = target_by_mode[out_of_margin_mode] - 0.03

    protocols, matrix, gates = _control_evaluation(
        results,
        control_targets=card["control_targets"],
    )

    out_of_margin = next(
        protocol for protocol in protocols if protocol["protocol_id"] == out_of_margin_protocol
    )
    assert out_of_margin["status"] == "failed"
    assert out_of_margin["integrity_status"] == "passed"
    assert out_of_margin["numerical_equivalence_status"] == "failed"
    assert out_of_margin["protocol_failures"] == []
    assert gates["control_integrity"]["status"] == "passed"
    assert gates["control_integrity"]["failed_protocols"] == []
    assert gates["numerical_equivalence"]["status"] == "failed"
    assert gates["numerical_equivalence"]["failed_protocols"] == [out_of_margin_protocol]
    matrix_row = next(row for row in matrix if row["protocol_id"] == out_of_margin_protocol)
    assert matrix_row["integrity_status"] == "passed"
    assert matrix_row["numerical_equivalence_status"] == "failed"


def test_control_evaluator_uses_reported_test_metric_not_initial_ensemble() -> None:
    card = _cards()
    target_by_mode = {
        target["control_mode"]: target["published_mean"] for target in card["control_targets"]
    }
    results = {
        protocol_id: [
            (_task(protocol_id, seed), _control_payload(mode, target_by_mode[mode]))
            for seed in range(1, 21)
        ]
        for protocol_id, mode in DCL_DIAGNOSTIC_CONTROL_PROTOCOLS.items()
    }
    for protocol_results in results.values():
        for _task_row, payload in protocol_results:
            payload["artifacts"]["method"]["diagnostics"]["initial_evaluation"] = {
                "test": {"accuracy": 0.0}
            }

    protocols, matrix, gates = _control_evaluation(
        results,
        control_targets=card["control_targets"],
    )

    assert gates["numerical_equivalence"]["status"] == "passed"
    assert all(protocol["numerical_equivalence_status"] == "passed" for protocol in protocols)
    mean_by_mode = {row["candidate"]: row["replication_mean"] for row in matrix}
    assert mean_by_mode.keys() == target_by_mode.keys()
    for mode, target in target_by_mode.items():
        assert mean_by_mode[mode] == pytest.approx(target)


def test_control_evaluator_rejects_a_nonisolated_control_trace() -> None:
    card = _cards()
    target_by_mode = {
        target["control_mode"]: target["published_mean"] for target in card["control_targets"]
    }
    results = {
        protocol_id: [
            (_task(protocol_id, seed), _control_payload(mode, target_by_mode[mode]))
            for seed in range(1, 21)
        ]
        for protocol_id, mode in DCL_DIAGNOSTIC_CONTROL_PROTOCOLS.items()
    }
    corrupted_protocol = next(iter(DCL_DIAGNOSTIC_CONTROL_PROTOCOLS))
    corrupted_payload = results[corrupted_protocol][0][1]
    corrupted_payload["artifacts"]["method"]["diagnostics"]["round_trace"] = [{"round": 1}]

    protocols, _matrix, gates = _control_evaluation(
        results,
        control_targets=card["control_targets"],
    )

    corrupted = next(
        protocol for protocol in protocols if protocol["protocol_id"] == corrupted_protocol
    )
    assert corrupted["status"] == "failed"
    assert corrupted["protocol_failures"][0]["control_diagnostic_failures"] == [
        "isolated control executed or reported a DCL update"
    ]
    assert corrupted["integrity_status"] == "failed"
    assert corrupted["numerical_equivalence_status"] == "passed"
    assert gates["control_integrity"]["status"] == "failed"
    assert corrupted_protocol in gates["control_integrity"]["failed_protocols"]
    assert gates["numerical_equivalence"]["status"] == "passed"


def test_isolated_control_validator_rejects_wrong_learner_metadata() -> None:
    payload = _control_payload("learner_0", 0.861)
    payload["artifacts"]["method"]["diagnostics"]["control"]["learner_ids"] = [
        "decision_tree",
        "gaussian_nb",
        "knn",
    ]

    assert _isolated_control_failures(payload, expected_mode="learner_0") == [
        "control learner metadata differ from the locked protocol"
    ]


def test_dcl_diagnostic_evaluator_writes_nonclaimable_structured_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _cards()
    target_by_mode = {
        target["control_mode"]: target["published_mean"] for target in card["control_targets"]
    }
    results = {
        protocol_id: [
            (_task(protocol_id, seed), _control_payload(mode, target_by_mode[mode]))
            for seed in range(1, 21)
        ]
        for protocol_id, mode in DCL_DIAGNOSTIC_CONTROL_PROTOCOLS.items()
    }
    monkeypatch.setattr(
        diagnostics_module,
        "load_manifest",
        lambda *args, **kwargs: (
            {"campaign_id": "dcl-controls", "manifest_sha256": "a" * 64},
            [object()],
        ),
    )
    monkeypatch.setattr(
        diagnostics_module,
        "_validate_manifest_identity",
        lambda tasks: ("controls", set(DCL_DIAGNOSTIC_CONTROL_PROTOCOLS)),
    )
    monkeypatch.setattr(diagnostics_module, "_load_reconcile", lambda path: {})
    monkeypatch.setattr(
        diagnostics_module,
        "_collect_results",
        lambda *args, **kwargs: results,
    )

    report = diagnostics_module.evaluate_dcl_diagnostics(
        tmp_path / "manifest.jsonl",
        reconcile_path=tmp_path / "reconcile.json",
        acceptance_path=ACCEPTANCE_PATH,
        output_dir=tmp_path / "out",
    )
    payload = json.loads(Path(report.report_path).read_text(encoding="utf-8"))

    assert report.status == "passed"
    assert report.gate_statuses == {
        "control_integrity": "passed",
        "numerical_equivalence": "passed",
        "confidence": "not_applicable",
        "dynamics": "not_applicable",
    }
    assert payload["schema_version"] == 2
    assert payload["paper_claim_allowed"] is False
    assert payload["protocol_conformity"] == card["protocol_conformity"] == "failed"
    assert payload["gates"]["control_integrity"]["status"] == "passed"
    assert payload["gates"]["numerical_equivalence"]["status"] == "passed"
    assert Path(report.matrix_path).is_file()


def test_confidence_evaluator_uses_table2_only_and_requires_nb_most() -> None:
    card = _cards()
    primary_protocols = {
        protocol_id: settings
        for protocol_id, settings in DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS.items()
        if settings in {("training_accuracy", "wald"), ("kfold_oof", "wald")}
    }
    results = {
        protocol_id: [
            (
                _task(protocol_id, seed),
                _confidence_payload(
                    estimator=estimator,
                    interval=interval,
                    seed=seed,
                ),
            )
            for seed in range(1, 21)
        ]
        for protocol_id, (estimator, interval) in primary_protocols.items()
    }

    protocols, matrix, gates = _confidence_evaluation(
        results,
        diagnostic_targets=card["diagnostic_targets"],
        confidence_candidates=card["confidence_candidates"],
    )

    assert gates["control_integrity"] == {"status": "not_applicable"}
    assert gates["numerical_equivalence"] == {"status": "not_applicable"}
    assert gates["confidence"] == {
        "status": "passed",
        "failed_protocols": [],
        "test_information_used": False,
    }
    assert gates["dynamics"]["status"] == "passed"
    assert set(gates["dynamics"]["eligible_protocols"]) == set(primary_protocols)
    assert gates["dynamics"]["selection_basis"] == "table2_trajectory_only"
    assert len(matrix) == 8
    assert all(protocol["status"] == "passed" for protocol in protocols)
    assert all(protocol["integrity_status"] == "not_applicable" for protocol in protocols)
    assert all(
        protocol["numerical_equivalence_status"] == "not_applicable" for protocol in protocols
    )
    assert all(protocol["protocol_conformity"] == "pending" for protocol in protocols)
    assert all(protocol["nb_receives_most"] is True for protocol in protocols)
    assert all(protocol["test_information_used"] is False for protocol in protocols)


def test_confidence_evaluator_rejects_a_corrupted_round_trace() -> None:
    card = _cards()
    protocol_id = next(
        protocol_id
        for protocol_id, settings in DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS.items()
        if settings == ("training_accuracy", "wald")
    )
    estimator, interval = DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS[protocol_id]
    results = {
        protocol_id: [
            (
                _task(protocol_id, seed),
                _confidence_payload(
                    estimator=estimator,
                    interval=interval,
                    seed=seed,
                ),
            )
            for seed in range(1, 21)
        ]
    }
    corrupted_payload = results[protocol_id][0][1]
    corrupted_payload["artifacts"]["method"]["diagnostics"]["round_trace"][0]["learners"][0][
        "q"
    ] = 41.0

    protocols, _matrix, gates = _confidence_evaluation(
        results,
        diagnostic_targets=card["diagnostic_targets"],
        confidence_candidates=card["confidence_candidates"],
    )

    assert protocols[0]["status"] == "failed"
    assert protocols[0]["confidence_status"] == "failed"
    assert protocols[0]["dynamics_status"] == "failed"
    assert protocols[0]["protocol_failures"][0]["round_trace_failures"]
    assert gates["confidence"]["status"] == "failed"
    assert gates["dynamics"]["status"] == "failed"
    assert gates["dynamics"]["eligible_protocols"] == []


def test_confidence_evaluator_blocks_incomplete_cells_and_test_leaks() -> None:
    card = _cards()
    protocol_id = next(
        protocol_id
        for protocol_id, settings in DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS.items()
        if settings == ("training_accuracy", "wald")
    )
    estimator, interval = DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS[protocol_id]
    incomplete = {
        protocol_id: [
            (
                _task(protocol_id, seed),
                _confidence_payload(estimator=estimator, interval=interval, seed=seed),
            )
            for seed in range(1, 20)
        ]
    }

    protocols, _matrix, gates = _confidence_evaluation(
        incomplete,
        diagnostic_targets=card["diagnostic_targets"],
        confidence_candidates=card["confidence_candidates"],
    )

    assert protocols[0]["status"] == "incomplete"
    assert gates["confidence"]["status"] == "incomplete"
    assert gates["dynamics"]["status"] == "incomplete"

    leaked = {
        protocol_id: [
            (
                _task(protocol_id, seed),
                _confidence_payload(
                    estimator=estimator,
                    interval=interval,
                    seed=seed,
                    test_leak=seed == 1,
                ),
            )
            for seed in range(1, 21)
        ]
    }
    with pytest.raises(CampaignError, match="held-out test metrics"):
        _confidence_evaluation(
            leaked,
            diagnostic_targets=card["diagnostic_targets"],
            confidence_candidates=card["confidence_candidates"],
        )


def test_dcl_diagnostic_reconcile_rejects_duplicate_rows() -> None:
    task = _task("protocol", 1)
    duplicate = {"task_id": task.task_id, "status": "failed"}

    with pytest.raises(CampaignError, match="duplicate task rows"):
        _collect_results(
            [task],
            reconcile={
                "campaign_id": "campaign",
                "manifest_sha256": "digest",
                "tasks": [duplicate, dict(duplicate)],
            },
            campaign_id="campaign",
            manifest_sha256="digest",
        )


@pytest.mark.parametrize(("status", "expected_exit"), [("passed", 0), ("blocked", 1)])
def test_evaluate_dcl_diagnostics_cli_returns_gate_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    expected_exit: int,
) -> None:
    report = DCLDiagnosticReport(
        campaign_id="dcl-test",
        diagnostic_kind="controls",
        status=status,
        gate_statuses={
            "control_integrity": "passed" if status == "passed" else "failed",
            "numerical_equivalence": "passed" if status == "passed" else "failed",
            "confidence": "not_applicable",
            "dynamics": "not_applicable",
        },
        report_path=str(tmp_path / "report.json"),
        matrix_path=str(tmp_path / "matrix.csv"),
    )
    monkeypatch.setattr(cli, "evaluate_dcl_diagnostics", lambda *args, **kwargs: report)

    exit_code = cli.main(
        [
            "evaluate-dcl-diagnostics",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--reconcile",
            str(tmp_path / "reconcile.json"),
            "--acceptance",
            str(tmp_path / "acceptance.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == expected_exit
    assert json.loads(capsys.readouterr().out)["status"] == status
