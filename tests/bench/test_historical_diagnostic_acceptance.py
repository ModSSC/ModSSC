from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bench.campaign.acceptance import diagnostics as diagnostic_gate
from bench.campaign.acceptance.historical import HistoricalAcceptanceError

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_CARD = REPO_ROOT / "bench/configs/diagnostics/co_training/webkb_course_v2.yaml"
CONFIRMATION_CARD = (
    REPO_ROOT / "bench/configs/reproductions/co_training/webkb_course_table2_v2.yaml"
)


def test_v2_cards_are_locked_together() -> None:
    cards = diagnostic_gate.validate_cards(
        diagnostic_card=DIAGNOSTIC_CARD,
        confirmation_card=CONFIRMATION_CARD,
    )

    assert cards["diagnostic"]["seeds"] == [1, 2, 3, 4, 5]
    assert cards["confirmation"]["seeds"] == [6, 7, 8, 9, 10]
    assert len(cards["diagnostic"]["sha256"]) == 64
    assert len(cards["confirmation"]["sha256"]) == 64


def test_run_loader_rejects_any_test_metric_before_acceptance(tmp_path: Path) -> None:
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps(
            {
                "run": {"status": "success", "seed": 1},
                "error": None,
                "protocol": {
                    "kind": "inductive",
                    "report_splits": ["train_labeled"],
                    "split_for_model_selection": None,
                    "use_test_split": False,
                },
                "config": diagnostic_gate._diagnostic_run_contract(seed=1),
                "metrics": {
                    "train_labeled": {"accuracy": 1.0, "macro_f1": 1.0},
                    "train_labeled_fulltext": {"accuracy": 1.0, "macro_f1": 1.0},
                    "train_labeled_inlinks": {"accuracy": 1.0, "macro_f1": 1.0},
                    "test": {"accuracy": 0.0, "macro_f1": 0.0},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(HistoricalAcceptanceError, match="diagnostic metrics must contain exactly"):
        diagnostic_gate._load_diagnostic_run(run_json, expected_git_sha="a" * 40)


def test_evaluate_seals_complete_disjoint_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load(
        path: Path, *, expected_git_sha: str
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        seed = int(path.stem.removeprefix("seed"))
        assert expected_git_sha == "a" * 40
        return (
            seed,
            {
                "rounds": 30,
                "final_feature_count_view1": 2000,
                "final_feature_count_view2": 2000,
            },
            {
                "source": {"path": str(path), "sha256": f"{seed:064x}"},
                "provenance": {
                    "git_diff_sha256": "b" * 64,
                    "environment": {
                        "python": "3.12",
                        "python_implementation": "CPython",
                        "numpy": "2",
                        "scikit_learn": "1.8",
                        "modssc": "1.1",
                        "platform": "test",
                    },
                    "config_hash": f"{seed + 10:064x}",
                    "split_fingerprint": f"{seed + 20:064x}",
                    "replay_manifest_sha256": f"{seed + 30:064x}",
                    "run_contract_sha256": f"{seed + 40:064x}",
                },
            },
        )

    monkeypatch.setattr(diagnostic_gate, "_load_diagnostic_run", fake_load)
    report = diagnostic_gate.evaluate_diagnostic_runs(
        run_json_paths=[Path(f"seed{seed}.json") for seed in range(1, 6)],
        expected_git_sha="a" * 40,
        diagnostic_card=DIAGNOSTIC_CARD,
        confirmation_card=CONFIRMATION_CARD,
    )

    seal = report["sealed_provenance"]
    assert report["status"] == "passed"
    assert report["scientific_scope"]["strict_epistemic_blind"] is False
    assert seal["gate"]["test_metrics_present"] is False
    assert seal["gate"]["confirmation_authorized"] is True
    assert seal["confirmation_card"]["seeds"] == [6, 7, 8, 9, 10]
    assert len(seal["seal_sha256"]) == 64


def test_immutable_report_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "sealed.json"
    diagnostic_gate.write_immutable_report(report={"status": "first"}, output_json=output)

    with pytest.raises(HistoricalAcceptanceError, match="refusing to overwrite"):
        diagnostic_gate.write_immutable_report(report={"status": "second"}, output_json=output)

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}
