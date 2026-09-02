from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.orchestrators.reporting import write_seed_sweep_summary
from modssc.evaluation import AcceptanceSpec, parse_acceptance_spec
from modssc.runtime.execution import RunIdentity
from modssc.runtime.protocol import effective_config_sha256, protocol_sha256
from modssc.runtime.software import software_sha256


def _config(seed: int) -> dict[str, object]:
    return {
        "run": {"seed": seed},
        "method": {"id": "pseudo_label", "params": {}},
    }


def _versions() -> dict[str, str]:
    return {"python": "x", "modssc": "x", "numpy": "x", "git_sha": "x"}


def _run_report(
    path: Path,
    *,
    seed: int,
    accuracy: float | None,
    status: str = "success",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = _config(seed)
    versions = _versions()
    identity = RunIdentity(
        config_sha256=protocol_sha256(config),
        seed=seed,
        code_sha256=software_sha256(versions),
    )
    path.write_text(
        json.dumps(
            {
                "run": {
                    "seed": seed,
                    "name": f"seed-{seed}",
                    "run_id": identity.short_id,
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "finished_at": "2026-01-01T00:00:01+00:00",
                    "status": status,
                    "benchmark_mode": True,
                    "config_path": "card.yaml",
                    "error_code": (
                        "E_EVALUATION_NOT_EVALUABLE" if status == "not_evaluable" else None
                    ),
                },
                "hashes": {
                    "config_hash": f"{seed:064x}",
                    "effective_config_hash": effective_config_sha256(config),
                    "protocol_sha256": protocol_sha256(config),
                    "software_sha256": software_sha256(versions),
                    "execution_identity_sha256": identity.sha256,
                },
                "execution_identity": identity.to_dict(),
                "resolution": {
                    "device": {"requested": "cpu", "resolved": "cpu"},
                    "backend": {"requested": {}, "resolved": {}},
                    "dtype": {"requested": {}, "resolved": {}},
                    "normalization": {"requested": {}, "resolved": {}},
                    "splits": {"requested": ["test"], "resolved": {}},
                    "limits": {"requested": None, "resolved": None, "changes": []},
                },
                "protocol": {
                    "kind": "inductive",
                    "use_test_split": True,
                    "report_splits": ["test"],
                    "split_for_model_selection": "val",
                },
                "versions": versions,
                "config": config,
                "artifacts": {},
                "fallback_events": [],
                "metrics": (
                    None
                    if accuracy is None
                    else {
                        "test": {"accuracy": accuracy},
                        "terminal": {"test": {"accuracy": accuracy - 0.05}},
                        "reported": {
                            "test": {
                                "accuracy": accuracy - 0.1,
                                "policy": "median_last_20_checkpoints",
                                "selection_uses_test": False,
                            }
                        },
                    }
                ),
                "run_info": {"run_time_seconds": float(seed), "gpu_device": "CPU"},
                "task_info": {},
                "graph_info": None,
                "hpo": None,
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_seed_sweep_summary_is_certifiable_only_when_all_seeds_succeed(tmp_path: Path) -> None:
    reports = [
        _run_report(tmp_path / "run-1" / "run.json", seed=1, accuracy=0.8),
        _run_report(tmp_path / "run-2" / "run.json", seed=2, accuracy=1.0),
    ]

    output = write_seed_sweep_summary(
        output_dir=tmp_path / "complete",
        config_path=Path("card.yaml"),
        base_name="example",
        requested_seeds=[1, 2],
        run_json_paths=reports,
        expected_protocol_hashes={seed: protocol_sha256(_config(seed)) for seed in (1, 2)},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["sweep"]["status"] == "success"
    assert payload["sweep"]["certifiable"] is True
    assert payload["sweep"]["missing_seeds"] == []
    assert payload["metrics"]["test"]["accuracy"]["std_ddof"] == 1
    assert payload["metrics"]["test"]["accuracy"]["population_std"] == pytest.approx(0.1)
    assert payload["metrics"]["test"]["accuracy"]["ci95_low"] is not None
    assert payload["metrics"]["terminal"]["test"]["accuracy"]["mean"] == pytest.approx(0.85)
    assert payload["metrics"]["reported"]["test"]["accuracy"]["mean"] == pytest.approx(0.8)

    incomplete = write_seed_sweep_summary(
        output_dir=tmp_path / "incomplete",
        config_path=Path("card.yaml"),
        base_name="example",
        requested_seeds=[1, 2],
        run_json_paths=reports[:1],
        expected_protocol_hashes={seed: protocol_sha256(_config(seed)) for seed in (1, 2)},
    )
    incomplete_payload = json.loads(incomplete.read_text(encoding="utf-8"))

    assert incomplete_payload["sweep"]["status"] == "partial_failure"
    assert incomplete_payload["sweep"]["certifiable"] is False
    assert incomplete_payload["sweep"]["failed_run_count"] == 0
    assert incomplete_payload["sweep"]["missing_run_count"] == 1
    assert incomplete_payload["sweep"]["failed_seeds"] == []
    assert incomplete_payload["sweep"]["missing_seeds"] == [2]


def test_seed_sweep_keeps_not_evaluable_distinct_from_failure(tmp_path: Path) -> None:
    reports = [
        _run_report(
            tmp_path / f"run-{seed}" / "run.json",
            seed=seed,
            accuracy=None,
            status="not_evaluable",
        )
        for seed in (1, 2)
    ]

    output = write_seed_sweep_summary(
        output_dir=tmp_path / "aggregate",
        config_path=Path("card.yaml"),
        base_name="not-evaluable",
        requested_seeds=[1, 2],
        run_json_paths=reports,
        expected_protocol_hashes={seed: protocol_sha256(_config(seed)) for seed in (1, 2)},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["sweep"]["status"] == "not_evaluable"
    assert payload["sweep"]["certifiable"] is False
    assert payload["sweep"]["successful_run_count"] == 0
    assert payload["sweep"]["not_evaluable_run_count"] == 2
    assert payload["sweep"]["failed_run_count"] == 0
    assert payload["metrics"] == {}


def _acceptance_spec() -> AcceptanceSpec:
    return parse_acceptance_spec(
        {
            "protocol_id": "paper-table-1",
            "method_id": "pseudo_label",
            "repetitions": 2,
            "fidelity_ceiling": "paper_matched",
            "conformity": {
                "status": "passed",
                "basis": "native equation and protocol review",
                "evidence": ["tests/evaluation/test_acceptance.py"],
                "review": {
                    "reviewed_by": "test-suite",
                    "reviewed_at": "2026-08-29T00:00:00+00:00",
                },
            },
            "target": {
                "path": "metrics.test.accuracy",
                "published_mean": 0.9,
                "margin_absolute": 0.11,
            },
        }
    )


def test_seed_sweep_persists_native_acceptance_report(tmp_path: Path) -> None:
    reports = [
        _run_report(tmp_path / "run-1" / "run.json", seed=1, accuracy=0.8),
        _run_report(tmp_path / "run-2" / "run.json", seed=2, accuracy=1.0),
    ]

    output = write_seed_sweep_summary(
        output_dir=tmp_path / "accepted",
        config_path=Path("card.yaml"),
        base_name="example",
        requested_seeds=[1, 2],
        run_json_paths=reports,
        expected_protocol_hashes={seed: protocol_sha256(_config(seed)) for seed in (1, 2)},
        acceptance=_acceptance_spec(),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["sweep"]["status"] == "success"
    assert payload["acceptance"]["assessment_status"] == "passed"
    assert payload["acceptance"]["fidelity_status"] == "paper_matched"
    assert len(payload["acceptance"]["acceptance_sha256"]) == 64


def test_seed_sweep_marks_incomplete_native_acceptance_not_evaluable(tmp_path: Path) -> None:
    report = _run_report(tmp_path / "run-1" / "run.json", seed=1, accuracy=0.9)

    output = write_seed_sweep_summary(
        output_dir=tmp_path / "incomplete-acceptance",
        config_path=Path("card.yaml"),
        base_name="example",
        requested_seeds=[1, 2],
        run_json_paths=[report],
        expected_protocol_hashes={seed: protocol_sha256(_config(seed)) for seed in (1, 2)},
        acceptance=_acceptance_spec(),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["sweep"]["status"] == "partial_failure"
    assert payload["acceptance"]["assessment_status"] == "not_evaluable"
    assert "repetitions_incomplete_or_non_success" in payload["acceptance"]["reasons"]
