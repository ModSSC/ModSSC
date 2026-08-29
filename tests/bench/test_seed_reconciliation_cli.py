from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from bench import main as bench_main
from bench.errors import BenchRuntimeError
from bench.seed_sweep import apply_global_seed, sweep_run_name
from bench.utils.hashing import hash_any
from bench.utils.identity import effective_config_sha256, protocol_sha256
from modssc.evaluation import SeedReconciliationError
from modssc.runtime.execution import RunIdentity
from modssc.runtime.software import software_sha256


def _config(
    path: Path,
    *,
    seeds: list[int] | None = None,
    output_dir: str = "runs",
    acceptance_target: float | None = None,
) -> Path:
    raw: dict[str, Any] = {
        "run": {
            "name": "separate-seed-runs",
            "seed": 1,
            "output_dir": output_dir,
            "log_level": "basic",
        },
        "dataset": {"id": "toy"},
        "sampling": {"seed": 1, "plan": {"split": {"kind": "holdout"}}},
        "preprocess": {
            "seed": 1,
            "fit_on": "train_labeled",
            "cache": False,
            "plan": {"steps": [{"id": "core.to_numpy"}]},
        },
        "method": {
            "kind": "inductive",
            "id": "pseudo_label",
            "device": {"device": "cpu", "dtype": "float32"},
            "params": {},
        },
        "evaluation": {
            "split_for_model_selection": "val",
            "report_splits": ["test"],
            "metrics": ["accuracy"],
        },
    }
    if seeds is not None:
        raw["run"]["seeds"] = seeds
    if acceptance_target is not None:
        raw["run"].update({"benchmark_mode": True, "fail_fast": True})
        raw["acceptance"] = {
            "protocol_id": "paper-table-1",
            "method_id": "pseudo_label",
            "repetitions": len(seeds or []),
            "fidelity_ceiling": "paper_matched",
            "conformity": {
                "status": "passed",
                "basis": "native implementation review",
                "evidence": ["tests/bench/test_seed_reconciliation_cli.py"],
                "review": {
                    "reviewed_by": "test-suite",
                    "reviewed_at": "2026-08-29T00:00:00+00:00",
                },
            },
            "target": {
                "path": "metrics.test.accuracy",
                "published_mean": acceptance_target,
                "margin_absolute": 0.01,
            },
        }
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _expected_hashes(config_path: Path, *, seed: int) -> tuple[dict[str, Any], str, str]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeded = apply_global_seed(
        raw,
        seed=seed,
        run_name=sweep_run_name(raw["run"]["name"], seed=seed, index=0, total=1),
        seeded_sections=raw["run"].get("seeded_sections"),
    )
    return seeded, hash_any(seeded), protocol_sha256(seeded)


def _run_report(
    path: Path,
    *,
    seed: int,
    effective_config: dict[str, Any],
    config_hash: str,
    protocol_hash: str,
    status: str = "success",
    versions: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    report_versions = versions or {
        "python": "x",
        "modssc": "x",
        "numpy": "x",
        "git_sha": "x",
    }
    identity = RunIdentity(
        config_sha256=protocol_hash,
        seed=seed,
        code_sha256=software_sha256(report_versions),
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
                    "error_code": None,
                },
                "hashes": {
                    "config_hash": config_hash,
                    "effective_config_hash": effective_config_sha256(effective_config),
                    "protocol_sha256": protocol_hash,
                    "software_sha256": software_sha256(report_versions),
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
                "versions": report_versions,
                "config": effective_config,
                "artifacts": {},
                "fallback_events": [],
                "metrics": (
                    {"test": {"accuracy": float(seed) / 100.0}} if status == "success" else None
                ),
                "run_info": {"run_time_seconds": 1.0, "gpu_device": "CPU"},
                "task_info": {},
                "graph_info": None,
                "hpo": None,
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_reconcile_cli_discovers_reports_below_explicit_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path / "card.yaml", seeds=[11, 12])
    runs_root = tmp_path / "independent-runs"
    output_dir = tmp_path / "summary"
    config_12, config_hash_12, protocol_hash_12 = _expected_hashes(config_path, seed=12)
    _run_report(
        runs_root / "worker-b" / "run.json",
        seed=12,
        effective_config=config_12,
        config_hash=config_hash_12,
        protocol_hash=protocol_hash_12,
    )
    config_11, config_hash_11, protocol_hash_11 = _expected_hashes(config_path, seed=11)
    _run_report(
        runs_root / "worker-a" / "nested" / "run.json",
        seed=11,
        effective_config=config_11,
        config_hash=config_hash_11,
        protocol_hash=protocol_hash_11,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modssc-bench",
            "reconcile",
            "--config",
            str(config_path),
            "--runs-root",
            str(runs_root),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert bench_main.main() == 0

    payload = json.loads((output_dir / "aggregate.json").read_text(encoding="utf-8"))
    assert payload["sweep"]["status"] == "success"
    assert payload["sweep"]["categories"] == {
        "success": [11, 12],
        "failed": [],
        "not_evaluable": [],
        "missing": [],
    }
    assert [run["seed"] for run in payload["runs"]] == [11, 12]
    assert payload["metrics"]["test"]["accuracy"]["values"] == [0.11, 0.12]


@pytest.mark.parametrize(
    ("acceptance_target", "expected_code", "expected_status"),
    [
        (0.015, 0, "passed"),
        (0.9, 1, "failed"),
    ],
)
def test_reconcile_exit_code_includes_native_acceptance_status(
    tmp_path: Path,
    acceptance_target: float,
    expected_code: int,
    expected_status: str,
) -> None:
    config_path = _config(
        tmp_path / "card.yaml",
        seeds=[1, 2],
        acceptance_target=acceptance_target,
    )
    runs_root = tmp_path / "independent-runs"
    for seed in (1, 2):
        report_config, config_hash, protocol_hash = _expected_hashes(
            config_path,
            seed=seed,
        )
        _run_report(
            runs_root / f"seed-{seed}" / "run.json",
            seed=seed,
            effective_config=report_config,
            config_hash=config_hash,
            protocol_hash=protocol_hash,
        )

    assert (
        bench_main.reconcile_seed_runs(
            config_path,
            runs_root=runs_root,
            output_dir=None,
        )
        == expected_code
    )
    payload = json.loads((runs_root / "aggregate.json").read_text(encoding="utf-8"))
    assert payload["sweep"]["status"] == "success"
    assert payload["acceptance"]["assessment_status"] == expected_status


@pytest.mark.parametrize(
    ("acceptance_target", "expected_code", "expected_status"),
    [
        (0.015, 0, "passed"),
        (0.9, 1, "failed"),
    ],
)
def test_local_seed_sweep_exit_code_includes_native_acceptance_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    acceptance_target: float,
    expected_code: int,
    expected_status: str,
) -> None:
    output_root = tmp_path / "runs"
    config_path = _config(
        tmp_path / "card.yaml",
        seeds=[1, 2],
        output_dir=str(output_root),
        acceptance_target=acceptance_target,
    )

    def _successful_seed(
        _config_path: Path,
        *,
        raw: dict[str, Any],
        cfg: Any,
    ) -> bench_main.SingleRunResult:
        config_hash, protocol_hash = bench_main._expected_report_hashes(raw, cfg=cfg)
        run_json = _run_report(
            Path(cfg.run.output_dir) / cfg.run.name / "run.json",
            seed=cfg.run.seed,
            effective_config=raw,
            config_hash=config_hash,
            protocol_hash=protocol_hash,
        )
        return bench_main.SingleRunResult(
            code=0,
            run_dir=run_json.parent,
            run_json_path=run_json,
        )

    monkeypatch.setattr(bench_main, "_run_experiment_single", _successful_seed)

    assert bench_main.run_experiment(config_path) == expected_code
    aggregate_paths = list(output_root.rglob("aggregate.json"))
    assert len(aggregate_paths) == 1
    payload = json.loads(aggregate_paths[0].read_text(encoding="utf-8"))
    assert payload["sweep"]["status"] == "success"
    assert payload["acceptance"]["assessment_status"] == expected_status


def test_reconcile_defaults_output_to_root_and_reports_missing_separately(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path / "card.yaml", seeds=[1, 2])
    runs_root = tmp_path / "independent-runs"
    report_config, config_hash, protocol_hash = _expected_hashes(config_path, seed=1)
    _run_report(
        runs_root / "only-one" / "run.json",
        seed=1,
        effective_config=report_config,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
    )

    assert bench_main.reconcile_seed_runs(config_path, runs_root=runs_root, output_dir=None) == 1

    payload = json.loads((runs_root / "aggregate.json").read_text(encoding="utf-8"))
    assert payload["sweep"]["status"] == "partial_failure"
    assert payload["sweep"]["failed_run_count"] == 0
    assert payload["sweep"]["missing_run_count"] == 1
    assert payload["sweep"]["missing_seeds"] == [2]


def test_reconcile_requires_explicit_legacy_identity_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path / "card.yaml", seeds=[1])
    runs_root = tmp_path / "legacy-runs"
    report_config, config_hash, protocol_hash = _expected_hashes(config_path, seed=1)
    report_path = _run_report(
        runs_root / "run" / "run.json",
        seed=1,
        effective_config=report_config,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload.pop("execution_identity")
    payload["hashes"].pop("execution_identity_sha256")
    payload["run"]["run_id"] = "legacy-run-id"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchRuntimeError, match="required for a modern run report"):
        bench_main.reconcile_seed_runs(config_path, runs_root=runs_root)

    assert (
        bench_main.reconcile_seed_runs(
            config_path,
            runs_root=runs_root,
            require_execution_identity=False,
        )
        == 1
    )
    aggregate = json.loads((runs_root / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["sweep"]["status"] == "success"
    assert aggregate["sweep"]["execution_identity_complete"] is False
    assert aggregate["sweep"]["certifiable"] is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modssc-bench",
            "reconcile",
            "--config",
            str(config_path),
            "--runs-root",
            str(runs_root),
            "--allow-legacy-run-identity",
        ],
    )
    assert bench_main.main() == 1


def test_reconcile_rejects_ambiguous_or_invalid_inputs(tmp_path: Path) -> None:
    config_path = _config(tmp_path / "card.yaml", seeds=[1])

    with pytest.raises(ValueError, match="existing directory"):
        bench_main.reconcile_seed_runs(config_path, runs_root=tmp_path / "absent")

    no_seeds_path = _config(tmp_path / "without-seeds.yaml")
    with pytest.raises(ValueError, match="non-empty run.seeds"):
        bench_main.reconcile_seed_runs(no_seeds_path, runs_root=tmp_path)

    duplicate_root = tmp_path / "duplicates"
    report_config, expected_hash, expected_protocol_hash = _expected_hashes(config_path, seed=1)
    _run_report(
        duplicate_root / "first" / "run.json",
        seed=1,
        effective_config=report_config,
        config_hash=expected_hash,
        protocol_hash=expected_protocol_hash,
    )
    _run_report(
        duplicate_root / "second" / "run.json",
        seed=1,
        effective_config=report_config,
        config_hash=expected_hash,
        protocol_hash=expected_protocol_hash,
    )
    with pytest.raises(SeedReconciliationError, match="duplicate observed seed"):
        bench_main.reconcile_seed_runs(config_path, runs_root=duplicate_root)


def test_reconcile_rejects_report_from_a_different_effective_config(tmp_path: Path) -> None:
    config_path = _config(tmp_path / "card.yaml", seeds=[1])
    runs_root = tmp_path / "wrong-config"
    report_config, _expected_config, expected_protocol = _expected_hashes(config_path, seed=1)
    _run_report(
        runs_root / "run" / "run.json",
        seed=1,
        effective_config=report_config,
        config_hash="e" * 64,
        protocol_hash=expected_protocol,
    )

    with pytest.raises(SeedReconciliationError, match="config hash mismatch for seed 1"):
        bench_main.reconcile_seed_runs(config_path, runs_root=runs_root)


def test_reconcile_rejects_incomplete_schema_and_mixed_software(tmp_path: Path) -> None:
    config_path = _config(tmp_path / "card.yaml", seeds=[1, 2])
    incomplete_root = tmp_path / "incomplete-schema"
    report_config, config_hash, protocol_hash = _expected_hashes(config_path, seed=1)
    incomplete_path = _run_report(
        incomplete_root / "run" / "run.json",
        seed=1,
        effective_config=report_config,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
    )
    incomplete = json.loads(incomplete_path.read_text(encoding="utf-8"))
    incomplete["hashes"].pop("protocol_sha256")
    incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")

    with pytest.raises(BenchRuntimeError, match="protocol_sha256"):
        bench_main.reconcile_seed_runs(config_path, runs_root=incomplete_root)

    mixed_root = tmp_path / "mixed-software"
    for seed, numpy_version in ((1, "x"), (2, "y")):
        report_config, config_hash, protocol_hash = _expected_hashes(config_path, seed=seed)
        _run_report(
            mixed_root / f"seed-{seed}" / "run.json",
            seed=seed,
            effective_config=report_config,
            config_hash=config_hash,
            protocol_hash=protocol_hash,
            versions={
                "python": "x",
                "modssc": "x",
                "numpy": numpy_version,
                "git_sha": "x",
            },
        )

    with pytest.raises(SeedReconciliationError, match="software hash differs"):
        bench_main.reconcile_seed_runs(config_path, runs_root=mixed_root)


def test_reconcile_rejects_falsified_config_and_versions_payloads(tmp_path: Path) -> None:
    config_path = _config(tmp_path / "card.yaml", seeds=[1])
    report_config, config_hash, protocol_hash = _expected_hashes(config_path, seed=1)

    config_root = tmp_path / "falsified-config"
    config_path_report = _run_report(
        config_root / "run" / "run.json",
        seed=1,
        effective_config=report_config,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
    )
    config_payload = json.loads(config_path_report.read_text(encoding="utf-8"))
    config_payload["config"]["method"]["params"]["tampered"] = True
    config_path_report.write_text(json.dumps(config_payload), encoding="utf-8")

    with pytest.raises(SeedReconciliationError, match="effective config hash does not match"):
        bench_main.reconcile_seed_runs(config_path, runs_root=config_root)

    versions_root = tmp_path / "falsified-versions"
    versions_path = _run_report(
        versions_root / "run" / "run.json",
        seed=1,
        effective_config=report_config,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
    )
    versions_payload = json.loads(versions_path.read_text(encoding="utf-8"))
    versions_payload["versions"]["numpy"] = "tampered"
    versions_path.write_text(json.dumps(versions_payload), encoding="utf-8")

    with pytest.raises(SeedReconciliationError, match="software hash does not match"):
        bench_main.reconcile_seed_runs(config_path, runs_root=versions_root)


def test_local_fail_fast_sweep_writes_partial_aggregate_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "runs"
    config_path = _config(
        tmp_path / "card.yaml",
        seeds=[1, 2],
        output_dir=str(output_root),
    )

    def _fail_first_seed(
        _config_path: Path,
        *,
        raw: dict[str, Any],
        cfg: Any,
    ) -> Any:
        config_hash, protocol_hash = bench_main._expected_report_hashes(raw, cfg=cfg)
        _run_report(
            Path(cfg.run.output_dir) / cfg.run.name / "run.json",
            seed=cfg.run.seed,
            effective_config=raw,
            config_hash=config_hash,
            protocol_hash=protocol_hash,
            status="failed",
        )
        raise BenchRuntimeError("E_TEST_FAILURE", "stop after first seed")

    monkeypatch.setattr(bench_main, "_run_experiment_single", _fail_first_seed)

    with pytest.raises(BenchRuntimeError, match="stop after first seed"):
        bench_main.run_experiment(config_path)

    aggregate_paths = list(output_root.rglob("aggregate.json"))
    assert len(aggregate_paths) == 1
    payload = json.loads(aggregate_paths[0].read_text(encoding="utf-8"))
    assert payload["sweep"]["status"] == "failed"
    assert payload["sweep"]["failed_seeds"] == [1]
    assert payload["sweep"]["missing_seeds"] == [2]
    assert payload["sweep"]["completed_run_count"] == 1
