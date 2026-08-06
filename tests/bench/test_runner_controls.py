from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bench import main as bench_main
from bench.context import RunContext
from bench.schema import BenchConfigError, ExperimentConfig
from modssc.sampling.result import SamplingResult
from modssc.sampling.storage import load_split


def _minimal_config() -> dict[str, Any]:
    return {
        "run": {
            "name": "runner_controls",
            "seed": 1,
            "seeds": [1, 2, 3, 4, 5],
            "seeded_sections": ["dataset", "sampling", "preprocess"],
            "output_dir": "runs",
        },
        "dataset": {"id": "toy", "options": {"seed": 1}},
        "sampling": {"seed": 1, "plan": {"split": {"kind": "holdout"}}},
        "preprocess": {
            "seed": 1,
            "fit_on": "train_labeled",
            "plan": {"output_key": "features.X", "steps": [{"id": "core.to_numpy"}]},
        },
        "method": {
            "kind": "inductive",
            "id": "pseudo_label",
            "device": {"device": "cpu", "dtype": "float32"},
            "params": {},
        },
        "evaluation": {
            "split_for_model_selection": "val",
            "report_splits": ["val", "test"],
            "metrics": ["accuracy"],
        },
    }


def test_method_profile_defaults_to_standardized() -> None:
    cfg = ExperimentConfig.from_dict(_minimal_config())

    assert cfg.method.profile == "standardized"


def test_method_profile_accepts_explicit_non_empty_value() -> None:
    raw = _minimal_config()
    raw["method"]["profile"] = "paper_2013"

    cfg = ExperimentConfig.from_dict(raw)

    assert cfg.method.profile == "paper_2013"


def test_method_profile_rejects_empty_value() -> None:
    raw = _minimal_config()
    raw["method"]["profile"] = ""

    with pytest.raises(BenchConfigError, match="profile must be a non-empty string"):
        ExperimentConfig.from_dict(raw)


def test_explicit_model_seed_overrides_component_derivation() -> None:
    raw = _minimal_config()
    raw["run"]["model_seed"] = 0
    explicit = ExperimentConfig.from_dict(raw)
    implicit_raw = _minimal_config()
    implicit = ExperimentConfig.from_dict(implicit_raw)
    ctx = RunContext.from_run_config(
        name="model_seed",
        seed=explicit.run.seed,
        run_id="run-id",
        output_dir="runs",
        config_path=None,
        fail_fast=True,
    )

    assert explicit.run.model_seed == 0
    assert bench_main._resolve_method_seed(ctx, explicit) == 0
    assert implicit.run.model_seed is None
    assert bench_main._resolve_method_seed(ctx, implicit) == ctx.seed_for("method")


def test_explicit_seed_runs_exactly_once_and_overrides_yaml_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = _minimal_config()
    calls: list[tuple[dict[str, Any], ExperimentConfig]] = []

    monkeypatch.setattr(bench_main, "load_yaml", lambda _path: raw)

    def _fake_single(
        _config_path: Path,
        *,
        raw: dict[str, Any],
        cfg: ExperimentConfig,
    ) -> bench_main.SingleRunResult:
        calls.append((raw, cfg))
        return bench_main.SingleRunResult(
            code=0,
            run_dir=tmp_path,
            run_json_path=tmp_path / "run.json",
        )

    monkeypatch.setattr(bench_main, "_run_experiment_single", _fake_single)

    assert bench_main.run_experiment(Path("config.yaml"), seed=37) == 0
    assert len(calls) == 1
    seeded_raw, seeded_cfg = calls[0]
    assert seeded_raw["run"]["name"] == "runner_controls-seed37"
    assert seeded_raw["run"]["seed"] == 37
    assert "seeds" not in seeded_raw["run"]
    assert seeded_raw["dataset"]["options"]["seed"] == 37
    assert seeded_raw["sampling"]["seed"] == 37
    assert seeded_raw["preprocess"]["seed"] == 37
    assert seeded_cfg.run.seed == 37
    assert seeded_cfg.run.seeds is None


def test_seed_and_num_runs_are_mutually_exclusive_at_api_boundary() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        bench_main.run_experiment(Path("config.yaml"), seed=2, num_runs=5)


def test_seed_and_num_runs_are_mutually_exclusive_in_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.main",
            "--config",
            "config.yaml",
            "--seed",
            "2",
            "--num-runs",
            "5",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        bench_main.main()

    assert exc_info.value.code == 2


def test_cli_forwards_single_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench.main", "--config", "config.yaml", "--seed", "17"],
    )
    monkeypatch.setattr(bench_main, "_resolve_log_level_for_run", lambda *_args: "basic")
    monkeypatch.setattr(bench_main, "configure_logging", lambda _level: None)

    def _fake_run(
        config_path: Path,
        *,
        num_runs: int | None,
        seed: int | None,
    ) -> int:
        captured.update(config_path=config_path, num_runs=num_runs, seed=seed)
        return 0

    monkeypatch.setattr(bench_main, "run_experiment", _fake_run)

    assert bench_main.main() == 0
    assert captured == {
        "config_path": Path("config.yaml"),
        "num_runs": None,
        "seed": 17,
    }


def test_sampling_partition_is_persisted_as_replayable_run_artifact(tmp_path: Path) -> None:
    ctx = RunContext.from_run_config(
        name="sampling_replay",
        seed=3,
        run_id="run-id",
        output_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
        fail_fast=True,
    )
    ctx.ensure_dirs()
    sampling = SamplingResult(
        schema_version=1,
        created_at="now",
        dataset_fingerprint="dataset-fingerprint",
        split_fingerprint="split-fingerprint",
        plan={"split": {"kind": "holdout"}},
        indices={
            "train": np.array([0, 1], dtype=np.int64),
            "val": np.array([2], dtype=np.int64),
            "test": np.array([3], dtype=np.int64),
            "train_labeled": np.array([0], dtype=np.int64),
            "train_unlabeled": np.array([1], dtype=np.int64),
        },
        refs={
            "train": "train",
            "val": "train",
            "test": "train",
            "train_labeled": "train",
            "train_unlabeled": "train",
        },
        masks={},
        stats={"train_labeled": {"n": 1}},
    )

    artifact = bench_main._persist_sampling_replay(ctx, sampling)

    assert artifact["format"] == "modssc.sampling.storage.v1"
    assert artifact["path"] == "sampling_split"
    assert artifact["manifest"] == "MANIFEST.json"
    assert len(artifact["manifest_sha256"]) == 64
    replay_dir = ctx.run_dir / artifact["path"]
    assert (replay_dir / "split.json").is_file()
    assert (replay_dir / "arrays.npz").is_file()
    manifest = json.loads((replay_dir / artifact["manifest"]).read_text(encoding="utf-8"))
    assert manifest["dataset_fingerprint"] == sampling.dataset_fingerprint
    assert manifest["split_fingerprint"] == sampling.split_fingerprint
    assert set(manifest["files"]) == {"split.json", "arrays.npz"}
    replayed = load_split(replay_dir)
    assert replayed.dataset_fingerprint == sampling.dataset_fingerprint
    assert replayed.split_fingerprint == sampling.split_fingerprint
    np.testing.assert_array_equal(replayed.indices["train"], sampling.indices["train"])
    np.testing.assert_array_equal(
        replayed.indices["train_labeled"], sampling.indices["train_labeled"]
    )


def test_single_run_passes_one_resource_measurement_to_failed_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = _minimal_config()
    raw["run"].pop("seeds")
    raw["run"]["output_dir"] = str(tmp_path)
    raw["run"]["fail_fast"] = False
    cfg = ExperimentConfig.from_dict(raw)
    measurement = bench_main.report_orch.begin_run_resource_measurement()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        bench_main.report_orch,
        "begin_run_resource_measurement",
        lambda: measurement,
    )
    monkeypatch.setattr(
        bench_main,
        "_collect_code_runtime_versions",
        lambda: {"python": "x", "modssc": "x", "numpy": "x", "git_sha": "x"},
    )
    monkeypatch.setattr(
        bench_main,
        "_benchmark_contract_preflight",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("preflight failed")),
    )

    write_run_summary = bench_main.report_orch.write_run_summary

    def _capture_summary(**kwargs: Any) -> None:
        captured.update(kwargs)
        write_run_summary(**kwargs)

    monkeypatch.setattr(bench_main.report_orch, "write_run_summary", _capture_summary)

    result = bench_main._run_experiment_single(tmp_path / "config.yaml", raw=raw, cfg=cfg)

    assert result.code == 1
    assert captured["status"] == "failed"
    assert captured["resource_measurement"] is measurement
    assert "preflight failed" in str(captured["error"])
    payload = json.loads(result.run_json_path.read_text(encoding="utf-8"))
    assert payload["run"]["status"] == "failed"
    assert payload["run_info"]["run_time_seconds"] >= 0.0
    assert payload["run_info"]["peak_ram_bytes"] > 0
