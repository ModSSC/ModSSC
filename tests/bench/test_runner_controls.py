from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bench import main as bench_main
from bench.context import RunContext
from bench.errors import BenchRuntimeError
from bench.execution_contracts import execution_contract_payload_sha256
from bench.schema import BenchConfigError, ExperimentConfig
from modssc.runtime import MethodExecutionOutcome, MethodNotEvaluableError
from modssc.runtime.contracts import (
    ContractIssue,
    ExecutionContractError,
    ExecutionContractReport,
)
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
    assert cfg.evaluation.test_selection_policy == "forbid"


def test_hpo_not_evaluable_message_preserves_the_native_reason() -> None:
    assert (
        bench_main._hpo_not_evaluable_message(
            {"status": "not_evaluable", "reason": "all_trials_not_evaluable"}
        )
        == "HPO produced no evaluable trial: all_trials_not_evaluable"
    )
    assert (
        bench_main._hpo_not_evaluable_message({"status": "not_evaluable"})
        == "HPO produced no evaluable trial"
    )


def test_benchmark_contract_forbids_test_for_model_selection() -> None:
    raw = _minimal_config()
    raw["run"]["benchmark_mode"] = True
    raw["dataset"]["download"] = False
    raw["dataset"]["cache_dir"] = "/tmp/modssc-datasets"
    raw["evaluation"]["split_for_model_selection"] = "test"
    cfg = ExperimentConfig.from_dict(raw)

    with pytest.raises(BenchConfigError) as error:
        bench_main._benchmark_contract_preflight(
            cfg=cfg,
            raw=raw,
            preprocess_steps=["core.to_numpy"],
            view_preprocess_steps=[],
        )

    assert error.value.code == "E_BENCH_TEST_SELECTION_FORBIDDEN"


def test_benchmark_contract_allows_declared_paper_protocol_test_selection() -> None:
    raw = _minimal_config()
    raw["run"].update({"benchmark_mode": True, "fail_fast": True})
    raw["dataset"].update({"download": False, "cache_dir": "/tmp/modssc-datasets"})
    raw["evaluation"].update(
        {
            "split_for_model_selection": "test",
            "test_selection_policy": "paper_protocol",
            "during_fit_splits": ["test"],
            "report_splits": ["test"],
        }
    )
    cfg = ExperimentConfig.from_dict(raw)

    bench_main._benchmark_contract_preflight(
        cfg=cfg,
        raw=raw,
        preprocess_steps=["core.to_numpy"],
        view_preprocess_steps=[],
    )


def test_benchmark_contract_rejects_unused_paper_protocol_test_selection_policy() -> None:
    raw = _minimal_config()
    raw["run"].update({"benchmark_mode": True, "fail_fast": True})
    raw["dataset"].update({"download": False, "cache_dir": "/tmp/modssc-datasets"})
    raw["evaluation"]["test_selection_policy"] = "paper_protocol"
    cfg = ExperimentConfig.from_dict(raw)

    with pytest.raises(BenchConfigError) as error:
        bench_main._benchmark_contract_preflight(
            cfg=cfg,
            raw=raw,
            preprocess_steps=["core.to_numpy"],
            view_preprocess_steps=[],
        )

    assert error.value.code == "E_BENCH_TEST_SELECTION_POLICY_INVALID"


def test_evaluation_rejects_unknown_test_selection_policy() -> None:
    raw = _minimal_config()
    raw["evaluation"]["test_selection_policy"] = "allow"

    with pytest.raises(BenchConfigError, match="test_selection_policy"):
        ExperimentConfig.from_dict(raw)


def test_benchmark_contract_accepts_explicit_null_for_fixed_terminal_protocol() -> None:
    raw = _minimal_config()
    raw["run"].update({"benchmark_mode": True, "fail_fast": True})
    raw["dataset"].update({"download": False, "cache_dir": "/tmp/modssc-datasets"})
    raw["evaluation"].update(
        {
            "split_for_model_selection": None,
            "during_fit_splits": [],
            "report_splits": ["test"],
        }
    )
    cfg = ExperimentConfig.from_dict(raw)

    bench_main._benchmark_contract_preflight(
        cfg=cfg,
        raw=raw,
        preprocess_steps=["core.to_numpy"],
        view_preprocess_steps=[],
    )


def test_benchmark_contract_requires_explicit_model_selection_declaration() -> None:
    raw = _minimal_config()
    raw["run"].update({"benchmark_mode": True, "fail_fast": True})
    raw["dataset"].update({"download": False, "cache_dir": "/tmp/modssc-datasets"})
    del raw["evaluation"]["split_for_model_selection"]
    cfg = ExperimentConfig.from_dict(raw)

    with pytest.raises(BenchConfigError) as error:
        bench_main._benchmark_contract_preflight(
            cfg=cfg,
            raw=raw,
            preprocess_steps=["core.to_numpy"],
            view_preprocess_steps=[],
        )

    assert error.value.code == "E_BENCH_SPLIT_MODEL_SELECTION_REQUIRED"


def test_benchmark_contract_rejects_null_when_evaluating_during_fit() -> None:
    raw = _minimal_config()
    raw["run"].update({"benchmark_mode": True, "fail_fast": True})
    raw["dataset"].update({"download": False, "cache_dir": "/tmp/modssc-datasets"})
    raw["evaluation"].update(
        {
            "split_for_model_selection": None,
            "during_fit_splits": ["test"],
            "report_splits": ["test"],
        }
    )
    cfg = ExperimentConfig.from_dict(raw)

    with pytest.raises(BenchConfigError) as error:
        bench_main._benchmark_contract_preflight(
            cfg=cfg,
            raw=raw,
            preprocess_steps=["core.to_numpy"],
            view_preprocess_steps=[],
        )

    assert error.value.code == "E_BENCH_SPLIT_MODEL_SELECTION_REQUIRED"


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


def test_run_execution_controls_have_safe_defaults() -> None:
    cfg = ExperimentConfig.from_dict(_minimal_config())

    assert cfg.run.resume_policy == "never"
    assert cfg.run.checkpoint_dir is None
    assert cfg.run.software_dependencies == []


def test_run_execution_controls_accept_explicit_native_checkpointing() -> None:
    raw = _minimal_config()
    raw["run"]["resume_policy"] = "auto"
    raw["run"]["checkpoint_dir"] = "/tmp/modssc-checkpoints"

    cfg = ExperimentConfig.from_dict(raw)

    assert cfg.run.resume_policy == "auto"
    assert cfg.run.checkpoint_dir == "/tmp/modssc-checkpoints"


def test_resolved_device_is_runtime_output_not_yaml_input() -> None:
    raw = _minimal_config()
    raw["method"]["device"]["resolved_device"] = "cpu"

    with pytest.raises(BenchConfigError, match="Unknown keys in method.device"):
        ExperimentConfig.from_dict(raw)


def test_run_execution_controls_accept_selective_software_dependencies() -> None:
    raw = _minimal_config()
    raw["run"]["software_dependencies"] = ["TorchVision", "faiss_cpu"]

    cfg = ExperimentConfig.from_dict(raw)

    assert cfg.run.software_dependencies == ["faiss-cpu", "torchvision"]


@pytest.mark.parametrize(
    "dependencies",
    [["TorchVision", "torchvision"], [""], [7]],
)
def test_run_execution_controls_reject_invalid_software_dependencies(
    dependencies: list[object],
) -> None:
    raw = _minimal_config()
    raw["run"]["software_dependencies"] = dependencies

    with pytest.raises(BenchConfigError, match="software_dependencies"):
        ExperimentConfig.from_dict(raw)


@pytest.mark.parametrize("resume_policy", ["", "always", "AUTO", None])
def test_run_execution_controls_reject_unknown_resume_policy(
    resume_policy: str | None,
) -> None:
    raw = _minimal_config()
    raw["run"]["resume_policy"] = resume_policy

    with pytest.raises(BenchConfigError, match="run.resume_policy"):
        ExperimentConfig.from_dict(raw)


def test_run_execution_controls_reject_empty_checkpoint_dir() -> None:
    raw = _minimal_config()
    raw["run"]["checkpoint_dir"] = ""

    with pytest.raises(BenchConfigError, match="checkpoint_dir"):
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


def test_seed_index_runs_exactly_one_declared_yaml_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = _minimal_config()
    calls: list[ExperimentConfig] = []
    monkeypatch.setattr(bench_main, "load_yaml", lambda _path: raw)

    def _fake_single(
        _config_path: Path,
        *,
        raw: dict[str, Any],
        cfg: ExperimentConfig,
    ) -> bench_main.SingleRunResult:
        del raw
        calls.append(cfg)
        return bench_main.SingleRunResult(
            code=0,
            run_dir=tmp_path,
            run_json_path=tmp_path / "run.json",
        )

    monkeypatch.setattr(bench_main, "_run_experiment_single", _fake_single)

    assert bench_main.run_experiment(Path("config.yaml"), seed_index=2) == 0
    assert len(calls) == 1
    assert calls[0].run.seed == 3
    assert calls[0].run.seeds is None


def test_seed_index_requires_declared_seed_and_valid_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _minimal_config()
    raw["run"].pop("seeds")
    monkeypatch.setattr(bench_main, "load_yaml", lambda _path: raw)

    with pytest.raises(ValueError, match="requires a non-empty run.seeds"):
        bench_main.run_experiment(Path("config.yaml"), seed_index=0)

    raw["run"]["seeds"] = [7]
    with pytest.raises(ValueError, match="outside run.seeds"):
        bench_main.run_experiment(Path("config.yaml"), seed_index=1)


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
        seed_index: int | None,
    ) -> int:
        captured.update(
            config_path=config_path,
            num_runs=num_runs,
            seed=seed,
            seed_index=seed_index,
        )
        return 0

    monkeypatch.setattr(bench_main, "run_experiment", _fake_run)

    assert bench_main.main() == 0
    assert captured == {
        "config_path": Path("config.yaml"),
        "num_runs": None,
        "seed": 17,
        "seed_index": None,
    }


def test_cli_forwards_seed_index(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench.main", "--config", "config.yaml", "--seed-index", "4"],
    )
    monkeypatch.setattr(bench_main, "_resolve_log_level_for_run", lambda *_args: "basic")
    monkeypatch.setattr(bench_main, "configure_logging", lambda _level: None)

    def _fake_run(config_path: Path, **kwargs: Any) -> int:
        captured.update(config_path=config_path, **kwargs)
        return 0

    monkeypatch.setattr(bench_main, "run_experiment", _fake_run)

    assert bench_main.main() == 0
    assert captured == {
        "config_path": Path("config.yaml"),
        "num_runs": None,
        "seed": None,
        "seed_index": 4,
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
        lambda **_kwargs: {
            "python": "x",
            "modssc": "x",
            "numpy": "x",
            "git_sha": "x",
        },
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


def test_method_not_evaluable_error_preserves_diagnostics_in_run_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = _minimal_config()
    raw["run"].pop("seeds")
    raw["run"]["output_dir"] = str(tmp_path)
    raw["run"]["fail_fast"] = False
    cfg = ExperimentConfig.from_dict(raw)
    error = MethodNotEvaluableError(
        MethodExecutionOutcome(
            status="not_evaluable",
            reason="declared convergence was not reached",
            diagnostics={"converged": False, "iterations": 3},
        )
    )

    monkeypatch.setattr(
        bench_main,
        "_collect_code_runtime_versions",
        lambda **_kwargs: {
            "python": "x",
            "modssc": "x",
            "numpy": "x",
            "git_sha": "x",
        },
    )
    monkeypatch.setattr(
        bench_main,
        "_benchmark_contract_preflight",
        lambda **_kwargs: (_ for _ in ()).throw(error),
    )

    result = bench_main._run_experiment_single(tmp_path / "config.yaml", raw=raw, cfg=cfg)
    payload = json.loads(result.run_json_path.read_text(encoding="utf-8"))

    assert result.code == 1
    assert payload["run"]["status"] == "not_evaluable"
    assert payload["run"]["error_code"] == "E_METHOD_NOT_EVALUABLE"
    assert payload["artifacts"]["method"]["diagnostics"] == {
        "converged": False,
        "iterations": 3,
    }


def test_execution_contract_error_preserves_report_and_sha_in_run_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = _minimal_config()
    raw["run"].pop("seeds")
    raw["run"]["output_dir"] = str(tmp_path)
    raw["run"]["fail_fast"] = False
    cfg = ExperimentConfig.from_dict(raw)
    report = ExecutionContractReport(
        method_id="pseudo_label",
        issues=(ContractIssue(code="E_INPUT_RANK", message="rank mismatch"),),
    )
    try:
        try:
            raise ExecutionContractError(report)
        except ExecutionContractError as exc:
            raise BenchRuntimeError("E_BENCH_EXECUTION_CONTRACT", "rejected") from exc
    except BenchRuntimeError as exc:
        wrapped = exc

    monkeypatch.setattr(
        bench_main,
        "_collect_code_runtime_versions",
        lambda **_kwargs: {
            "python": "x",
            "modssc": "x",
            "numpy": "x",
            "git_sha": "x",
        },
    )
    monkeypatch.setattr(
        bench_main,
        "_benchmark_contract_preflight",
        lambda **_kwargs: (_ for _ in ()).throw(wrapped),
    )

    result = bench_main._run_experiment_single(tmp_path / "config.yaml", raw=raw, cfg=cfg)
    payload = json.loads(result.run_json_path.read_text(encoding="utf-8"))
    artifact = payload["artifacts"]["method"]
    persisted_report = artifact["execution_contract"]
    digest = artifact["execution_contract_sha256"]

    assert result.code == 1
    assert payload["run"]["status"] == "failed"
    assert payload["run"]["error_code"] == "E_BENCH_EXECUTION_CONTRACT"
    assert persisted_report["status"] == "incompatible"
    assert digest == execution_contract_payload_sha256(persisted_report)
    assert payload["resolution"]["execution_contract"] == {
        "status": "incompatible",
        "sha256": digest,
    }
