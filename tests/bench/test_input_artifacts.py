from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from bench import main as bench_main
from bench.orchestrators import input_artifacts as artifact_orch
from bench.schema import BenchConfigError, ExperimentConfig
from bench.utils.io import load_yaml
from modssc.runtime import ArtifactContract, artifact_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
TOY_CONFIG = REPO_ROOT / "bench" / "configs" / "experiments" / "toy_inductive.yaml"


def _minimal_config() -> dict[str, Any]:
    return {
        "run": {"name": "artifact_contract", "seed": 1, "output_dir": "runs"},
        "dataset": {"id": "toy"},
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


def test_run_schema_parses_native_input_artifact_contracts() -> None:
    raw = _minimal_config()
    raw["run"].update(
        {
            "artifact_root": "${MODEL_CACHE}",
            "input_artifacts": [
                {"path": "encoder/weights.bin", "kind": "file", "sha256": "a" * 64},
                {"path": "tokenizer", "kind": "tree", "sha256": "b" * 64},
            ],
        }
    )

    cfg = ExperimentConfig.from_dict(raw)

    assert cfg.run.artifact_root == "${MODEL_CACHE}"
    assert cfg.run.input_artifacts == [
        ArtifactContract("encoder/weights.bin", "a" * 64, "file"),
        ArtifactContract("tokenizer", "b" * 64, "tree"),
    ]


def test_run_schema_requires_root_for_declared_inputs() -> None:
    raw = _minimal_config()
    raw["run"]["input_artifacts"] = [{"path": "weights.bin", "kind": "file", "sha256": "a" * 64}]

    with pytest.raises(BenchConfigError, match="artifact_root"):
        ExperimentConfig.from_dict(raw)


@pytest.mark.parametrize(
    "entry",
    [
        {"path": "weights.bin", "kind": "file"},
        {"path": "../weights.bin", "kind": "file", "sha256": "a" * 64},
        {"path": "weights.bin", "kind": "archive", "sha256": "a" * 64},
        {"path": "weights.bin", "kind": "file", "sha256": "invalid"},
        {"path": "weights.bin", "kind": "file", "sha256": "a" * 64, "model": "x"},
    ],
)
def test_run_schema_rejects_invalid_input_artifacts(entry: dict[str, Any]) -> None:
    raw = _minimal_config()
    raw["run"].update({"artifact_root": "/cache", "input_artifacts": [entry]})

    with pytest.raises(BenchConfigError, match="input_artifacts"):
        ExperimentConfig.from_dict(raw)


def test_run_schema_rejects_duplicate_input_artifact_paths() -> None:
    raw = _minimal_config()
    entry = {"path": "weights.bin", "kind": "file", "sha256": "a" * 64}
    raw["run"].update({"artifact_root": "/cache", "input_artifacts": [entry, entry]})

    with pytest.raises(BenchConfigError, match="duplicate path"):
        ExperimentConfig.from_dict(raw)


def test_thin_orchestrator_resolves_relative_root_and_revalidates(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "run.yaml"
    artifact_root = config_path.parent / "cache"
    artifact_root.mkdir(parents=True)
    weights = artifact_root / "weights.bin"
    weights.write_bytes(b"weights")
    contract = ArtifactContract(
        path="weights.bin",
        kind="file",
        sha256=artifact_sha256(artifact_root, path="weights.bin", kind="file"),
    )

    preflight = artifact_orch.preflight(
        [contract],
        artifact_root="cache",
        config_path=config_path,
    )
    payload = preflight.report_payload()

    assert preflight.root == artifact_root.resolve()
    assert payload["revalidated_before_success"] is False
    assert str(artifact_root.resolve()) not in json.dumps(payload)

    original = weights.stat()
    os.utime(weights, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
    with pytest.raises(RuntimeError, match="E_BENCH_INPUT_ARTIFACT_INTEGRITY"):
        artifact_orch.revalidate(preflight)


def test_short_runner_verifies_before_dataset_and_revalidates_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "weights.bin").write_bytes(b"weights")
    digest = artifact_sha256(artifact_root, path="weights.bin", kind="file")

    raw = load_yaml(TOY_CONFIG)
    raw["run"].update(
        {
            "name": "input_artifact_short_gate",
            "output_dir": str(tmp_path / "runs"),
            "artifact_root": str(artifact_root),
            "input_artifacts": [{"path": "weights.bin", "kind": "file", "sha256": digest}],
        }
    )
    raw["preprocess"]["cache"] = False
    cfg = ExperimentConfig.from_dict(raw)
    events: list[str] = []

    original_preflight = bench_main.input_artifact_orch.preflight
    original_revalidate = bench_main.input_artifact_orch.revalidate
    original_load = bench_main.ds_orch.load
    original_summary = bench_main.report_orch.write_run_summary

    def _preflight(*args: Any, **kwargs: Any):
        events.append("artifact_preflight")
        return original_preflight(*args, **kwargs)

    def _load(*args: Any, **kwargs: Any):
        events.append("dataset_load")
        assert events.index("artifact_preflight") < events.index("dataset_load")
        return original_load(*args, **kwargs)

    def _revalidate(*args: Any, **kwargs: Any):
        events.append("artifact_revalidate")
        return original_revalidate(*args, **kwargs)

    def _summary(**kwargs: Any) -> None:
        events.append(f"summary_{kwargs['status']}")
        if kwargs["status"] == "success":
            assert events.index("artifact_revalidate") < events.index("summary_success")
        original_summary(**kwargs)

    monkeypatch.setattr(bench_main.input_artifact_orch, "preflight", _preflight)
    monkeypatch.setattr(bench_main.input_artifact_orch, "revalidate", _revalidate)
    monkeypatch.setattr(bench_main.ds_orch, "load", _load)
    monkeypatch.setattr(bench_main.report_orch, "write_run_summary", _summary)

    result = bench_main._run_experiment_single(TOY_CONFIG, raw=raw, cfg=cfg)

    assert result.code == 0
    payload = json.loads(result.run_json_path.read_text(encoding="utf-8"))
    recorded = payload["artifacts"]["input_artifacts"]
    assert recorded["revalidated_before_success"] is True
    assert recorded["attestations"][0]["contract"]["sha256"] == digest


def test_short_runner_fails_closed_when_input_changes_during_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    weights = artifact_root / "weights.bin"
    weights.write_bytes(b"weights")
    digest = artifact_sha256(artifact_root, path="weights.bin", kind="file")

    raw = load_yaml(TOY_CONFIG)
    raw["run"].update(
        {
            "name": "input_artifact_mutation_gate",
            "output_dir": str(tmp_path / "runs"),
            "fail_fast": False,
            "artifact_root": str(artifact_root),
            "input_artifacts": [{"path": "weights.bin", "kind": "file", "sha256": digest}],
        }
    )
    raw["preprocess"]["cache"] = False
    cfg = ExperimentConfig.from_dict(raw)
    original_evaluate = bench_main.eval_orch.evaluate_inductive

    def _evaluate_and_mutate(*args: Any, **kwargs: Any):
        metrics = original_evaluate(*args, **kwargs)
        weights.write_bytes(b"weights changed during execution")
        return metrics

    monkeypatch.setattr(bench_main.eval_orch, "evaluate_inductive", _evaluate_and_mutate)

    result = bench_main._run_experiment_single(TOY_CONFIG, raw=raw, cfg=cfg)

    assert result.code == 1
    payload = json.loads(result.run_json_path.read_text(encoding="utf-8"))
    assert payload["run"]["status"] == "failed"
    assert payload["run"]["error_code"] == "E_BENCH_INPUT_ARTIFACT_INTEGRITY"
    assert payload["artifacts"]["input_artifacts"]["revalidated_before_success"] is False
    assert "changed after preflight" in payload["error"]
