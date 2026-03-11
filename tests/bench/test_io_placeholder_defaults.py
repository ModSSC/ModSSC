import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

bench_main = importlib.import_module("bench.main")
report_orch = importlib.import_module("bench.orchestrators.reporting")


def _load_yaml_helper():
    module_path = Path(__file__).resolve().parents[2] / "bench" / "utils" / "io.py"
    spec = importlib.util.spec_from_file_location("bench_utils_io_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_yaml


load_yaml = _load_yaml_helper()


def test_load_yaml_placeholder_defaults_honor_cache_root(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "run:",
                "  output_dir: ${MODSSC_OUTPUT_DIR}/p01",
                "dataset:",
                "  cache_dir: ${MODSSC_DATASET_CACHE_DIR}",
                "preprocess:",
                "  cache_dir: ${MODSSC_PREPROCESS_CACHE_DIR}",
            ]
        ),
        encoding="utf-8",
    )

    root = (tmp_path / "global_root").resolve()
    monkeypatch.setenv("MODSSC_CACHE_ROOT", str(root))
    monkeypatch.delenv("MODSSC_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("MODSSC_DATASET_CACHE_DIR", raising=False)
    monkeypatch.delenv("MODSSC_PREPROCESS_CACHE_DIR", raising=False)

    data = load_yaml(cfg)
    assert data["run"]["output_dir"] == str(root / "output" / "p01")
    assert data["dataset"]["cache_dir"] == str(root / "datasets")
    assert data["preprocess"]["cache_dir"] == str(root / "preprocess")


def test_load_yaml_placeholder_dataset_uses_cache_dir_alias(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("dataset:\n  cache_dir: ${MODSSC_DATASET_CACHE_DIR}\n", encoding="utf-8")

    root = (tmp_path / "global_root").resolve()
    dataset_alias = (tmp_path / "custom_dataset_cache").resolve()
    monkeypatch.setenv("MODSSC_CACHE_ROOT", str(root))
    monkeypatch.setenv("MODSSC_CACHE_DIR", str(dataset_alias))
    monkeypatch.delenv("MODSSC_DATASET_CACHE_DIR", raising=False)

    data = load_yaml(cfg)
    assert data["dataset"]["cache_dir"] == str(dataset_alias)


def _write_fake_run_json(
    path: Path,
    *,
    seed: int,
    status: str,
    metrics: dict | None,
    error: str | None = None,
    error_code: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": {
            "name": f"demo-seed{seed}",
            "seed": seed,
            "run_id": f"rid-{seed}",
            "status": status,
            "error_code": error_code,
        },
        "metrics": metrics,
        "error": error,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_write_seed_sweep_summary_aggregates_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    run1 = tmp_path / "seed1" / "run.json"
    run2 = tmp_path / "seed2" / "run.json"
    run3 = tmp_path / "seed3" / "run.json"
    _write_fake_run_json(
        run1,
        seed=1,
        status="success",
        metrics={"val": {"accuracy": 0.5}, "test": {"accuracy": 0.6, "macro_f1": 0.7}},
    )
    _write_fake_run_json(
        run2,
        seed=2,
        status="success",
        metrics={"val": {"accuracy": 0.7}, "test": {"accuracy": 0.8, "macro_f1": 0.9}},
    )
    _write_fake_run_json(
        run3,
        seed=3,
        status="failed",
        metrics=None,
        error="boom",
        error_code="E_DEMO",
    )

    out = report_orch.write_seed_sweep_summary(
        output_dir=output_dir,
        config_path=tmp_path / "config.yaml",
        base_name="demo",
        requested_seeds=[1, 2, 3],
        run_json_paths=[run1, run2, run3],
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert out == output_dir / "aggregate.json"
    assert payload["sweep"]["status"] == "partial_failure"
    assert payload["sweep"]["requested_run_count"] == 3
    assert payload["sweep"]["successful_run_count"] == 2
    assert payload["sweep"]["failed_run_count"] == 1
    assert payload["metrics"]["val"]["accuracy"]["mean"] == 0.6
    assert payload["metrics"]["val"]["accuracy"]["std"] == pytest.approx(0.1)
    assert payload["metrics"]["test"]["accuracy"]["values"] == [0.6, 0.8]
    assert payload["metrics"]["test"]["macro_f1"]["max"] == 0.9
    assert len(payload["runs"]) == 3
    assert payload["runs"][2]["status"] == "failed"
    assert payload["runs"][2]["error_code"] == "E_DEMO"


def test_run_experiment_writes_seed_sweep_aggregate(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "runs"
    raw = {
        "run": {
            "name": "demo",
            "seed": 7,
            "seeds": [7, 8],
            "output_dir": str(output_dir),
        },
        "dataset": {"id": "toy"},
        "sampling": {"plan": {"split": {"kind": "holdout"}}},
        "preprocess": {"plan": {"output_key": "features.X", "steps": [{"id": "core.ensure_2d"}]}},
        "method": {
            "kind": "inductive",
            "id": "pseudo_label",
            "device": {"device": "cpu", "dtype": "float32"},
        },
        "evaluation": {"report_splits": ["val"], "metrics": ["accuracy"]},
    }

    monkeypatch.setattr(bench_main, "load_yaml", lambda _path: raw)

    def _fake_single(_config_path: Path, *, raw: dict, cfg) -> bench_main.SingleRunResult:
        run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}-fake"
        run_json = run_dir / "run.json"
        _write_fake_run_json(
            run_json,
            seed=int(cfg.run.seed),
            status="success",
            metrics={"val": {"accuracy": 0.5 if int(cfg.run.seed) == 7 else 0.7}},
        )
        return bench_main.SingleRunResult(code=0, run_dir=run_dir, run_json_path=run_json)

    monkeypatch.setattr(bench_main, "_run_experiment_single", _fake_single)

    code = bench_main.run_experiment(tmp_path / "config.yaml")
    sweep_dirs = sorted(output_dir.glob("demo-sweep-*"))
    assert len(sweep_dirs) == 1
    payload = json.loads((sweep_dirs[0] / "aggregate.json").read_text(encoding="utf-8"))

    assert code == 0
    assert payload["sweep"]["output_dir"] == str(sweep_dirs[0])
    assert payload["sweep"]["requested_seeds"] == [7, 8]
    assert payload["metrics"]["val"]["accuracy"]["mean"] == 0.6
    assert len(payload["runs"]) == 2
