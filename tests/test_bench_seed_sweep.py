from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from bench import main as bench_main
from bench.schema import BenchConfigError, ExperimentConfig
from bench.seed_sweep import apply_global_seed


def _base_config() -> dict:
    return {
        "run": {
            "name": "toy",
            "seed": 0,
            "output_dir": "runs",
            "fail_fast": False,
        },
        "dataset": {"id": "toy"},
        "sampling": {
            "plan": {
                "split": {
                    "kind": "holdout",
                    "test_fraction": 0.0,
                    "val_fraction": 0.2,
                    "stratify": True,
                    "shuffle": True,
                },
                "labeling": {
                    "mode": "fraction",
                    "value": 0.2,
                    "strategy": "balanced",
                    "min_per_class": 1,
                },
                "imbalance": {"kind": "none"},
                "policy": {"respect_official_test": True, "allow_override_official": False},
            }
        },
        "preprocess": {
            "cache": True,
            "plan": {
                "output_key": "features.X",
                "steps": [{"id": "core.ensure_2d"}, {"id": "core.to_numpy"}],
            },
        },
        "method": {
            "kind": "inductive",
            "id": "pseudo_label",
            "device": {"device": "cpu", "dtype": "float32"},
            "params": {
                "classifier_id": "knn",
                "classifier_backend": "numpy",
                "max_iter": 1,
                "confidence_threshold": 0.8,
            },
        },
        "evaluation": {"report_splits": ["val"], "metrics": ["accuracy"]},
    }


def test_run_seeds_is_accepted() -> None:
    raw = _base_config()
    raw["run"]["seeds"] = [1, 2, 3]
    cfg = ExperimentConfig.from_dict(raw)
    assert cfg.run.seeds == [1, 2, 3]


def test_run_seeds_rejects_duplicates() -> None:
    raw = _base_config()
    raw["run"]["seeds"] = [1, 1]
    with pytest.raises(BenchConfigError, match="run\\.seeds must not contain duplicates"):
        ExperimentConfig.from_dict(raw)


def test_apply_global_seed_overrides_all_seed_blocks() -> None:
    raw = _base_config()
    raw["run"]["seeds"] = [2, 3]
    raw["sampling"]["seed"] = 10
    raw["preprocess"]["seed"] = 11
    raw["views"] = {"seed": 12, "plan": {"views": [{"name": "v1"}]}}
    raw["graph"] = {"enabled": False, "seed": 13}
    raw["augmentation"] = {"enabled": False, "seed": 14, "mode": "fixed", "weak": {}, "strong": {}}
    raw["search"] = {
        "enabled": True,
        "kind": "random",
        "seed": 15,
        "n_trials": 1,
        "repeats": 1,
        "objective": {
            "split": "val",
            "metric": "accuracy",
            "direction": "maximize",
            "aggregate": "mean",
        },
        "space": {"method": {"params": {"max_iter": [1]}}},
    }

    out = apply_global_seed(raw, seed=99, run_name="toy-seed99")
    assert raw["run"]["seed"] == 0
    assert out["run"]["seed"] == 99
    assert out["run"]["name"] == "toy-seed99"
    assert "seeds" not in out["run"]
    assert out["sampling"]["seed"] == 99
    assert out["preprocess"]["seed"] == 99
    assert out["views"]["seed"] == 99
    assert out["graph"]["seed"] == 99
    assert out["augmentation"]["seed"] == 99
    assert out["search"]["seed"] == 99


def test_run_experiment_dispatches_multi_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _base_config()
    raw["run"]["seeds"] = [7, 8, 9]

    calls: list[tuple[int, int | None, int | None]] = []

    def fake_load_yaml(_path: Path) -> dict:
        return deepcopy(raw)

    def fake_single(_config_path: Path, *, raw: dict, cfg: ExperimentConfig) -> int:
        calls.append((cfg.run.seed, cfg.sampling.seed, cfg.preprocess.seed))
        return 1 if cfg.run.seed == 8 else 0

    monkeypatch.setattr(bench_main, "load_yaml", fake_load_yaml)
    monkeypatch.setattr(bench_main, "_run_experiment_single", fake_single)

    code = bench_main.run_experiment(tmp_path / "cfg.yaml")
    assert code == 1
    assert calls == [(7, 7, 7), (8, 8, 8), (9, 9, 9)]


def test_run_experiment_single_path_when_no_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _base_config()
    calls: list[int] = []

    def fake_load_yaml(_path: Path) -> dict:
        return deepcopy(raw)

    def fake_single(_config_path: Path, *, raw: dict, cfg: ExperimentConfig) -> int:
        calls.append(cfg.run.seed)
        return 0

    monkeypatch.setattr(bench_main, "load_yaml", fake_load_yaml)
    monkeypatch.setattr(bench_main, "_run_experiment_single", fake_single)

    code = bench_main.run_experiment(tmp_path / "cfg.yaml")
    assert code == 0
    assert calls == [0]
