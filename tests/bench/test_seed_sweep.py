from __future__ import annotations

from bench.schema import ExperimentConfig
from bench.seed_sweep import apply_global_seed


def _minimal_config(run: dict[str, object]) -> dict[str, object]:
    return {
        "run": {
            "name": "seeded_sections",
            "seed": 1,
            "output_dir": "runs",
            **run,
        },
        "dataset": {"id": "toy"},
        "sampling": {"seed": 11, "plan": {"split": {"kind": "holdout"}}},
        "preprocess": {
            "seed": 12,
            "fit_on": "train_labeled",
            "plan": {"output_key": "features.X", "steps": [{"id": "core.to_numpy"}]},
        },
        "method": {
            "kind": "inductive",
            "id": "pseudo_label",
            "device": {"device": "auto", "dtype": "float32"},
            "params": {},
        },
        "evaluation": {
            "split_for_model_selection": "val",
            "report_splits": ["val", "test"],
            "metrics": ["accuracy"],
        },
    }


def test_seeded_sections_preserves_absent_empty_and_explicit_values() -> None:
    assert ExperimentConfig.from_dict(_minimal_config({})).run.seeded_sections is None
    assert (
        ExperimentConfig.from_dict(_minimal_config({"seeded_sections": []})).run.seeded_sections
        == []
    )
    assert ExperimentConfig.from_dict(
        _minimal_config({"seeded_sections": ["sampling"]})
    ).run.seeded_sections == ["sampling"]


def test_apply_global_seed_allows_empty_seeded_sections() -> None:
    raw = _minimal_config({"seeds": [1, 2], "seeded_sections": []})

    seeded = apply_global_seed(raw, seed=99, run_name="seed99", seeded_sections=[])

    assert seeded["run"]["seed"] == 99
    assert seeded["run"]["name"] == "seed99"
    assert "seeds" not in seeded["run"]
    assert seeded["sampling"]["seed"] == 11
    assert seeded["preprocess"]["seed"] == 12
