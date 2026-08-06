from __future__ import annotations

import pytest

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
        "dataset": {"id": "toy", "options": {"seed": 10}},
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
    assert seeded["dataset"]["options"]["seed"] == 10


def test_apply_global_seed_does_not_seed_dataset_by_default() -> None:
    raw = _minimal_config({"seeds": [1, 2]})

    seeded = apply_global_seed(raw, seed=99)

    assert seeded["sampling"]["seed"] == 99
    assert seeded["preprocess"]["seed"] == 99
    assert seeded["dataset"]["options"]["seed"] == 10


def test_apply_global_seed_can_explicitly_seed_dataset_options() -> None:
    raw = _minimal_config(
        {"seeds": [1, 2], "seeded_sections": ["dataset", "sampling", "preprocess"]}
    )

    seeded = apply_global_seed(
        raw,
        seed=99,
        seeded_sections=["dataset", "sampling", "preprocess"],
    )

    assert seeded["dataset"]["options"]["seed"] == 99
    assert seeded["sampling"]["seed"] == 99
    assert seeded["preprocess"]["seed"] == 99


def test_apply_global_seed_creates_missing_dataset_options() -> None:
    raw = _minimal_config({})
    raw["dataset"] = {"id": "toy"}

    seeded = apply_global_seed(raw, seed=99, seeded_sections=["dataset"])

    assert seeded["dataset"]["options"]["seed"] == 99


def test_apply_global_seed_rejects_non_mapping_dataset_options() -> None:
    raw = _minimal_config({})
    raw["dataset"] = {"id": "toy", "options": "invalid"}

    with pytest.raises(ValueError, match="dataset.options must be a mapping"):
        apply_global_seed(raw, seed=99, seeded_sections=["dataset"])
