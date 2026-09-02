from __future__ import annotations

from pathlib import Path

import pytest

from bench.schema import ExperimentConfig
from bench.utils.io import load_yaml

CARDS_ROOT = Path(__file__).resolve().parents[2] / "bench" / "configs" / "reproductions"


def _load(relative_path: str) -> ExperimentConfig:
    return ExperimentConfig.from_dict(load_yaml(CARDS_ROOT / relative_path))


@pytest.mark.parametrize(
    ("relative_path", "method_id", "profile", "seeds", "label_mode", "label_value"),
    [
        (
            "pseudo_label/mnist.yaml",
            "pseudo_label",
            "paper:lee2013-mnist-table2-600",
            list(range(1, 11)),
            "per_class",
            60,
        ),
        (
            "tri_training/wdbc_table3_j48.yaml",
            "tri_training",
            "paper:zhou-li-2005-wdbc-table3-j48",
            [1, 2, 3],
            "fraction",
            0.2,
        ),
        (
            "tri_training/vote_table3_j48.yaml",
            "tri_training",
            "paper:zhou-li-2005-vote-table3-j48",
            [1, 2, 3],
            "fraction",
            0.2,
        ),
        (
            "democratic_co_learning/adult.yaml",
            "democratic_co_learning",
            "paper:zhou-goldman-2004-adult-table3",
            list(range(1, 21)),
            "count",
            60,
        ),
        (
            "democratic_co_learning/vote.yaml",
            "democratic_co_learning",
            "paper:zhou-goldman-2004-vote-table3",
            list(range(1, 21)),
            "count",
            40,
        ),
    ],
)
def test_classic_cards_keep_their_native_scientific_contracts(
    relative_path: str,
    method_id: str,
    profile: str,
    seeds: list[int],
    label_mode: str,
    label_value: float,
) -> None:
    config = _load(relative_path)
    labeling = config.sampling.plan["labeling"]

    assert config.method.method_id == method_id
    assert config.method.profile == profile
    assert config.run.seeds == seeds
    assert config.dataset.download is False
    assert labeling["mode"] == label_mode
    assert labeling["value"] == label_value
    assert config.evaluation.report_splits in (["test"], ["val", "test"])


def test_classic_ensemble_cards_use_native_classifier_backends() -> None:
    tri_training = _load("tri_training/vote_table3_j48.yaml")
    democratic = _load("democratic_co_learning/vote.yaml")

    assert tri_training.method.params["classifier_backend"] == "numpy"
    assert tri_training.method.params["prediction_rule"] == "soft_average"
    assert tri_training.method.params["require_convergence"] is True
    assert tri_training.method.params["classifier_params"]["unpruned"] is True

    classifier_specs = democratic.method.params["classifier_specs"]
    assert democratic.method.params["require_convergence"] is True
    assert [spec["classifier_id"] for spec in classifier_specs] == [
        "gaussian_nb",
        "decision_tree",
        "knn",
    ]
    assert {spec["classifier_backend"] for spec in classifier_specs} == {"numpy"}
