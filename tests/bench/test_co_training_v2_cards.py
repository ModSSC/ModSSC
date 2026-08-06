from __future__ import annotations

from pathlib import Path

from bench.schema import ExperimentConfig
from bench.utils.io import load_yaml
from modssc.inductive.methods.co_training import CoTrainingSpec, _validate_protocol

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str) -> ExperimentConfig:
    return ExperimentConfig.from_dict(load_yaml(REPO_ROOT / relative))


def test_co_training_v2_cards_are_test_blind_then_use_disjoint_confirmation_seeds() -> None:
    diagnostic = _load("bench/configs/diagnostics/co_training/webkb_course_v2.yaml")
    confirmation = _load("bench/configs/reproductions/co_training/webkb_course_table2_v2.yaml")

    assert diagnostic.run.seeds == [1, 2, 3, 4, 5]
    assert confirmation.run.seeds == [6, 7, 8, 9, 10]
    assert set(diagnostic.run.seeds).isdisjoint(confirmation.run.seeds)
    assert diagnostic.evaluation.report_splits == ["train_labeled"]
    assert confirmation.evaluation.report_splits == ["test"]
    assert diagnostic.evaluation.split_for_model_selection is None
    assert confirmation.evaluation.split_for_model_selection is None
    assert diagnostic.sampling.plan["split"]["stratify"] is True
    assert confirmation.sampling.plan["split"]["stratify"] is False
    assert diagnostic.method.profile.endswith(":diagnostic-dev")
    assert not confirmation.method.profile.endswith(":diagnostic-dev")

    for config in (diagnostic, confirmation):
        assert config.method.params["dynamic_feature_selection"] == ("mutual_information_presence")
        assert config.method.params["feature_selection_max_features"] == 2000
        assert config.method.params["selection_score"] == "craven_1998_normalized_nb"
        _validate_protocol(CoTrainingSpec(**config.method.params))


def test_co_training_v1_card_remains_immutable() -> None:
    v1 = _load("bench/configs/reproductions/co_training/webkb_course_table2.yaml")

    assert v1.method.params["protocol"] == "fixed_pool_binary"
    assert "dynamic_feature_selection" not in v1.method.params
    assert "feature_selection_max_features" not in v1.method.params
    assert "selection_score" not in v1.method.params
    _validate_protocol(CoTrainingSpec(**v1.method.params))
