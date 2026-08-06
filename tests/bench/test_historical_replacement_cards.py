from __future__ import annotations

from pathlib import Path

from bench.orchestrators.views import _plan_from_dict as views_plan_from_dict
from bench.schema import ExperimentConfig
from bench.utils.io import load_yaml
from modssc.inductive.methods.co_training import CoTrainingSpec
from modssc.inductive.methods.self_training import SelfTrainingSpec

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "bench" / "configs"
REPRODUCTION_ROOT = CONFIG_ROOT / "reproductions"


def _load_config(relative_path: str) -> ExperimentConfig:
    return ExperimentConfig.from_dict(load_yaml(REPRODUCTION_ROOT / relative_path))


def test_self_training_wine_card_freezes_the_li_zhou_reconstruction() -> None:
    cfg = _load_config("self_training/wine_table3.yaml")

    assert cfg.run.seeds == list(range(1, 51))
    assert cfg.run.seeded_sections == ["sampling", "preprocess"]
    assert cfg.dataset.id == "wine"
    assert cfg.dataset.download is False
    assert cfg.preprocess.fit_on == "train_labeled"
    assert [step["id"] for step in cfg.preprocess.plan["steps"]] == [
        "labels.encode",
        "core.ensure_2d",
        "tabular.standard_scaler",
        "core.to_numpy",
    ]
    assert cfg.sampling.plan["split"] == {
        "kind": "holdout",
        "test_fraction": 0.25,
        "val_fraction": 0.0,
        "stratify": True,
        "shuffle": True,
    }
    assert cfg.sampling.plan["labeling"] == {
        "mode": "fraction",
        "value": 0.1,
        "strategy": "proportional",
        "min_per_class": 1,
        "per_class": False,
        "fixed_indices": None,
    }
    assert cfg.method.method_id == "self_training"
    assert cfg.method.profile == "paper:li-zhou-2005-setred-table3-wine-self-training"
    assert cfg.method.params == {
        "classifier_id": "knn",
        "classifier_backend": "numpy",
        "classifier_params": {"k": 1, "metric": "euclidean", "weights": "uniform"},
        "max_iter": 40,
        "confidence_threshold": None,
        "max_new_labels": None,
        "min_new_labels": 1,
        "use_group_propagation": False,
        "selection_strategy": "li_zhou_2005_1nn_distance",
        "paper_pool_size_unspecified": None,
        "paper_candidates_per_class_unspecified": 1,
        "paper_distance_confidence_unspecified": "margin",
    }
    SelfTrainingSpec(**cfg.method.params)
    assert cfg.evaluation.split_for_model_selection is None
    assert cfg.evaluation.report_splits == ["test"]


def test_self_training_wine_confirmation_v2_uses_fresh_splits_and_dynamic_distances() -> None:
    cfg = _load_config("self_training/wine_table3_confirmation_v2.yaml")

    assert cfg.run.seed == 51
    assert cfg.run.seeds == list(range(51, 101))
    assert cfg.run.seeded_sections == ["sampling", "preprocess"]
    assert cfg.dataset.id == "wine"
    assert cfg.dataset.download is False
    assert cfg.preprocess.fit_on == "train_labeled"
    assert [step["id"] for step in cfg.preprocess.plan["steps"]] == [
        "labels.encode",
        "core.ensure_2d",
        "core.to_numpy",
    ]
    assert cfg.sampling.plan["split"] == {
        "kind": "holdout",
        "test_fraction": 0.25,
        "val_fraction": 0.0,
        "stratify": True,
        "shuffle": True,
    }
    assert cfg.sampling.plan["labeling"] == {
        "mode": "fraction",
        "value": 0.1,
        "strategy": "proportional",
        "min_per_class": 1,
        "per_class": False,
        "fixed_indices": None,
    }
    assert cfg.method.method_id == "self_training"
    assert (
        cfg.method.profile == "paper:li-zhou-2005-setred-table3-wine-self-training-confirmation-v2"
    )
    assert cfg.method.params == {
        "classifier_id": "knn",
        "classifier_backend": "numpy",
        "classifier_params": {"k": 1, "metric": "euclidean", "weights": "uniform"},
        "max_iter": 40,
        "confidence_threshold": None,
        "max_new_labels": None,
        "min_new_labels": 1,
        "use_group_propagation": False,
        "selection_strategy": "li_zhou_2005_1nn_distance",
        "paper_pool_size_unspecified": 75,
        "paper_candidates_per_class_unspecified": 1,
        "paper_distance_confidence_unspecified": "nearest_neighbor_distance",
        "paper_feature_scaling_unspecified": "dynamic_labeled_minmax",
    }
    SelfTrainingSpec(**cfg.method.params)
    assert cfg.evaluation.split_for_model_selection is None
    assert cfg.evaluation.report_splits == ["test"]


def test_self_training_v2_controls_are_unclaimable_diagnostic_profiles() -> None:
    confirmation = _load_config("self_training/wine_table3_confirmation_v2.yaml")
    controls = [
        ExperimentConfig.from_dict(
            load_yaml(CONFIG_ROOT / "diagnostics" / "self_training" / filename)
        )
        for filename in (
            "wine_nn_l_confirmation_v2.yaml",
            "wine_nn_a_confirmation_v2.yaml",
        )
    ]

    assert not confirmation.method.profile.endswith(":diagnostic-dev")
    assert {config.method.profile for config in controls} == {
        "paper:li-zhou-2005-wine-nn-l-confirmation-v2:diagnostic-dev",
        "paper:li-zhou-2005-wine-nn-a-confirmation-v2:diagnostic-dev",
    }
    for config in controls:
        assert config.run.seeds == list(range(51, 101))
        assert config.method.profile.endswith(":diagnostic-dev")
        assert config.method.params["max_iter"] == 0
        assert config.evaluation.report_splits == ["test"]


def test_co_training_webkb_card_freezes_the_blum_mitchell_protocol() -> None:
    cfg = _load_config("co_training/webkb_course_table2.yaml")

    assert cfg.run.seeds == [1, 2, 3, 4, 5]
    assert cfg.run.seeded_sections == ["sampling", "preprocess", "views"]
    assert cfg.dataset.id == "webkb_course_cotraining"
    assert cfg.dataset.download is False
    assert cfg.preprocess.fit_on == "train"
    assert cfg.sampling.plan["split"] == {
        "kind": "holdout",
        "test_fraction": 263 / 1051,
        "val_fraction": 0.0,
        "stratify": True,
        "shuffle": True,
    }
    assert cfg.sampling.plan["labeling"] == {
        "mode": "count",
        "value": 12,
        "strategy": "proportional",
        "min_per_class": 1,
        "per_class": False,
        "fixed_indices": None,
    }
    assert cfg.views is not None
    views = cfg.views.plan["views"]
    assert [view["name"] for view in views] == ["fulltext", "inlinks"]
    assert [view["input_columns"]["indices"] for view in views] == [[0], [1]]
    assert all(
        [step["id"] for step in view["preprocess"]["steps"]]
        == ["text.ensure_strings", "text.count_vectorizer"]
        for view in views
    )
    assert all(
        view["preprocess"]["steps"][1]["params"] == {"dense": True, "strip_html": True}
        for view in views
    )
    views_plan_from_dict(cfg.views.plan)

    assert cfg.method.method_id == "co_training"
    assert cfg.method.profile == "paper:blum-mitchell-1998-webkb-course-table2"
    assert cfg.method.params == {
        "classifier_id": "multinomial_nb",
        "classifier_backend": "sklearn",
        "classifier_params": {"alpha": 1.0, "fit_prior": True},
        "view_keys": ["fulltext", "inlinks"],
        "protocol": "fixed_pool_binary",
        "p": 1,
        "n": 3,
        "u": 75,
        "k": 30,
        "positive_label": 1,
        "negative_label": 0,
        "confidence_threshold": None,
    }
    CoTrainingSpec(**cfg.method.params)
    assert cfg.evaluation.split_for_model_selection is None
    assert cfg.evaluation.report_splits == ["test"]
