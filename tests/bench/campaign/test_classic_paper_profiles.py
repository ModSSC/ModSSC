from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bench.campaign import generate as generate_module
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from bench.schema import ExperimentConfig
from bench.utils.io import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
REPRODUCTION_ROOT = REPO_ROOT / "bench" / "configs" / "reproductions"


def _load_config(relative_path: str) -> ExperimentConfig:
    return ExperimentConfig.from_dict(load_yaml(REPRODUCTION_ROOT / relative_path))


def test_pseudo_label_mnist_table2_card_tracks_lee2013_plus_pl() -> None:
    cfg = _load_config("pseudo_label/mnist.yaml")

    assert cfg.method.method_id == "pseudo_label"
    assert cfg.method.profile == "paper:lee2013-mnist-table2-600"
    assert cfg.method.params == {
        "training_mode": "joint_mlp",
        "classifier_id": "mlp",
        "classifier_backend": "torch",
        "classifier_params": {},
        "paper_input_dim": 784,
        "paper_hidden_units": 5000,
        "paper_num_classes": 10,
        "paper_epochs": 601,
        "paper_labeled_batch_size": 32,
        "paper_unlabeled_batch_size": 256,
        "paper_hidden_dropout": 0.5,
        "paper_input_dropout": 0.0,
        "paper_initial_learning_rate": 1.5,
        "paper_learning_rate_decay": 0.998,
        "paper_momentum_initial": 0.5,
        "paper_momentum_final": 0.99,
        "paper_momentum_ramp_epochs": 500,
        "paper_alpha_final": 3.0,
        "paper_alpha_start_epoch": 100,
        "paper_alpha_end_epoch": 600,
    }
    assert cfg.dataset.id == "mnist"
    assert cfg.dataset.download is False
    assert cfg.run.seeds == list(range(1, 11))
    assert cfg.run.seeded_sections == ["sampling", "preprocess"]
    assert cfg.evaluation.split_for_model_selection == "val"
    assert cfg.evaluation.report_splits == ["val", "test"]

    split = cfg.sampling.plan["split"]
    labeling = cfg.sampling.plan["labeling"]
    assert split == {
        "kind": "holdout",
        "test_fraction": 0.2,
        "val_fraction": 1.0 / 60.0,
        "stratify": True,
        "shuffle": True,
    }
    assert labeling == {
        "mode": "per_class",
        "value": 60,
        "strategy": "balanced",
        "min_per_class": 60,
        "per_class": True,
        "fixed_indices": None,
    }
    assert cfg.sampling.plan["policy"]["respect_official_test"] is True


def test_tri_training_wdbc_table3_card_is_offline_and_test_blind() -> None:
    cfg = _load_config("tri_training/wdbc_table3_j48.yaml")

    assert cfg.method.method_id == "tri_training"
    assert cfg.method.profile == "paper:zhou-li-2005-wdbc-table3-j48"
    assert cfg.method.params["classifier_id"] == "decision_tree"
    assert cfg.method.params["classifier_params"] == {
        "min_num_obj": 2,
        "unpruned": True,
        "binary_splits": False,
    }
    assert cfg.dataset.id == "wdbc"
    assert cfg.dataset.options == {}
    assert cfg.dataset.download is False
    assert cfg.run.seeds == [1, 2, 3]
    assert cfg.run.seeded_sections == ["sampling", "preprocess"]
    assert cfg.evaluation.split_for_model_selection is None
    assert cfg.evaluation.report_splits == ["test"]
    assert cfg.sampling.plan["component_seeds"] == {"split": 2005}

    split = cfg.sampling.plan["split"]
    labeling = cfg.sampling.plan["labeling"]
    assert split == {
        "kind": "holdout",
        "test_fraction": 0.25,
        "val_fraction": 0.0,
        "stratify": True,
        "shuffle": True,
    }
    assert labeling["mode"] == "fraction"
    assert labeling["value"] == 0.2
    assert labeling["strategy"] == "proportional"
    assert cfg.method.params["classifier_backend"] == "numpy"
    assert cfg.method.params["max_iter"] == 100
    assert cfg.method.params["confidence_threshold"] is None
    assert cfg.method.params["prediction_rule"] == "soft_average"

    test_count = round(569 * split["test_fraction"])
    train_count = 569 - test_count
    labeled_count = round(train_count * labeling["value"])
    assert (test_count, labeled_count, train_count - labeled_count) == (142, 85, 342)


def test_tri_training_vote_table3_card_is_offline_and_test_blind() -> None:
    cfg = _load_config("tri_training/vote_table3_j48.yaml")

    assert cfg.method.method_id == "tri_training"
    assert cfg.method.profile == "paper:zhou-li-2005-vote-table3-j48"
    assert cfg.dataset.id == "vote"
    assert cfg.dataset.options == {}
    assert cfg.dataset.download is False
    assert cfg.run.seeds == [1, 2, 3]
    assert cfg.run.seeded_sections == ["sampling", "preprocess"]
    assert cfg.evaluation.split_for_model_selection is None
    assert cfg.evaluation.report_splits == ["test"]
    assert cfg.method.params["classifier_backend"] == "numpy"
    assert cfg.method.params["retain_initial_ensemble"] is True
    assert cfg.method.params["prediction_rule"] == "soft_average"
    assert cfg.method.params["classifier_params"]["unpruned"] is True
    assert cfg.method.params["classifier_params"]["binary_splits"] is False
    assert [step["id"] for step in cfg.preprocess.plan["steps"]] == [
        "labels.encode",
        "core.copy_raw",
        "core.to_numpy",
    ]
    feature_schema = cfg.method.params["classifier_params"]["feature_schema"]
    assert len(feature_schema) == 16
    assert all(item == {"type": "nominal", "values": ["n", "y"]} for item in feature_schema)
    assert cfg.method.params["classifier_params"]["missing_values"] == ["?"]

    split = cfg.sampling.plan["split"]
    labeling = cfg.sampling.plan["labeling"]
    assert cfg.sampling.plan["component_seeds"] == {"split": 2005}
    assert split == {
        "kind": "holdout",
        "test_fraction": 0.25,
        "val_fraction": 0.0,
        "stratify": True,
        "shuffle": True,
    }
    assert labeling["mode"] == "fraction"
    assert labeling["value"] == 0.2
    assert labeling["strategy"] == "proportional"

    test_count = round(435 * split["test_fraction"])
    train_count = 435 - test_count
    labeled_count = round(train_count * labeling["value"])
    assert (test_count, labeled_count, train_count - labeled_count) == (109, 65, 261)


def test_democratic_co_learning_adult_table3_card_is_offline_and_test_blind() -> None:
    cfg = _load_config("democratic_co_learning/adult.yaml")

    assert cfg.method.method_id == "democratic_co_learning"
    assert cfg.method.profile == "paper:zhou-goldman-2004-adult-table3"
    assert cfg.dataset.id == "adult"
    assert cfg.dataset.download is False
    assert cfg.dataset.options == {}
    assert cfg.run.seeds == list(range(1, 21))
    assert cfg.run.seeded_sections == ["sampling", "preprocess"]
    assert cfg.evaluation.split_for_model_selection is None
    assert cfg.evaluation.report_splits == ["test"]
    assert [item["classifier_backend"] for item in cfg.method.params["classifier_specs"]] == [
        "numpy",
        "numpy",
        "numpy",
    ]
    assert cfg.method.params["classifier_specs"][0]["classifier_params"] == {
        "alpha": 1.0,
        "fit_prior": True,
    }
    assert cfg.method.params["classifier_specs"][1]["classifier_params"] == {
        "min_num_obj": 2,
        "unpruned": True,
        "binary_splits": False,
    }
    assert cfg.method.params["classifier_specs"][2]["classifier_params"] == {
        "k": 3,
        "metric": "euclidean",
        "weights": "uniform",
    }

    partition = cfg.sampling.plan["partition"]
    sample_count = partition["max_samples"]
    split = cfg.sampling.plan["split"]
    labeling = cfg.sampling.plan["labeling"]
    assert split["val_fraction"] == 0.0
    assert split["stratify"] is False
    assert partition == {"max_samples": 3442, "shuffle": True}
    assert labeling["mode"] == "count"
    assert labeling["value"] == 60
    assert labeling["strategy"] == "random"

    test_count = round(sample_count * split["test_fraction"])
    unlabeled_count = sample_count - test_count - labeling["value"]
    assert (labeling["value"], unlabeled_count, test_count) == (60, 1691, 1691)


def test_democratic_co_learning_vote_table3_card_is_offline_and_test_blind() -> None:
    cfg = _load_config("democratic_co_learning/vote.yaml")

    assert cfg.method.method_id == "democratic_co_learning"
    assert cfg.method.profile == "paper:zhou-goldman-2004-vote-table3"
    assert cfg.dataset.id == "vote"
    assert cfg.dataset.options == {}
    assert cfg.dataset.download is False
    assert cfg.run.seeds == list(range(1, 21))
    assert cfg.run.seeded_sections == ["sampling", "preprocess"]
    assert cfg.evaluation.split_for_model_selection is None
    assert cfg.evaluation.report_splits == ["test"]
    assert cfg.method.params["max_iter"] == 20
    assert cfg.method.params["confidence_level"] == 0.95
    assert cfg.method.params["min_confidence"] == 0.5
    assert [item["classifier_id"] for item in cfg.method.params["classifier_specs"]] == [
        "gaussian_nb",
        "decision_tree",
        "knn",
    ]
    assert [item["classifier_backend"] for item in cfg.method.params["classifier_specs"]] == [
        "numpy",
        "numpy",
        "numpy",
    ]
    assert [step["id"] for step in cfg.preprocess.plan["steps"]] == [
        "labels.encode",
        "core.copy_raw",
        "core.to_numpy",
    ]
    for classifier_spec in cfg.method.params["classifier_specs"]:
        feature_schema = classifier_spec["classifier_params"]["feature_schema"]
        assert len(feature_schema) == 16
        assert all(item == {"type": "nominal", "values": ["n", "y"]} for item in feature_schema)
        assert classifier_spec["classifier_params"]["missing_values"] == ["?"]
    assert cfg.method.params["classifier_specs"][0]["classifier_params"]["alpha"] == 1.0
    assert cfg.method.params["classifier_specs"][0]["classifier_params"]["fit_prior"] is True
    assert cfg.method.params["classifier_specs"][1]["classifier_params"]["unpruned"] is True
    assert cfg.method.params["classifier_specs"][1]["classifier_params"]["binary_splits"] is False
    assert cfg.method.params["classifier_specs"][2]["classifier_params"]["metric"] == "euclidean"
    assert cfg.method.params["classifier_specs"][2]["classifier_params"]["weights"] == "uniform"

    split = cfg.sampling.plan["split"]
    labeling = cfg.sampling.plan["labeling"]
    assert split["val_fraction"] == 0.0
    assert split["stratify"] is False
    assert labeling["mode"] == "count"
    assert labeling["value"] == 40
    assert labeling["strategy"] == "random"

    test_count = round(435 * split["test_fraction"])
    unlabeled_count = 435 - test_count - labeling["value"]
    assert (labeling["value"], unlabeled_count, test_count) == (40, 200, 195)


def test_explicit_classic_paper_campaign_has_honest_fidelity_statuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = REPO_ROOT / "tools" / "hpc" / "specs" / "tri-dcl-paper.example.yaml"
    monkeypatch.setattr(
        generate_module,
        "collect_runtime_versions",
        lambda **kwargs: {
            "git_sha": "REPLACE_WITH_CLEAN_COMMIT",
            "git_dirty": False,
            "git_diff_sha256": "0" * 64,
        },
    )

    generated = generate_campaign(
        spec,
        repo_root=REPO_ROOT,
        output_dir=tmp_path / "paper",
        _allow_template_placeholders=True,
    )
    meta, tasks = load_manifest(Path(generated.manifest_path))

    assert generated.task_count == 23
    assert meta["counts_by_method"] == {
        "democratic_co_learning": 20,
        "tri_training": 3,
    }
    assert {task.assigned_site for task in tasks} == {"regional"}
    assert {task.resource_profile for task in tasks} == {"cpu_tabular"}
    assert {task.method_profile for task in tasks} == {
        "paper:zhou-goldman-2004-adult-table3",
        "paper:zhou-li-2005-wdbc-table3-j48",
    }
    assert {task.method_id: task.fidelity_status for task in tasks} == {
        "democratic_co_learning": "not_claimable",
        "tri_training": "paper_approx",
    }
    assert {task.method_id: task.dataset_id for task in tasks} == {
        "democratic_co_learning": "adult",
        "tri_training": "wdbc",
    }
    assert all(task.fidelity_status != "paper_matched" for task in tasks)
    dcl_tasks = [task for task in tasks if task.method_id == "democratic_co_learning"]
    assert len({task.dataset_request_sha256 for task in dcl_tasks}) == 1
    assert len({task.expected_dataset_fingerprint for task in dcl_tasks}) == 1
    assert len({task.split_request_sha256 for task in dcl_tasks}) == 20
    assert len({task.expected_split_fingerprint for task in dcl_tasks}) == 20

    site = yaml.safe_load(
        (REPO_ROOT / "tools" / "hpc" / "config" / "profiles" / "regional.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    directives = site["profiles"]["cpu_tabular"]["directives"]
    assert "gres" not in directives
    assert directives["cpus-per-task"] == 4
