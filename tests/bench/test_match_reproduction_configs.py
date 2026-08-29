from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from bench import main as bench_main
from bench.schema import BenchConfigError, ExperimentConfig
from modssc.inductive.registry import get_method_info
from modssc.sampling.plan import SamplingPlan

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTIONS_ROOT = REPO_ROOT / "bench" / "configs" / "reproductions"
DIAGNOSTICS_ROOT = REPO_ROOT / "bench" / "configs" / "diagnostics" / "paper_canaries"

CASES = (
    (
        "fixmatch/cifar10-250.yaml",
        "fixmatch",
        "paper:sohn2020-cifar10-table2-250",
        [1, 2, 3, 4, 5],
        25,
        "google_fixmatch",
    ),
    (
        "flexmatch/cifar10-250.yaml",
        "flexmatch",
        "paper:zhang2021-cifar10-table1-250",
        [0, 1, 2],
        25,
        "torchssl",
    ),
    (
        "free_match/cifar10-40.yaml",
        "free_match",
        "paper:wang2023-cifar10-table1-40",
        [0, 1, 2],
        4,
        "torchssl",
    ),
    (
        "softmatch/cifar10-250.yaml",
        "softmatch",
        "paper:chen2023-cifar10-table2-250",
        [0, 1, 2],
        25,
        "torchssl",
    ),
)


def _load(relative: str) -> tuple[Path, dict[str, Any], ExperimentConfig]:
    path = REPRODUCTIONS_ROOT / relative
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return path, raw, ExperimentConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("relative", "method_id", "profile", "seeds", "per_class", "reference"),
    CASES,
)
def test_match_cards_use_native_sampling_and_checkpoint_contracts(
    relative: str,
    method_id: str,
    profile: str,
    seeds: list[int],
    per_class: int,
    reference: str,
) -> None:
    path, _, cfg = _load(relative)

    assert "Fidelity ceiling: paper_matched" in path.read_text(encoding="utf-8")
    assert cfg.run.benchmark_mode is True
    assert cfg.run.seeds == seeds
    assert cfg.run.seeded_sections == ["sampling", "preprocess", "augmentation"]
    assert cfg.run.resume_policy == "auto"
    assert cfg.run.checkpoint_dir is None
    assert cfg.dataset.id == "cifar10"
    assert cfg.dataset.download is False
    assert cfg.dataset.integrity is not None
    assert cfg.dataset.integrity.fingerprint == (
        "46bfdd98e26ed954f611b41565bc64a3ba5b5497773f2f92a7ebeaa8ff465a58"
    )
    assert cfg.dataset.integrity.content_sha256 == (
        "2ab0d29a8fa44ec94f2d282630b74d31a4fc9986c706173dada6e3a52d75dcfb"
    )

    plan = SamplingPlan.from_dict(cfg.sampling.plan)
    assert plan.component_seeds.resolve(seeds[0])["labeling"] == seeds[0]
    assert plan.partition.ordered_indices_artifact is None
    assert plan.labeling.fixed_indices is None
    assert plan.labeling.fixed_indices_artifact is None
    assert plan.split.kind == "holdout"
    assert plan.split.test_fraction == pytest.approx(0.0)
    assert plan.split.val_fraction == pytest.approx(0.0)
    assert plan.split.stratify is False
    assert plan.split.shuffle is False
    assert plan.labeling.mode == "per_class"
    assert plan.labeling.value == per_class
    assert plan.labeling.strategy == "balanced"
    assert plan.labeling.min_per_class == per_class
    assert plan.labeling.per_class is True
    assert plan.labeling.selection_order == "permutation"
    assert plan.labeling.rng_backend == "legacy_random_state"
    assert plan.labeling.unlabeled_pool == "includes_labeled"

    if reference == "google_fixmatch":
        assert plan.partition.ordering == "class_balanced_stream"
        assert plan.partition.shuffle is False
        assert plan.split.val_size == 1
        assert plan.split.holdout_from == "start"
        assert plan.labeling.selection_scope == "partition"
    else:
        assert plan.partition.ordering == "canonical"
        assert plan.split.val_size is None
        assert plan.labeling.selection_scope == "train"

    assert cfg.method.method_id == method_id
    assert cfg.method.profile == profile
    assert cfg.method.device.device == "cuda"
    assert cfg.method.params["reference_implementation"] == reference
    assert cfg.method.params["batch_size"] == 64
    assert cfg.method.params["mu"] == 7
    assert cfg.method.params["max_steps"] == 1 << 20
    assert cfg.method.params["training_mode"] == "fixed_steps"
    assert cfg.method.params["allow_short_run"] is False
    assert get_method_info(method_id).capabilities.supports_checkpointing


@pytest.mark.parametrize(
    ("relative", "method_id", "reference"),
    [(case[0], case[1], case[5]) for case in CASES],
)
def test_match_cards_keep_the_reference_augmentation_and_reporting_contract(
    relative: str,
    method_id: str,
    reference: str,
) -> None:
    _, raw, cfg = _load(relative)

    assert cfg.augmentation is not None
    assert cfg.augmentation.mode == "online"
    assert cfg.augmentation.modality == "vision"
    assert cfg.augmentation.online_augmenter_id == "vision.cifar_reference"
    assert cfg.method.model is not None
    model = cfg.method.model.classifier_params
    assert cfg.method.model.classifier_id == "wide_resnet_cifar"
    assert cfg.method.model.classifier_backend == "torch"
    assert model["depth"] == 28
    assert model["widen_factor"] == 2
    assert model["optimizer"] == "sgd"
    assert model["scheduler"] == "cosine"
    assert model["max_steps"] == 1 << 20
    assert model["ema_decay"] == pytest.approx(0.999)
    assert cfg.method.model.ema is True

    strong_steps = cfg.augmentation.strong["steps"]
    if reference == "google_fixmatch":
        assert cfg.augmentation.online_augmenter_params == {"profile": "google_fixmatch_ra"}
        assert [step["id"] for step in strong_steps] == [
            "vision.random_horizontal_flip",
            "vision.random_crop_pad",
            "vision.randaugment",
            "vision.cutout",
        ]
        assert cfg.method.params["sampler_mode"] == "shuffle_repeat"
        assert cfg.method.params["interleave_bn"] is True
        assert cfg.method.params["evaluation_interval_steps"] == 1024
        assert cfg.method.params["checkpoint_interval_steps"] == 1024
        assert cfg.method.params["reporting_policy"] == "median_last_checkpoints"
        assert cfg.method.params["reporting_window_checkpoints"] == 20
        normalize = next(
            step for step in raw["preprocess"]["plan"]["steps"] if step["id"] == "vision.normalize"
        )
        assert normalize["params"] == {
            "mean": [127.5, 127.5, 127.5],
            "std": [127.5, 127.5, 127.5],
        }
    else:
        assert cfg.augmentation.online_augmenter_params == {"profile": "torchssl_ra"}
        assert [step["id"] for step in strong_steps] == [
            "vision.randaugment",
            "vision.random_horizontal_flip",
            "vision.random_crop_pad",
        ]
        assert cfg.method.params["sampler_mode"] == "replacement"
        assert cfg.method.params["interleave_bn"] is False
        assert cfg.method.params["evaluation_interval_steps"] == 5000
        assert cfg.method.params["evaluation_tail_interval_steps"] == 1000
        assert cfg.method.params["evaluation_tail_start_fraction"] == pytest.approx(0.8)
        assert cfg.method.params["checkpoint_interval_steps"] == 5000
        assert cfg.method.params["reporting_policy"] == "best_historical_checkpoint"
        assert cfg.method.params["reporting_window_checkpoints"] == 20

    assert cfg.evaluation.metrics == ["accuracy"]
    if reference == "google_fixmatch":
        assert cfg.evaluation.split_for_model_selection == "val"
        assert cfg.evaluation.test_selection_policy == "forbid"
    else:
        assert cfg.evaluation.split_for_model_selection == "test"
        assert cfg.evaluation.test_selection_policy == "paper_protocol"
    assert set(raw["evaluation"]).isdisjoint(
        {
            "evaluation_interval_steps",
            "checkpoint_policy",
            "reporting_policy",
            "reporting_window_checkpoints",
        }
    )
    assert method_id in cfg.run.name


def test_match_diagnostic_cards_share_the_production_native_sampling_plan() -> None:
    production_by_method = {
        method_id: _load(relative)[1]["sampling"]["plan"] for relative, method_id, *_ in CASES
    }
    diagnostic_paths = sorted(
        path
        for method_id in production_by_method
        for path in (DIAGNOSTICS_ROOT / method_id).glob("*.yaml")
    )
    assert len(diagnostic_paths) == 4

    for path in diagnostic_paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = ExperimentConfig.from_dict(raw)
        assert cfg.run.resume_policy == "auto"
        assert cfg.sampling.plan == production_by_method[cfg.method.method_id]
        plan = SamplingPlan.from_dict(cfg.sampling.plan)
        assert plan.partition.ordered_indices_artifact is None
        assert plan.labeling.fixed_indices_artifact is None
        if cfg.method.method_id == "fixmatch":
            assert cfg.evaluation.test_selection_policy == "forbid"
        else:
            assert cfg.evaluation.test_selection_policy == "paper_protocol"


@pytest.mark.parametrize(
    "relative",
    [
        "flexmatch/cifar10-250.yaml",
        "free_match/cifar10-40.yaml",
        "softmatch/cifar10-250.yaml",
    ],
)
def test_declared_paper_test_selection_passes_benchmark_preflight(relative: str) -> None:
    _, production_raw, production_cfg = _load(relative)
    method_id = production_cfg.method.method_id
    diagnostic_path = next((DIAGNOSTICS_ROOT / method_id).glob("*.yaml"))
    diagnostic_raw = yaml.safe_load(diagnostic_path.read_text(encoding="utf-8"))
    diagnostic_cfg = ExperimentConfig.from_dict(diagnostic_raw)

    for raw, cfg in (
        (production_raw, production_cfg),
        (diagnostic_raw, diagnostic_cfg),
    ):
        bench_main._benchmark_contract_preflight(
            cfg=cfg,
            raw=raw,
            preprocess_steps=[step["id"] for step in raw["preprocess"]["plan"]["steps"]],
            view_preprocess_steps=[],
        )


def test_free_match_card_pins_pre_registered_entropy_weight() -> None:
    _, _, cfg = _load("free_match/cifar10-40.yaml")

    assert cfg.method.params["lambda_e"] == pytest.approx(0.05)
    assert cfg.method.params["use_quantile"] is False


def test_softmatch_card_pins_result_producing_torchssl_alignment() -> None:
    _, _, cfg = _load("softmatch/cifar10-250.yaml")

    assert cfg.method.params["dist_align"] is True
    assert cfg.method.params["dist_uniform"] is False


def test_method_profile_is_opaque_to_the_generic_bench_schema() -> None:
    _, raw, _ = _load("fixmatch/cifar10-250.yaml")
    raw["method"]["profile"] = "standardized"

    cfg = ExperimentConfig.from_dict(raw)

    assert cfg.method.profile == "standardized"
    assert SamplingPlan.from_dict(cfg.sampling.plan).partition.ordering == ("class_balanced_stream")


def test_generic_online_augmenter_contract_checks_structure_not_paper_identity() -> None:
    _, raw, _ = _load("fixmatch/cifar10-250.yaml")
    raw.pop("acceptance")
    raw["method"]["id"] = "flexmatch"

    cfg = ExperimentConfig.from_dict(raw)
    assert cfg.method.method_id == "flexmatch"
    assert cfg.augmentation is not None
    assert cfg.augmentation.online_augmenter_params == {"profile": "google_fixmatch_ra"}

    _, raw, _ = _load("fixmatch/cifar10-250.yaml")
    raw["augmentation"].pop("online_augmenter_id")
    with pytest.raises(BenchConfigError, match="require online_augmenter_id"):
        ExperimentConfig.from_dict(raw)

    _, raw, _ = _load("fixmatch/cifar10-250.yaml")
    raw["augmentation"]["mode"] = "fixed"
    with pytest.raises(BenchConfigError, match="requires augmentation.mode='online'"):
        ExperimentConfig.from_dict(raw)
