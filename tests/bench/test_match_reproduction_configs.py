from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from bench.schema import BenchConfigError, ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTIONS_ROOT = REPO_ROOT / "bench" / "configs" / "reproductions"

CASES = (
    (
        "fixmatch/cifar10-250.yaml",
        "fixmatch",
        "paper:sohn2020-cifar10-table2-250",
        [1, 2, 3, 4, 5],
        25,
        "fixmatch-google-cifar10-250-seeds1-5.npz",
        49_999,
    ),
    (
        "flexmatch/cifar10-250.yaml",
        "flexmatch",
        "paper:zhang2021-cifar10-table1-250",
        [0, 1, 2],
        25,
        "torchssl-cifar10-250-seeds0-2.npz",
        50_000,
    ),
    (
        "free_match/cifar10-40.yaml",
        "free_match",
        "paper:wang2023-cifar10-table1-40",
        [0, 1, 2],
        4,
        "torchssl-cifar10-40-seeds0-2.npz",
        50_000,
    ),
    (
        "softmatch/cifar10-250.yaml",
        "softmatch",
        "paper:chen2023-cifar10-table2-250",
        [0, 1, 2],
        25,
        "torchssl-cifar10-250-seeds0-2.npz",
        50_000,
    ),
)


def test_match_reference_manifest_pins_sources_licenses_and_pixel_fixtures() -> None:
    root = REPO_ROOT / "provenance" / "article10" / "match_audit"
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    repositories = {entry["id"]: entry for entry in manifest["repositories"]}

    assert {source: entry["commit"] for source, entry in repositories.items()} == {
        "google_fixmatch": "d4985a158065947dba803e626ee9a6721709c570",
        "torchssl": "03193a1b7883727db1ce9c092e083091e18aedbb",
        "usb": "1ef4cbebcc0b368158315aeb425053858cf6c845",
    }
    assert {
        "fixmatch.py",
        "libml/data.py",
    }.issubset(repositories["google_fixmatch"]["audited_files"])
    assert "models/flexmatch/flexmatch.py" in repositories["torchssl"]["audited_files"]
    assert {
        "models/flexmatch/flexmatch_utils.py",
        "models/freematch_entropy/freematch.py",
        "models/freematch_entropy/freematch_utils.py",
        "models/softmatch/softmatch_utils.py",
    }.issubset(repositories["torchssl"]["audited_files"])
    assert "semilearn/algorithms/softmatch/utils.py" in repositories["usb"]["audited_files"]
    assert not (root / "sources").exists()
    for entry in repositories.values():
        assert entry["source_distribution"] == "not_vendored"
        for upstream_path, expected_sha256 in entry["audited_files"].items():
            assert isinstance(upstream_path, str) and upstream_path
            assert len(expected_sha256) == 64
            int(expected_sha256, 16)
        license_record = entry["license"]
        license_path = root / license_record["local_copy"]
        assert (
            hashlib.sha256(license_path.read_bytes()).hexdigest() == license_record["local_sha256"]
        )
        assert license_record["local_sha256"] == license_record["upstream_sha256"]
    for local_path, expected_sha256 in manifest["local_artifacts"].items():
        artifact_path = root / local_path
        assert artifact_path.is_file()
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == expected_sha256

    fixtures = json.loads((root / "PIXEL_FIXTURES.json").read_text(encoding="utf-8"))
    assert fixtures["schema_version"] == 1
    assert {source: entry["commit"] for source, entry in fixtures["stacks"].items()} == {
        "google_fixmatch": "d4985a158065947dba803e626ee9a6721709c570",
        "torchssl": "03193a1b7883727db1ce9c092e083091e18aedbb",
    }


def _load(relative: str) -> tuple[Path, dict[str, Any]]:
    path = REPRODUCTIONS_ROOT / relative
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return path, raw


@pytest.mark.parametrize(
    (
        "relative",
        "method_id",
        "profile",
        "seeds",
        "per_class",
        "artifact_name",
        "train_size",
    ),
    CASES,
)
def test_match_paper_cards_are_explicit_and_share_the_training_stack(
    relative: str,
    method_id: str,
    profile: str,
    seeds: list[int],
    per_class: int,
    artifact_name: str,
    train_size: int,
) -> None:
    path, raw = _load(relative)
    cfg = ExperimentConfig.from_dict(raw)

    assert "Fidelity ceiling: paper_matched" in path.read_text(encoding="utf-8")
    assert cfg.run.benchmark_mode is True
    assert cfg.run.seeds == seeds
    assert cfg.run.seeded_sections == ["sampling", "preprocess", "augmentation"]
    assert cfg.dataset.id == "cifar10"
    assert cfg.dataset.download is False
    assert cfg.method.method_id == method_id
    assert cfg.method.profile == profile
    assert cfg.method.device.device == "cuda"

    labeling = cfg.sampling.plan["labeling"]
    assert labeling["mode"] == "per_class"
    assert labeling["value"] == per_class
    assert labeling["strategy"] == "balanced"
    assert cfg.sampling.plan["split"]["val_fraction"] == pytest.approx(0.0)
    assert cfg.sampling.plan["policy"]["respect_official_test"] is True
    partition_artifact = cfg.sampling.plan["partition"]["ordered_indices_artifact"]
    artifact_path = Path(partition_artifact["path"])
    assert artifact_path.name == artifact_name
    assert artifact_path.is_file()
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == partition_artifact["sha256"]
    assert partition_artifact["unlabeled_pool"] == "includes_labeled"
    assert partition_artifact["test_ref"] == "test"
    assert partition_artifact["expected_train_size"] == train_size
    assert partition_artifact["expected_labeled_size"] == per_class * 10
    assert partition_artifact["expected_unlabeled_size"] == train_size
    assert partition_artifact["expected_per_class"] == per_class
    if method_id != "fixmatch":
        with np.load(artifact_path, allow_pickle=False) as archive:
            for run_seed in seeds:
                unlabeled = archive[f"seed_{run_seed}__train_unlabeled"]
                assert np.array_equal(unlabeled, np.arange(50_000))
    with np.load(artifact_path, allow_pickle=False) as archive:
        metadata = json.loads(np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes())
        assert metadata["seeds"] == seeds
        assert metadata["dataset_fingerprint"] == (
            "46bfdd98e26ed954f611b41565bc64a3ba5b5497773f2f92a7ebeaa8ff465a58"
        )
        assert metadata["canonical_label_source_sha256"] == (
            "9dfee6f275bac0f14e63de8d1091cd1f4487a16d30c6d8726f61d1b8f999c745"
        )
        for seed in seeds:
            prefix = f"seed_{seed}__"
            train = archive[f"{prefix}train"]
            val = archive[f"{prefix}val"]
            test = archive[f"{prefix}test"]
            labeled = archive[f"{prefix}train_labeled"]
            unlabeled = archive[f"{prefix}train_unlabeled"]
            assert np.array_equal(train, unlabeled)
            assert np.setdiff1d(labeled, unlabeled).size == 0
            assert np.intersect1d(train, val).size == 0
            assert np.array_equal(
                np.sort(np.concatenate([train, val])),
                np.arange(50_000),
            )
            assert np.array_equal(test, np.arange(10_000))

    assert cfg.augmentation is not None
    assert cfg.augmentation.mode == "online"
    assert cfg.augmentation.modality == "vision"
    assert [step["id"] for step in cfg.augmentation.weak["steps"]] == [
        "vision.random_horizontal_flip",
        "vision.random_crop_pad",
    ]
    strong_steps = cfg.augmentation.strong["steps"]
    if method_id == "fixmatch":
        assert cfg.augmentation.reference_implementation == "google_fixmatch_ra"
        assert [step["id"] for step in strong_steps] == [
            "vision.random_horizontal_flip",
            "vision.random_crop_pad",
            "vision.randaugment",
            "vision.cutout",
        ]
        assert strong_steps[2]["params"] == {
            "num_ops": 2,
            "magnitude": 10,
            "num_magnitude_bins": 31,
        }
        assert strong_steps[3]["params"] == {
            "length": 16,
            "n_holes": 1,
            "fill": 0.0,
        }
        assert cfg.augmentation.reference_policy["randaugment"] == {
            "num_ops": 2,
            "configured_magnitude": 10,
            "magnitude_sampling": "integer_uniform_[1,10)",
        }
        assert cfg.augmentation.reference_policy["cutout"] == {
            "size_pixels": 16,
            "fill": 0,
        }
    else:
        assert cfg.augmentation.reference_implementation == "torchssl_ra"
        assert [step["id"] for step in strong_steps] == [
            "vision.randaugment",
            "vision.random_horizontal_flip",
            "vision.random_crop_pad",
        ]
        assert strong_steps[0]["params"] == {
            "num_ops": 3,
            "magnitude": 5,
            "num_magnitude_bins": 31,
        }
        assert cfg.augmentation.reference_policy["randaugment"] == {
            "num_ops": 3,
            "configured_magnitude": 5,
            "magnitude_sampling": "per_operation_uniform_full_range",
        }
        assert cfg.augmentation.reference_policy["cutout"] == {
            "size_fraction_sampling": "uniform_[0,0.5)",
            "fill_rgb": [125, 123, 114],
        }

    assert cfg.method.params["batch_size"] == 64
    assert cfg.method.params["mu"] == 7
    assert cfg.method.params["batch_size"] * cfg.method.params["mu"] == 448
    assert cfg.method.params["max_steps"] == 1 << 20
    assert cfg.method.params["training_mode"] == "fixed_steps"
    assert cfg.method.params["sampler_shuffle_buffer"] == 8192
    assert cfg.method.params["allow_short_run"] is False
    assert cfg.method.model is not None
    assert cfg.method.model.classifier_id == "wide_resnet_cifar"
    model = cfg.method.model.classifier_params
    if method_id == "fixmatch":
        assert cfg.method.params["reference_implementation"] == "google_fixmatch"
        assert cfg.method.params["sampler_mode"] == "shuffle_repeat"
        assert cfg.method.params["augmentation_profile"] == "google_fixmatch_ra"
        assert cfg.method.params["interleave_bn"] is True
        assert cfg.method.params["reporting_policy"] == "median_last_checkpoints"
        assert model["reference_implementation"] == "google_fixmatch"
        normalize = next(
            step for step in raw["preprocess"]["plan"]["steps"] if step["id"] == "vision.normalize"
        )
        assert normalize["params"] == {
            "mean": [127.5, 127.5, 127.5],
            "std": [127.5, 127.5, 127.5],
        }
        assert "input_mean" not in model
        assert "input_std" not in model
    else:
        assert cfg.method.params["reference_implementation"] == "torchssl"
        assert cfg.method.params["sampler_mode"] == "replacement"
        assert cfg.method.params["augmentation_profile"] == "torchssl_ra"
        assert cfg.method.params["interleave_bn"] is False
        assert cfg.method.params["reporting_policy"] == "best_historical_checkpoint"
        assert model["reference_implementation"] == "torchssl"
    assert model["depth"] == 28
    assert model["widen_factor"] == 2
    assert model["optimizer"] == "sgd"
    assert model["nesterov"] is True
    assert model["scheduler"] == "cosine"
    assert model["max_steps"] == 1 << 20
    assert model["ema_decay"] == pytest.approx(0.999)
    assert cfg.method.model.ema is True
    if method_id == "fixmatch":
        assert cfg.evaluation.evaluation_interval_steps == 1024
        assert cfg.evaluation.split_for_model_selection == "val"
        assert cfg.evaluation.checkpoint_policy == "periodic_keep_last_50"
        assert cfg.evaluation.reporting_policy == "median_last_20_test_checkpoints"
        assert cfg.evaluation.reporting_window_checkpoints == 20
        assert cfg.evaluation.report_splits == ["val", "test"]
    else:
        assert cfg.evaluation.evaluation_interval_steps == 5000
        assert cfg.evaluation.split_for_model_selection == "test"
        assert cfg.evaluation.checkpoint_policy == "best_evaluation_accuracy"
        assert cfg.evaluation.reporting_policy == "test_at_best_evaluation_checkpoint"
        assert cfg.evaluation.reporting_window_checkpoints is None
        assert cfg.evaluation.report_splits == ["test"]
    assert cfg.evaluation.metrics == ["accuracy"]


def test_free_match_card_pins_pre_registered_entropy_weight() -> None:
    _, raw = _load("free_match/cifar10-40.yaml")
    cfg = ExperimentConfig.from_dict(raw)

    assert cfg.method.params["lambda_e"] == pytest.approx(0.05)
    assert cfg.method.params["use_quantile"] is False


def test_softmatch_card_pins_result_producing_torchssl_alignment() -> None:
    _, raw = _load("softmatch/cifar10-250.yaml")
    cfg = ExperimentConfig.from_dict(raw)

    assert cfg.method.params["dist_align"] is True
    assert cfg.method.params["dist_uniform"] is False
    assert cfg.evaluation.evaluation_interval_steps == 5000


def test_match_acceptance_requires_auditable_sampler_contracts() -> None:
    registry = yaml.safe_load(
        (REPO_ROOT / "bench" / "campaigns" / "article10-paper-acceptance.yaml").read_text(
            encoding="utf-8"
        )
    )
    protocols = registry["protocols"]
    expected_algorithms = {
        "sohn-2020-cifar10-table2-250": ("tensorflow.data.Dataset.repeat().shuffle(buffer_size)"),
        "zhang-2021-cifar10-table1-250": ("torch.utils.data.RandomSampler(replacement=True)"),
        "wang-2023-cifar10-table1-40": ("torch.utils.data.RandomSampler(replacement=True)"),
        "chen-2023-cifar10-table2-250": ("torch.utils.data.RandomSampler(replacement=True)"),
    }
    for protocol_id, expected_algorithm in expected_algorithms.items():
        diagnostics = {
            item["path"]: item for item in protocols[protocol_id]["required_diagnostics"]
        }
        assert (
            diagnostics["artifacts.method.diagnostics.sampler_contract.reference_algorithm"][
                "value"
            ]
            == expected_algorithm
        )
        assert (
            diagnostics[
                "artifacts.method.diagnostics.sampler_contract.historical_bitstream_claimed"
            ]["value"]
            is False
        )
        assert diagnostics["artifacts.method.diagnostics.batch_size_labeled"]["value"] == 64
        assert diagnostics["artifacts.method.diagnostics.batch_size_unlabeled"]["value"] == 448
        assert diagnostics[
            "artifacts.method.diagnostics.checkpoint_policy.evaluation_interval_steps"
        ]["value"] == (1024 if protocol_id.startswith("sohn-") else 5000)

    free_match = {
        item["path"]: item
        for item in protocols["wang-2023-cifar10-table1-40"]["required_diagnostics"]
    }
    assert free_match["artifacts.method.diagnostics.lambda_e"]["value"] == pytest.approx(0.05)


def test_inclusive_unlabeled_pool_is_restricted_to_paper_profiles() -> None:
    _, raw = _load("fixmatch/cifar10-250.yaml")
    raw["method"]["profile"] = "standardized"

    with pytest.raises(BenchConfigError, match="restricted to paper profiles"):
        ExperimentConfig.from_dict(raw)


def test_match_paper_card_rejects_reference_augmentation_contradictions() -> None:
    _, raw = _load("fixmatch/cifar10-250.yaml")

    raw["method"]["id"] = "flexmatch"
    with pytest.raises(BenchConfigError, match="requires method.id"):
        ExperimentConfig.from_dict(raw)

    _, raw = _load("fixmatch/cifar10-250.yaml")
    raw["augmentation"]["enabled"] = False
    with pytest.raises(BenchConfigError, match="requires online vision augmentation"):
        ExperimentConfig.from_dict(raw)

    _, raw = _load("fixmatch/cifar10-250.yaml")
    raw["augmentation"]["reference_implementation"] = "torchssl_ra"
    with pytest.raises(BenchConfigError, match="reference_implementation"):
        ExperimentConfig.from_dict(raw)

    _, raw = _load("fixmatch/cifar10-250.yaml")
    raw["augmentation"]["reference_policy"]["randaugment"]["num_ops"] = 3
    with pytest.raises(BenchConfigError, match="reference_policy contradicts"):
        ExperimentConfig.from_dict(raw)

    _, raw = _load("fixmatch/cifar10-250.yaml")
    raw["augmentation"]["weak"]["steps"] = "not-a-step-list"
    with pytest.raises(BenchConfigError, match="augmentation.weak contradicts"):
        ExperimentConfig.from_dict(raw)

    _, raw = _load("fixmatch/cifar10-250.yaml")
    raw["augmentation"]["strong"]["steps"].pop()
    with pytest.raises(BenchConfigError, match="augmentation.strong contradicts"):
        ExperimentConfig.from_dict(raw)

    _, raw = _load("fixmatch/cifar10-250.yaml")
    raw["augmentation"]["strong"]["steps"][2]["params"]["magnitude"] = 5
    with pytest.raises(BenchConfigError, match="RandAugment parameters contradict"):
        ExperimentConfig.from_dict(raw)

    _, raw = _load("fixmatch/cifar10-250.yaml")
    raw["augmentation"]["strong"]["steps"][3]["params"]["length"] = 8
    with pytest.raises(BenchConfigError, match="Cutout parameters contradict"):
        ExperimentConfig.from_dict(raw)
