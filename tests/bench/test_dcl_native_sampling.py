from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.orchestrators import dataset as dataset_orch
from bench.orchestrators import sampling as sampling_orch
from bench.schema import BenchConfigError, ExperimentConfig
from bench.seed_sweep import apply_global_seed
from bench.utils.io import load_yaml
from modssc.data_loader.types import LoadedDataset, Split
from modssc.sampling.plan import SamplingPlan

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "bench/configs/reproductions/democratic_co_learning/vote.yaml"
DATASET_FINGERPRINT = "98f2cf80ea8e8fb8f3f546dc87d3a231a0ec10fe6d26b5dfe490fc832079b0dd"
DATASET_CONTENT_SHA256 = "5b95c771651aa62b985332026f63b423d7fe7dff2f0bc90ef2c336d4d2b70130"


def _dataset(*, content_sha256: str = DATASET_CONTENT_SHA256) -> LoadedDataset:
    return LoadedDataset(
        train=Split(
            X=np.zeros((435, 1), dtype=np.float32),
            y=np.arange(435, dtype=np.int64) % 2,
        ),
        test=None,
        meta={
            "dataset_fingerprint": DATASET_FINGERPRINT,
            "dataset_content_sha256": content_sha256,
        },
    )


def test_dcl_card_declares_a_native_statistical_protocol() -> None:
    raw = load_yaml(CONFIG_PATH)
    cfg = ExperimentConfig.from_dict(raw)
    text = CONFIG_PATH.read_text(encoding="utf-8")

    assert "statistical replication" in text
    assert cfg.run.seeds == list(range(1, 21))
    assert cfg.run.seeded_sections == ["sampling", "preprocess"]
    assert cfg.dataset.integrity is not None
    assert cfg.dataset.integrity.fingerprint == DATASET_FINGERPRINT
    assert cfg.dataset.integrity.content_sha256 == DATASET_CONTENT_SHA256

    plan = SamplingPlan.from_dict(cfg.sampling.plan)
    assert plan.partition.ordered_indices_artifact is None
    assert plan.labeling.fixed_indices_artifact is None
    assert plan.split.kind == "holdout"
    assert plan.split.test_fraction == pytest.approx(195 / 435)
    assert plan.split.val_fraction == pytest.approx(0.0)
    assert plan.split.stratify is False
    assert plan.split.shuffle is True
    assert plan.labeling.mode == "count"
    assert plan.labeling.value == 40
    assert plan.labeling.strategy == "random"
    assert plan.labeling.unlabeled_pool == "complement"


def test_dcl_native_partitions_are_deterministic_and_have_paper_sizes() -> None:
    raw = load_yaml(CONFIG_PATH)
    split_fingerprints: set[str] = set()
    test_partitions: set[tuple[int, ...]] = set()

    for seed in raw["run"]["seeds"]:
        effective = apply_global_seed(
            raw,
            seed=seed,
            seeded_sections=raw["run"]["seeded_sections"],
        )
        first = sampling_orch.run(
            _dataset(),
            plan_dict=effective["sampling"]["plan"],
            seed=effective["sampling"]["seed"],
            dataset_id="vote",
        )
        second = sampling_orch.run(
            _dataset(),
            plan_dict=effective["sampling"]["plan"],
            seed=effective["sampling"]["seed"],
            dataset_id="vote",
        )

        for name in ("train", "val", "test", "train_labeled", "train_unlabeled"):
            np.testing.assert_array_equal(first.indices[name], second.indices[name])
        assert first.train_idx.size == 240
        assert first.val_idx.size == 0
        assert first.test_idx.size == 195
        assert first.labeled_idx.size == 40
        assert first.unlabeled_idx.size == 200
        assert np.intersect1d(first.train_idx, first.test_idx).size == 0
        assert np.intersect1d(first.labeled_idx, first.unlabeled_idx).size == 0
        np.testing.assert_array_equal(
            np.sort(np.concatenate((first.train_idx, first.test_idx))),
            np.arange(435),
        )
        np.testing.assert_array_equal(
            np.sort(np.concatenate((first.labeled_idx, first.unlabeled_idx))),
            np.sort(first.train_idx),
        )
        split_fingerprints.add(first.split_fingerprint)
        test_partitions.add(tuple(int(index) for index in first.test_idx))

    assert len(split_fingerprints) == 20
    assert len(test_partitions) == 20


def test_dcl_dataset_integrity_is_yaml_declared_and_fails_closed() -> None:
    cfg = ExperimentConfig.from_dict(load_yaml(CONFIG_PATH))

    dataset_orch.verify_integrity(_dataset(), cfg.dataset)
    with pytest.raises(BenchConfigError, match="dataset.integrity.content_sha256 differs"):
        dataset_orch.verify_integrity(_dataset(content_sha256="0" * 64), cfg.dataset)
