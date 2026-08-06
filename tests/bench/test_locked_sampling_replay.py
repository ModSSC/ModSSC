from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bench.main as bench_main
from bench.campaign.dcl_partition_lock import (
    build_task_partition_selection,
    load_dcl_partition_selection,
)
from bench.schema import BenchConfigError, ExperimentConfig
from bench.seed_sweep import apply_global_seed
from bench.utils.io import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_DIR = REPO_ROOT / "bench/campaigns/locks/dcl-vote-zhou-goldman-2004-v1"
LOCK_SHA256 = "5f586b2ab21bd6c2b0e058ab9d588ec1fc04b41b7d93e5a125d0a5f2ea1b36fb"
DATASET_FINGERPRINT = "98f2cf80ea8e8fb8f3f546dc87d3a231a0ec10fe6d26b5dfe490fc832079b0dd"
DATASET_CONTENT_SHA256 = "5b95c771651aa62b985332026f63b423d7fe7dff2f0bc90ef2c336d4d2b70130"


def test_locked_sampling_replay_loads_selected_arrays_and_copies_exact_bytes(
    tmp_path: Path,
) -> None:
    lock = load_dcl_partition_selection(
        LOCK_DIR / "selected-partitions.json",
        expected_sha256=LOCK_SHA256,
        expected_dataset_fingerprint=DATASET_FINGERPRINT,
    )
    entry = lock.selected[0]
    source_replay = LOCK_DIR / "splits/seed-001"
    evidence = build_task_partition_selection(
        selection_path=str(lock.path),
        lock=lock,
        entry=entry,
        replay_path=str(source_replay),
    )
    raw = load_yaml(REPO_ROOT / "bench/configs/reproductions/democratic_co_learning/vote.yaml")
    effective = apply_global_seed(
        raw,
        seed=entry.seed,
        seeded_sections=raw["run"]["seeded_sections"],
    )
    effective["sampling"]["replay"] = evidence
    cfg = ExperimentConfig.from_dict(effective)
    dataset = SimpleNamespace(
        train=SimpleNamespace(y=np.zeros(435, dtype=np.int64)),
        test=None,
        meta={
            "dataset_fingerprint": DATASET_FINGERPRINT,
            "dataset_content_sha256": DATASET_CONTENT_SHA256,
        },
    )

    sampling, verified = bench_main._load_locked_sampling_replay(
        dataset=dataset,
        cfg=cfg,
        sampling_seed=entry.seed,
    )

    assert verified.entry == entry
    assert sampling.split_fingerprint == entry.split_fingerprint
    assert sampling.train_idx.size == 240
    assert sampling.labeled_idx.size == 40
    assert sampling.unlabeled_idx.size == 200
    assert sampling.test_idx.size == 195

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = bench_main._persist_sampling_replay(
        SimpleNamespace(run_dir=run_dir),
        sampling,
        source_dir=verified.replay_dir,
    )
    copied = run_dir / str(artifact["path"])
    for name in ("MANIFEST.json", "split.json", "arrays.npz"):
        assert (copied / name).read_bytes() == (source_replay / name).read_bytes()
    assert artifact["manifest_sha256"] == entry.split_manifest_sha256


def test_locked_sampling_replay_rejects_dataset_content_drift() -> None:
    lock = load_dcl_partition_selection(
        LOCK_DIR / "selected-partitions.json",
        expected_sha256=LOCK_SHA256,
        expected_dataset_fingerprint=DATASET_FINGERPRINT,
    )
    entry = lock.selected[0]
    evidence = build_task_partition_selection(
        selection_path=str(lock.path),
        lock=lock,
        entry=entry,
        replay_path=str(LOCK_DIR / "splits/seed-001"),
    )
    raw = load_yaml(REPO_ROOT / "bench/configs/reproductions/democratic_co_learning/vote.yaml")
    effective = apply_global_seed(
        raw,
        seed=entry.seed,
        seeded_sections=raw["run"]["seeded_sections"],
    )
    effective["sampling"]["replay"] = evidence
    cfg = ExperimentConfig.from_dict(effective)
    dataset = SimpleNamespace(
        train=SimpleNamespace(y=np.zeros(435, dtype=np.int64)),
        test=None,
        meta={
            "dataset_fingerprint": DATASET_FINGERPRINT,
            "dataset_content_sha256": "0" * 64,
        },
    )

    with pytest.raises(BenchConfigError, match="content digest differs"):
        bench_main._load_locked_sampling_replay(
            dataset=dataset,
            cfg=cfg,
            sampling_seed=entry.seed,
        )


def test_locked_sampling_replay_accepts_the_registered_diagnostic_profile() -> None:
    lock = load_dcl_partition_selection(
        LOCK_DIR / "selected-partitions.json",
        expected_sha256=LOCK_SHA256,
        expected_dataset_fingerprint=DATASET_FINGERPRINT,
    )
    entry = lock.selected[0]
    evidence = build_task_partition_selection(
        selection_path=str(lock.path),
        lock=lock,
        entry=entry,
        replay_path=str(LOCK_DIR / "splits/seed-001"),
    )
    raw = load_yaml(
        REPO_ROOT
        / "bench/configs/diagnostics/democratic_co_learning/vote_control_naive_bayes_v2.yaml"
    )
    effective = apply_global_seed(
        raw,
        seed=entry.seed,
        seeded_sections=raw["run"]["seeded_sections"],
    )
    effective["sampling"]["replay"] = evidence
    cfg = ExperimentConfig.from_dict(effective)
    dataset = SimpleNamespace(
        train=SimpleNamespace(y=np.zeros(435, dtype=np.int64)),
        test=None,
        meta={
            "dataset_fingerprint": DATASET_FINGERPRINT,
            "dataset_content_sha256": DATASET_CONTENT_SHA256,
        },
    )

    sampling, verified = bench_main._load_locked_sampling_replay(
        dataset=dataset,
        cfg=cfg,
        sampling_seed=entry.seed,
    )

    assert verified.entry == entry
    assert sampling.split_fingerprint == entry.split_fingerprint
