from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bench.schema import BenchConfigError, ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_online_augmenter_params_cannot_shadow_orchestration_seed() -> None:
    path = REPO_ROOT / "bench/configs/reproductions/fixmatch/cifar10-250.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["augmentation"]["online_augmenter_params"]["seed"] = 123

    with pytest.raises(BenchConfigError, match="must not redefine seed"):
        ExperimentConfig.from_dict(raw)


def test_online_augmentation_rejects_two_strong_views_in_the_schema() -> None:
    path = REPO_ROOT / "bench/configs/reproductions/fixmatch/cifar10-250.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["augmentation"]["strong_views"] = 2

    with pytest.raises(BenchConfigError, match="supports exactly one strong view"):
        ExperimentConfig.from_dict(raw)
