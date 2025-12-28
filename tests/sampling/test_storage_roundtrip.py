from __future__ import annotations

import numpy as np

from modssc.sampling.result import SamplingResult
from modssc.sampling.storage import load_split, save_split


def test_storage_roundtrip_indices(tmp_path) -> None:
    res = SamplingResult(
        schema_version=1,
        created_at="now",
        dataset_fingerprint="d",
        split_fingerprint="s",
        plan={"x": 1},
        indices={
            "train": np.array([0, 1]),
            "val": np.array([2]),
            "test": np.array([], dtype=np.int64),
            "train_labeled": np.array([0]),
            "train_unlabeled": np.array([1]),
        },
        refs={
            "train": "train",
            "val": "train",
            "test": "train",
            "train_labeled": "train",
            "train_unlabeled": "train",
        },
        masks={},
        stats={"ok": True},
    )
    p = save_split(res, tmp_path / "split", overwrite=True)
    out = load_split(p)
    assert out.indices["train"].tolist() == [0, 1]
    assert out.stats["ok"] is True


def test_storage_roundtrip_masks(tmp_path) -> None:
    masks = {
        "train": np.array([True, True, False, False, False]),
        "val": np.array([False, False, True, False, False]),
        "test": np.array([False, False, False, True, True]),
        "labeled": np.array([True, False, False, False, False]),
        "unlabeled": np.array([False, True, False, False, False]),
    }
    res = SamplingResult(
        schema_version=1,
        created_at="now",
        dataset_fingerprint="d",
        split_fingerprint="s",
        plan={"x": 1},
        indices={},
        refs={},
        masks=masks,
        stats={},
    )
    p = save_split(res, tmp_path / "splitg", overwrite=True)
    out = load_split(p)
    assert out.masks["train"].dtype == bool
    assert out.masks["test"].sum() == 2
