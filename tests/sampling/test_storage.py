"""Test coverage for sampling/storage.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from modssc.sampling.result import SamplingResult
from modssc.sampling.storage import load_split, save_split


def test_save_split_exists_no_overwrite():
    """Test save_split raises FileExistsError if directory exists and overwrite=False."""

    result = SamplingResult(
        schema_version=1,
        created_at="2023-01-01",
        dataset_fingerprint="ds_fp",
        split_fingerprint="sp_fp",
        plan={"method": "random"},
        refs={"dataset": "ds"},
        stats={"n_train": 10},
        indices={"train": np.arange(10)},
        masks={},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "split"
        out_dir.mkdir()

        with pytest.raises(FileExistsError, match="already exists"):
            save_split(result, out_dir, overwrite=False)

        save_split(result, out_dir, overwrite=True)
        assert (out_dir / "split.json").exists()


def test_load_split_extra_keys():
    """Test load_split ignores extra keys in npz file."""

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "split"
        out_dir.mkdir()

        meta = {
            "schema_version": 1,
            "created_at": "2023-01-01",
            "dataset_fingerprint": "ds_fp",
            "split_fingerprint": "sp_fp",
            "plan": {},
            "refs": {},
            "stats": {},
            "mode": "inductive",
        }
        (out_dir / "split.json").write_text(json.dumps(meta), encoding="utf-8")

        arrays = {
            "idx__train": np.arange(10),
            "mask__val": np.zeros(10, dtype=bool),
            "extra_stuff": np.arange(5),
        }
        np.savez_compressed(out_dir / "arrays.npz", **arrays)

        result = load_split(out_dir)

        assert "train" in result.indices
        assert "val" in result.masks
        assert len(result.indices) == 1
        assert len(result.masks) == 1
