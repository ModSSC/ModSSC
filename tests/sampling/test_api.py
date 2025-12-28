"""Test coverage for sampling/api.py utility functions and edge cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modssc.sampling.api import (
    _idx_to_mask,
    _resolve_dataset_fingerprint,
    _warn_on_sampling_stats,
    default_split_cache_dir,
)
from modssc.sampling.errors import MissingDatasetFingerprintError
from modssc.sampling.result import SamplingResult


def test_idx_to_mask_empty_indices():
    """Test _idx_to_mask with empty index array (line 387->389 branch)."""
    n = 10
    idx = np.array([], dtype=np.int64)
    mask = _idx_to_mask(n, idx)
    assert mask.dtype == bool
    assert mask.shape == (n,)
    assert not mask.any()


def test_idx_to_mask_with_indices():
    """Test _idx_to_mask with non-empty indices."""
    n = 10
    idx = np.array([1, 3, 5], dtype=np.int64)
    mask = _idx_to_mask(n, idx)
    expected = np.zeros(n, dtype=bool)
    expected[[1, 3, 5]] = True
    np.testing.assert_array_equal(mask, expected)


def test_default_split_cache_dir_with_env(monkeypatch):
    """Test default_split_cache_dir with MODSSC_SPLIT_CACHE_DIR set."""
    monkeypatch.setenv("MODSSC_SPLIT_CACHE_DIR", "/custom/cache")
    cache_dir = default_split_cache_dir()
    assert cache_dir == Path("/custom/cache").expanduser().resolve()


def test_default_split_cache_dir_no_env(monkeypatch):
    """Test default_split_cache_dir without environment variable."""
    monkeypatch.delenv("MODSSC_SPLIT_CACHE_DIR", raising=False)
    cache_dir = default_split_cache_dir()

    assert "modssc" in str(cache_dir).lower()
    assert "splits" in str(cache_dir)


def test_resolve_dataset_fingerprint_from_arg():
    """Test _resolve_dataset_fingerprint with provided fingerprint."""
    fp = _resolve_dataset_fingerprint(None, provided="test_fp")
    assert fp == "test_fp"


def test_resolve_dataset_fingerprint_from_meta():
    """Test _resolve_dataset_fingerprint from dataset.meta."""

    class DummyDataset:
        meta = {"dataset_fingerprint": "meta_fp"}

    fp = _resolve_dataset_fingerprint(DummyDataset(), None)
    assert fp == "meta_fp"


def test_resolve_dataset_fingerprint_from_meta_fallback():
    """Test _resolve_dataset_fingerprint fallback to 'fingerprint' key."""

    class DummyDataset:
        meta = {"fingerprint": "fallback_fp"}

    fp = _resolve_dataset_fingerprint(DummyDataset(), None)
    assert fp == "fallback_fp"


def test_resolve_dataset_fingerprint_missing():
    """Test _resolve_dataset_fingerprint raises when missing."""

    class DummyDataset:
        meta = {}

    with pytest.raises(MissingDatasetFingerprintError):
        _resolve_dataset_fingerprint(DummyDataset(), None)


def test_resolve_dataset_fingerprint_no_meta():
    """Test _resolve_dataset_fingerprint with dataset without meta."""

    class DummyDataset:
        pass

    with pytest.raises(MissingDatasetFingerprintError):
        _resolve_dataset_fingerprint(DummyDataset(), None)


def test_warn_on_sampling_stats_graph_missing_classes(caplog) -> None:
    masks = {
        "train": np.array([True, False, False]),
        "val": np.array([False, True, False]),
        "test": np.array([False, False, True]),
        "labeled": np.array([True, False, False]),
        "unlabeled": np.array([False, False, False]),
    }
    result = SamplingResult(
        schema_version=1,
        created_at="now",
        dataset_fingerprint="fp",
        split_fingerprint="sfp",
        plan={},
        masks=masks,
        stats={"labeled_class_dist": {"classes": {"0": 0, "1": 2}}},
    )
    with caplog.at_level("WARNING"):
        _warn_on_sampling_stats(result)


def test_warn_on_sampling_stats_inductive_missing_and_empty(caplog) -> None:
    indices = {
        "train": np.array([], dtype=np.int64),
        "val": np.array([], dtype=np.int64),
        "test": np.array([], dtype=np.int64),
        "train_labeled": np.array([], dtype=np.int64),
        "train_unlabeled": np.array([], dtype=np.int64),
    }
    result = SamplingResult(
        schema_version=1,
        created_at="now",
        dataset_fingerprint="fp",
        split_fingerprint="sfp",
        plan={},
        indices=indices,
        stats={
            "train": {"classes": {"0": 2, "1": 1}},
            "train_labeled": {"classes": {"0": 0, "1": 1}},
        },
    )
    with caplog.at_level("WARNING"):
        _warn_on_sampling_stats(result)


def test_warn_on_sampling_stats_graph_non_dict_classes(caplog) -> None:
    masks = {
        "train": np.array([True, False, False]),
        "val": np.array([False, True, False]),
        "test": np.array([False, False, True]),
        "labeled": np.array([True, False, False]),
        "unlabeled": np.array([False, False, False]),
    }
    result = SamplingResult(
        schema_version=1,
        created_at="now",
        dataset_fingerprint="fp",
        split_fingerprint="sfp",
        plan={},
        masks=masks,
        stats={"labeled_class_dist": {"classes": []}},
    )
    with caplog.at_level("WARNING"):
        _warn_on_sampling_stats(result)


def test_warn_on_sampling_stats_inductive_non_dict_stats(caplog) -> None:
    indices = {
        "train": np.array([0], dtype=np.int64),
        "val": np.array([], dtype=np.int64),
        "test": np.array([], dtype=np.int64),
        "train_labeled": np.array([0], dtype=np.int64),
        "train_unlabeled": np.array([], dtype=np.int64),
    }
    result = SamplingResult(
        schema_version=1,
        created_at="now",
        dataset_fingerprint="fp",
        split_fingerprint="sfp",
        plan={},
        indices=indices,
        stats={"train": "nope", "train_labeled": "nope"},
    )
    with caplog.at_level("WARNING"):
        _warn_on_sampling_stats(result)


def test_warn_on_sampling_stats_inductive_non_dict_classes(caplog) -> None:
    indices = {
        "train": np.array([0], dtype=np.int64),
        "val": np.array([], dtype=np.int64),
        "test": np.array([], dtype=np.int64),
        "train_labeled": np.array([0], dtype=np.int64),
        "train_unlabeled": np.array([], dtype=np.int64),
    }
    result = SamplingResult(
        schema_version=1,
        created_at="now",
        dataset_fingerprint="fp",
        split_fingerprint="sfp",
        plan={},
        indices=indices,
        stats={
            "train": {"classes": []},
            "train_labeled": {"classes": []},
        },
    )
    with caplog.at_level("WARNING"):
        _warn_on_sampling_stats(result)
