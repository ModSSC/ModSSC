"""Test coverage for sampling/result.py."""

from __future__ import annotations

import numpy as np
import pytest

from modssc.sampling.errors import SamplingValidationError
from modssc.sampling.result import SamplingResult


def _make_result(indices=None, refs=None, masks=None):
    return SamplingResult(
        schema_version=1,
        created_at="now",
        dataset_fingerprint="d",
        split_fingerprint="s",
        plan={},
        indices=indices or {},
        refs=refs or {},
        masks=masks or {},
    )


def test_result_inductive_invalid_dtype():
    """Test indices with invalid dtype."""
    res = _make_result(indices={"train": np.array([1.0, 2.0])})
    with pytest.raises(SamplingValidationError, match="indices must be integers"):
        res.validate(n_train=10, n_test=5, n_nodes=None)


def test_result_inductive_missing_ntest():
    """Test missing n_test when base is test."""
    res = _make_result(indices={"test": np.array([0])}, refs={"test": "test"})
    with pytest.raises(SamplingValidationError, match="n_test is required when base='test'"):
        res.validate(n_train=10, n_test=None, n_nodes=None)


def test_result_inductive_out_of_range():
    """Test indices out of range."""
    res = _make_result(indices={"train": np.array([10])})
    with pytest.raises(SamplingValidationError, match="out-of-range indices"):
        res.validate(n_train=10, n_test=5, n_nodes=None)


def test_result_inductive_duplicates():
    """Test duplicate indices."""
    res = _make_result(indices={"train": np.array([0, 0])})
    with pytest.raises(SamplingValidationError, match="contains duplicate indices"):
        res.validate(n_train=10, n_test=5, n_nodes=None)


def test_result_inductive_missing_keys():
    """Test missing required keys."""

    res = _make_result(indices={})
    with pytest.raises(SamplingValidationError, match="Missing train indices"):
        res.validate(n_train=10, n_test=5, n_nodes=None)

    res = _make_result(indices={"train": np.array([], dtype=int)})
    with pytest.raises(SamplingValidationError, match="Missing val indices"):
        res.validate(n_train=10, n_test=5, n_nodes=None)

    res = _make_result(indices={"train": np.array([], dtype=int), "val": np.array([], dtype=int)})
    with pytest.raises(SamplingValidationError, match="Missing test indices"):
        res.validate(n_train=10, n_test=5, n_nodes=None)


def test_result_inductive_invalid_refs():
    """Test invalid refs for train/val."""
    res = _make_result(
        indices={
            "train": np.array([], dtype=int),
            "val": np.array([], dtype=int),
            "test": np.array([], dtype=int),
        },
        refs={"train": "test"},
    )
    with pytest.raises(
        SamplingValidationError, match="train and val indices must be relative to dataset.train"
    ):
        res.validate(n_train=10, n_test=5, n_nodes=None)


def test_result_inductive_train_val_overlap():
    """Test overlap between train and val."""
    res = _make_result(
        indices={
            "train": np.array([0]),
            "val": np.array([0]),
            "test": np.array([], dtype=int),
            "train_labeled": np.array([0]),
            "train_unlabeled": np.array([], dtype=int),
        }
    )
    with pytest.raises(SamplingValidationError, match="train and val overlap"):
        res.validate(n_train=10, n_test=5, n_nodes=None)


def test_result_inductive_test_overlap():
    """Test overlap between test and train/val when test ref is train."""
    res = _make_result(
        indices={
            "train": np.array([0]),
            "val": np.array([1]),
            "test": np.array([0]),
            "train_labeled": np.array([0]),
            "train_unlabeled": np.array([], dtype=int),
        },
        refs={"test": "train"},
    )
    with pytest.raises(SamplingValidationError, match="test overlaps with train or val"):
        res.validate(n_train=10, n_test=5, n_nodes=None)


def test_result_inductive_missing_labeled():
    """Test missing labeled/unlabeled indices."""
    res = _make_result(
        indices={"train": np.array([0]), "val": np.array([1]), "test": np.array([2])}
    )
    with pytest.raises(SamplingValidationError, match="Missing labeled/unlabeled indices"):
        res.validate(n_train=10, n_test=5, n_nodes=None)


def test_result_inductive_labeled_unlabeled_overlap():
    """Test overlap between labeled and unlabeled."""
    res = _make_result(
        indices={
            "train": np.array([0]),
            "val": np.array([1]),
            "test": np.array([2]),
            "train_labeled": np.array([0]),
            "train_unlabeled": np.array([0]),
        }
    )
    with pytest.raises(SamplingValidationError, match="labeled and unlabeled overlap"):
        res.validate(n_train=10, n_test=5, n_nodes=None)


def test_result_inductive_labeled_unlabeled_mismatch():
    """Test labeled + unlabeled != train."""
    res = _make_result(
        indices={
            "train": np.array([0, 1]),
            "val": np.array([2]),
            "test": np.array([3]),
            "train_labeled": np.array([0]),
            "train_unlabeled": np.array([], dtype=int),
        }
    )
    with pytest.raises(
        SamplingValidationError, match=r"labeled \+ unlabeled must cover train exactly"
    ):
        res.validate(n_train=10, n_test=5, n_nodes=None)


def test_result_graph_missing_nnodes():
    """Test missing n_nodes for graph."""
    res = _make_result(masks={"train": np.array([])})
    with pytest.raises(
        SamplingValidationError, match="n_nodes is required to validate graph masks"
    ):
        res.validate(n_train=10, n_test=5, n_nodes=None)


def test_result_graph_missing_keys():
    """Test missing keys in graph masks."""
    res = _make_result(masks={"train": np.array([])})
    with pytest.raises(SamplingValidationError, match="Graph masks must have keys"):
        res.validate(n_train=10, n_test=5, n_nodes=10)


def test_result_graph_invalid_dtype():
    """Test invalid dtype in graph masks."""
    masks = {k: np.zeros(10, dtype=bool) for k in ["train", "val", "test", "labeled", "unlabeled"]}
    masks["train"] = np.zeros(10, dtype=int)
    res = _make_result(masks=masks)
    with pytest.raises(SamplingValidationError, match="must be bool"):
        res.validate(n_train=10, n_test=5, n_nodes=10)


def test_result_graph_invalid_shape():
    """Test invalid shape in graph masks."""
    masks = {k: np.zeros(10, dtype=bool) for k in ["train", "val", "test", "labeled", "unlabeled"]}
    masks["train"] = np.zeros(5, dtype=bool)
    res = _make_result(masks=masks)
    with pytest.raises(SamplingValidationError, match="must have shape"):
        res.validate(n_train=10, n_test=5, n_nodes=10)


def test_result_graph_labeled_not_subset():
    """Test labeled mask not subset of train mask."""
    masks = {k: np.zeros(10, dtype=bool) for k in ["train", "val", "test", "labeled", "unlabeled"]}
    masks["labeled"][0] = True

    res = _make_result(masks=masks)
    with pytest.raises(SamplingValidationError, match="labeled_mask must be subset of train_mask"):
        res.validate(n_train=10, n_test=5, n_nodes=10)


def test_result_graph_unlabeled_mismatch():
    """Test unlabeled mask mismatch."""
    masks = {k: np.zeros(10, dtype=bool) for k in ["train", "val", "test", "labeled", "unlabeled"]}
    masks["train"][0] = True
    masks["labeled"][0] = True

    masks["unlabeled"][0] = True
    res = _make_result(masks=masks)
    with pytest.raises(SamplingValidationError, match="unlabeled_mask must equal train_mask"):
        res.validate(n_train=10, n_test=5, n_nodes=10)


def test_result_graph_disjointness():
    """Test disjointness of train/val/test masks."""
    masks = {k: np.zeros(10, dtype=bool) for k in ["train", "val", "test", "labeled", "unlabeled"]}
    masks["train"][0] = True
    masks["val"][0] = True
    masks["unlabeled"][0] = True
    res = _make_result(masks=masks)
    with pytest.raises(SamplingValidationError, match="train/val/test masks must be disjoint"):
        res.validate(n_train=10, n_test=5, n_nodes=10)


def test_result_graph_valid():
    """Test valid graph result."""
    masks = {k: np.zeros(10, dtype=bool) for k in ["train", "val", "test", "labeled", "unlabeled"]}

    masks["train"][0] = True
    masks["train"][1] = True
    masks["val"][2] = True
    masks["test"][3] = True

    masks["labeled"][0] = True
    masks["unlabeled"][1] = True

    res = _make_result(masks=masks)

    res.validate(n_train=10, n_test=5, n_nodes=10)


def test_result_inductive_valid():
    """Test valid inductive result."""
    res = _make_result(
        indices={
            "train": np.array([0, 1]),
            "val": np.array([2]),
            "test": np.array([3]),
            "train_labeled": np.array([0]),
            "train_unlabeled": np.array([1]),
        }
    )

    res.validate(n_train=10, n_test=5, n_nodes=None)
