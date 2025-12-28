import numpy as np

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


def test_result_properties_inductive():
    """Test property accessors for inductive result (indices)."""
    indices = {
        "train": np.array([0, 1, 2]),
        "val": np.array([3]),
        "test": np.array([4, 5]),
        "train_labeled": np.array([0, 1]),
        "train_unlabeled": np.array([2]),
    }
    res = _make_result(indices=indices)

    assert not res.is_graph()

    np.testing.assert_array_equal(res.train_idx, indices["train"])
    np.testing.assert_array_equal(res.val_idx, indices["val"])
    np.testing.assert_array_equal(res.test_idx, indices["test"])
    np.testing.assert_array_equal(res.labeled_idx, indices["train_labeled"])
    np.testing.assert_array_equal(res.unlabeled_idx, indices["train_unlabeled"])


def test_result_properties_graph():
    """Test property accessors for graph result (masks)."""
    masks = {
        "train": np.array([True, True, True, False, False, False]),
        "val": np.array([False, False, False, True, False, False]),
        "test": np.array([False, False, False, False, True, True]),
        "labeled": np.array([True, True, False, False, False, False]),
        "unlabeled": np.array([False, False, True, False, False, False]),
    }
    res = _make_result(masks=masks)

    assert res.is_graph()

    np.testing.assert_array_equal(res.train_idx, [0, 1, 2])
    np.testing.assert_array_equal(res.val_idx, [3])
    np.testing.assert_array_equal(res.test_idx, [4, 5])
    np.testing.assert_array_equal(res.labeled_idx, [0, 1])
    np.testing.assert_array_equal(res.unlabeled_idx, [2])


def test_result_properties_inductive_defaults():
    """Test property accessors defaults when indices are missing."""
    res = _make_result(indices={})

    assert res.train_idx.size == 0
    assert res.val_idx.size == 0
    assert res.test_idx.size == 0
    assert res.labeled_idx.size == 0
    assert res.unlabeled_idx.size == 0
