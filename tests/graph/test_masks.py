import numpy as np
import pytest

from modssc.graph.errors import GraphValidationError
from modssc.graph.masks import Masks, masks_from_indices


def test_masks_dataclass():
    n = 10
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    unlabeled = np.zeros(n, dtype=bool)

    masks = Masks(train=train, val=val, test=test, unlabeled=unlabeled)
    assert masks.train is train
    assert masks.val is val
    assert masks.test is test
    assert masks.unlabeled is unlabeled

    d = masks.as_dict()
    assert d["train"] is train
    assert d["val"] is val
    assert d["test"] is test
    assert d["unlabeled"] is unlabeled


def test_masks_from_indices_basic():
    n = 10
    train_idx = [0, 1]
    val_idx = [2, 3]
    test_idx = [4, 5]

    masks = masks_from_indices(n=n, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    assert masks.train.sum() == 2
    assert masks.train[0] and masks.train[1]
    assert masks.val.sum() == 2
    assert masks.val[2] and masks.val[3]
    assert masks.test.sum() == 2
    assert masks.test[4] and masks.test[5]

    assert masks.unlabeled.sum() == 8
    assert not masks.unlabeled[0]
    assert masks.unlabeled[2]


def test_masks_from_indices_with_labeled_idx():
    n = 5
    train_idx = [0, 1, 2]
    labeled_idx = [0, 1]

    masks = masks_from_indices(
        n=n, train_idx=train_idx, val_idx=[], test_idx=[], labeled_idx=labeled_idx
    )

    assert masks.train.sum() == 3

    assert masks.unlabeled.sum() == 1
    assert masks.unlabeled[2]


def test_masks_from_indices_validation_n():
    with pytest.raises(GraphValidationError, match="n must be positive"):
        masks_from_indices(n=0, train_idx=[], val_idx=[], test_idx=[])


def test_masks_from_indices_validation_dims():
    n = 10
    with pytest.raises(GraphValidationError, match="train_idx must be 1D"):
        masks_from_indices(n=n, train_idx=[[0]], val_idx=[], test_idx=[])


def test_masks_from_indices_validation_bounds():
    n = 10
    with pytest.raises(GraphValidationError, match="train_idx contains indices outside"):
        masks_from_indices(n=n, train_idx=[10], val_idx=[], test_idx=[])

    with pytest.raises(GraphValidationError, match="val_idx contains indices outside"):
        masks_from_indices(n=n, train_idx=[], val_idx=[-1], test_idx=[])


def test_masks_from_indices_validation_labeled():
    n = 10
    with pytest.raises(GraphValidationError, match="labeled_idx must be 1D"):
        masks_from_indices(n=n, train_idx=[], val_idx=[], test_idx=[], labeled_idx=[[0]])

    with pytest.raises(GraphValidationError, match="labeled_idx contains indices outside"):
        masks_from_indices(n=n, train_idx=[], val_idx=[], test_idx=[], labeled_idx=[10])
