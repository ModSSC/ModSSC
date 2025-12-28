from unittest.mock import MagicMock

import numpy as np
import pytest

from modssc.supervised.errors import SupervisedValidationError
from modssc.supervised.utils import as_numpy, encode_labels, ensure_2d, onehot


def test_as_numpy_torch_like():
    mock_tensor = MagicMock()
    mock_tensor.detach.return_value = mock_tensor
    mock_tensor.cpu.return_value = mock_tensor
    mock_tensor.numpy.return_value = np.array([1, 2, 3])

    res = as_numpy(mock_tensor)
    assert isinstance(res, np.ndarray)
    assert np.array_equal(res, np.array([1, 2, 3]))
    mock_tensor.detach.assert_called_once()
    mock_tensor.cpu.assert_called_once()
    mock_tensor.numpy.assert_called_once()


def test_as_numpy_others():
    arr = np.array([1, 2])
    assert as_numpy(arr) is arr

    lst = [1, 2]
    res = as_numpy(lst)
    assert isinstance(res, np.ndarray)
    assert np.array_equal(res, np.array([1, 2]))


def test_ensure_2d_high_dim():
    arr = np.zeros((2, 3, 4))
    res = ensure_2d(arr)
    assert res.shape == (2, 12)


def test_ensure_2d_low_dim():
    arr = np.array([1, 2, 3])
    res = ensure_2d(arr)
    assert res.shape == (3, 1)

    arr2 = np.zeros((2, 3))
    res2 = ensure_2d(arr2)
    assert res2 is arr2


def test_ensure_2d_invalid():
    with pytest.raises(SupervisedValidationError, match="got ndim=0"):
        ensure_2d(np.array(1.0))


def test_onehot_empty():
    res = onehot(np.array([]), n_classes=3)
    assert res.shape == (0, 3)
    assert res.size == 0


def test_encode_labels():
    y = ["b", "a", "b", "c"]
    y_enc, classes = encode_labels(y)
    np.testing.assert_array_equal(classes, np.array(["a", "b", "c"]))
    np.testing.assert_array_equal(y_enc, np.array([1, 0, 1, 2]))
