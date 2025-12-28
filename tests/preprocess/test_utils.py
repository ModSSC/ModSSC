from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.preprocess.errors import OptionalDependencyError
from modssc.preprocess.numpy_adapter import to_numpy
from modssc.preprocess.optional import is_available, require


def test_to_numpy_ndarray():
    arr = np.array([1, 2, 3])
    assert to_numpy(arr) is arr


def test_to_numpy_torch_like():
    mock_tensor = MagicMock()
    mock_tensor.detach.return_value = mock_tensor
    mock_tensor.cpu.return_value = mock_tensor
    mock_tensor.numpy.return_value = np.array([1, 2, 3])

    res = to_numpy(mock_tensor)
    assert isinstance(res, np.ndarray)
    assert np.all(res == [1, 2, 3])
    mock_tensor.detach.assert_called_once()
    mock_tensor.cpu.assert_called_once()
    mock_tensor.numpy.assert_called_once()


def test_to_numpy_detach_fail():
    mock_obj = MagicMock()
    mock_obj.detach.side_effect = Exception("detach failed")

    mock_obj.numpy.return_value = np.array([1])

    res = to_numpy(mock_obj)
    assert isinstance(res, np.ndarray)


def test_to_numpy_cpu_fail():
    mock_obj = MagicMock()
    mock_obj.cpu.side_effect = Exception("cpu failed")
    mock_obj.numpy.return_value = np.array([1])

    res = to_numpy(mock_obj)
    assert isinstance(res, np.ndarray)


def test_to_numpy_numpy_fail():
    mock_obj = MagicMock()
    mock_obj.numpy.side_effect = Exception("numpy failed")

    class FailNumpy:
        def numpy(self):
            raise Exception("fail")

        def __array__(self):
            return np.array([99])

    obj = FailNumpy()
    res = to_numpy(obj)
    assert res[0] == 99


def test_to_numpy_numpy_returns_non_array():
    mock_obj = MagicMock()
    mock_obj.numpy.return_value = "not an array"

    res = to_numpy(mock_obj)
    assert isinstance(res, np.ndarray)


def test_is_available():
    assert is_available("os") is True
    assert is_available("non_existent_module_xyz_123") is False

    with patch("importlib.import_module", side_effect=Exception("Boom")):
        assert is_available("os") is False


def test_require():
    assert require(module="os", extra="test") is not None

    with pytest.raises(OptionalDependencyError) as exc:
        require(module="non_existent_module_xyz_123", extra="my_feature", purpose="testing")
    assert "my_feature" in str(exc.value)
    assert "testing" in str(exc.value)


def test_to_numpy_simple_list():
    data = [1, 2, 3]
    res = to_numpy(data)
    assert isinstance(res, np.ndarray)
    assert np.all(res == [1, 2, 3])
