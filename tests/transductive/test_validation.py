from types import SimpleNamespace

import numpy as np
import pytest

from modssc.transductive.errors import TransductiveValidationError
from modssc.transductive.validation import _as_numpy, validate_node_dataset


class MockTensor:
    def __init__(self, array, has_detach=True, has_cpu=True):
        self._array = array
        if has_detach:
            self.detach = self._detach

        if has_cpu:
            self.cpu = self._cpu

        self.numpy = self._numpy

    def _detach(self):
        return self

    def _cpu(self):
        return self

    def _numpy(self):
        return self._array


def test_as_numpy_tensor_detach_cpu():
    arr = np.array([1, 2, 3])
    tensor = MockTensor(arr, has_detach=True, has_cpu=True)
    assert np.array_equal(_as_numpy(tensor), arr)


def test_as_numpy_tensor_cpu_only():
    arr = np.array([1, 2, 3])
    tensor = MockTensor(arr, has_detach=False, has_cpu=True)
    assert np.array_equal(_as_numpy(tensor), arr)


def test_as_numpy_tensor_numpy_only():
    arr = np.array([1, 2, 3])
    tensor = MockTensor(arr, has_detach=False, has_cpu=False)
    assert np.array_equal(_as_numpy(tensor), arr)


def test_as_numpy_list():
    data = [1, 2, 3]
    assert np.array_equal(_as_numpy(data), np.array(data))


def test_validate_node_dataset_none():
    with pytest.raises(TransductiveValidationError, match="data must not be None"):
        validate_node_dataset(None)


def test_validate_node_dataset_X_wrong_dim():
    data = SimpleNamespace(
        X=np.array([1, 2, 3]),
        y=np.array([1, 2, 3]),
        graph=SimpleNamespace(edge_index=np.array([[0], [1]])),
        masks={},
    )
    with pytest.raises(TransductiveValidationError, match="X must be 2D"):
        validate_node_dataset(data)


def test_validate_node_dataset_y_wrong_dim():
    data = SimpleNamespace(
        X=np.array([[1], [2], [3]]),
        y=np.array([[1], [2], [3]]),
        graph=SimpleNamespace(edge_index=np.array([[0], [1]])),
        masks={},
    )
    with pytest.raises(TransductiveValidationError, match="y must be 1D"):
        validate_node_dataset(data)


def test_validate_node_dataset_mismatch_len():
    data = SimpleNamespace(
        X=np.array([[1], [2], [3]]),
        y=np.array([0, 1]),
        graph=SimpleNamespace(edge_index=np.array([[0], [1]])),
        masks={},
    )
    with pytest.raises(
        TransductiveValidationError, match="X and y must have the same first dimension"
    ):
        validate_node_dataset(data)


def test_validate_node_dataset_no_graph():
    data = SimpleNamespace(X=np.array([[1], [2]]), y=np.array([0, 1]), graph=None, masks={})
    with pytest.raises(TransductiveValidationError, match="data.graph must not be None"):
        validate_node_dataset(data)


def test_validate_node_dataset_no_edge_index():
    data = SimpleNamespace(
        X=np.array([[1], [2]]),
        y=np.array([0, 1]),
        graph=SimpleNamespace(),
        masks={},
    )
    with pytest.raises(TransductiveValidationError, match="graph.edge_index is required"):
        validate_node_dataset(data)


def test_validate_node_dataset_edge_index_wrong_shape():
    data = SimpleNamespace(
        X=np.array([[1], [2]]),
        y=np.array([0, 1]),
        graph=SimpleNamespace(edge_index=np.array([0, 1])),
        masks={},
    )
    with pytest.raises(TransductiveValidationError, match="edge_index must have shape"):
        validate_node_dataset(data)


def test_validate_node_dataset_edge_index_out_of_bounds():
    data = SimpleNamespace(
        X=np.array([[1], [2]]),
        y=np.array([0, 1]),
        graph=SimpleNamespace(edge_index=np.array([[0, 2], [1, 0]])),
        masks={},
    )
    with pytest.raises(
        TransductiveValidationError, match="edge_index has out of range node indices"
    ):
        validate_node_dataset(data)


def test_validate_node_dataset_empty_edge_index():
    data = SimpleNamespace(
        X=np.array([[1], [2]]),
        y=np.array([0, 1]),
        graph=SimpleNamespace(edge_index=np.empty((2, 0))),
        masks={},
    )
    validate_node_dataset(data)


def test_validate_node_dataset_mask_wrong_shape():
    data = SimpleNamespace(
        X=np.array([[1], [2]]),
        y=np.array([0, 1]),
        graph=SimpleNamespace(edge_index=np.array([[0, 1], [1, 0]])),
        masks={"train_mask": np.array([True])},
    )
    with pytest.raises(TransductiveValidationError, match="train_mask must have shape"):
        validate_node_dataset(data)


def test_validate_node_dataset_success():
    data = SimpleNamespace(
        X=np.array([[1], [2]]),
        y=np.array([0, 1]),
        graph=SimpleNamespace(edge_index=np.array([[0, 1], [1, 0]])),
        masks={"train_mask": np.array([True, False])},
    )
    validate_node_dataset(data)


def test_validate_node_dataset_non_contiguous_labels():
    data = SimpleNamespace(
        X=np.array([[1], [2]]),
        y=np.array([1, 2]),
        graph=SimpleNamespace(edge_index=np.array([[0, 1], [1, 0]])),
        masks={"train_mask": np.array([True, False])},
    )
    with pytest.raises(TransductiveValidationError, match="contiguous class ids"):
        validate_node_dataset(data)


def test_validate_node_dataset_non_integer_labels():
    data = SimpleNamespace(
        X=np.array([[1], [2]]),
        y=np.array([0.0, 1.5]),
        graph=SimpleNamespace(edge_index=np.array([[0, 1], [1, 0]])),
        masks={"train_mask": np.array([True, False])},
    )
    with pytest.raises(TransductiveValidationError, match="integer class ids"):
        validate_node_dataset(data)


def test_validate_node_dataset_float_integer_labels():
    data = SimpleNamespace(
        X=np.array([[1], [2]]),
        y=np.array([0.0, 1.0]),
        graph=SimpleNamespace(edge_index=np.array([[0, 1], [1, 0]])),
        masks={"train_mask": np.array([True, False])},
    )
    validate_node_dataset(data)


def test_validate_node_dataset_non_finite_labels():
    data = SimpleNamespace(
        X=np.array([[1], [2]]),
        y=np.array([0.0, np.nan]),
        graph=SimpleNamespace(edge_index=np.array([[0, 1], [1, 0]])),
        masks={"train_mask": np.array([True, False])},
    )
    with pytest.raises(TransductiveValidationError, match="finite integer class ids"):
        validate_node_dataset(data)


def test_validate_node_dataset_non_numeric_labels():
    data = SimpleNamespace(
        X=np.array([[1], [2]]),
        y=np.array(["0", "1"]),
        graph=SimpleNamespace(edge_index=np.array([[0, 1], [1, 0]])),
        masks={"train_mask": np.array([True, False])},
    )
    with pytest.raises(TransductiveValidationError, match="integer class ids"):
        validate_node_dataset(data)


def test_validate_node_dataset_all_negative_labels():
    data = SimpleNamespace(
        X=np.array([[1], [2]]),
        y=np.array([-1, -1]),
        graph=SimpleNamespace(edge_index=np.array([[0, 1], [1, 0]])),
        masks={"train_mask": np.array([True, False])},
    )
    validate_node_dataset(data)
