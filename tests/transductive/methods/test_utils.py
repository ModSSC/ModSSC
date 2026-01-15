import numpy as np
import pytest

from modssc.transductive.methods.utils import (
    _validate_graph_inputs,
    labels_to_onehot,
    normalize_edge_weight_numpy,
    normalize_edge_weight_torch,
    spmm_numpy,
    spmm_torch,
    to_numpy,
)

try:
    import torch
except Exception:
    torch = None


def test_validate_graph_inputs():
    n_nodes = 3
    edge_index = np.array([[0, 1, 2], [1, 2, 0]])
    edge_weight = np.array([0.5, 0.5, 0.5])

    ei, ew = _validate_graph_inputs(n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight)
    assert np.array_equal(ei, edge_index)
    assert np.array_equal(ew, edge_weight)

    ei, ew = _validate_graph_inputs(n_nodes=n_nodes, edge_index=edge_index, edge_weight=None)
    assert np.array_equal(ei, edge_index)
    assert np.array_equal(ew, np.ones(3, dtype=np.float32))

    with pytest.raises(ValueError, match="edge_index must have shape"):
        _validate_graph_inputs(n_nodes=n_nodes, edge_index=np.array([0, 1, 2]), edge_weight=None)

    with pytest.raises(ValueError, match="edge_weight must have shape"):
        _validate_graph_inputs(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=np.array([0.5, 0.5])
        )


def test_validate_graph_inputs_edge_cases():
    n_nodes = 3
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([0.5, 0.5])

    ei_float = edge_index.astype(float)
    ei, _ = _validate_graph_inputs(n_nodes=n_nodes, edge_index=ei_float, edge_weight=edge_weight)
    assert ei.dtype == np.int64

    with pytest.raises(ValueError, match="n_nodes must be positive"):
        _validate_graph_inputs(n_nodes=0, edge_index=edge_index, edge_weight=edge_weight)

    ei_neg = np.array([[-1, 0], [0, 1]])
    with pytest.raises(ValueError, match="edge_index must be non-negative"):
        _validate_graph_inputs(n_nodes=n_nodes, edge_index=ei_neg, edge_weight=edge_weight)

    ei_oob = np.array([[0, 3], [1, 2]])
    with pytest.raises(ValueError, match="edge_index contains node id >= n_nodes"):
        _validate_graph_inputs(n_nodes=n_nodes, edge_index=ei_oob, edge_weight=edge_weight)


def test_normalize_edge_weight_numpy():
    n_nodes = 3

    edge_index = np.array([[0, 0, 1], [0, 1, 2]])
    edge_weight = np.array([1.0, 1.0, 2.0])

    w_row = normalize_edge_weight_numpy(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="rw"
    )
    expected_row = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(w_row, expected_row)

    with pytest.raises(ValueError, match="Unknown norm mode"):
        normalize_edge_weight_numpy(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="invalid"
        )


def test_spmm_numpy():
    n_nodes = 3

    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([0.5, 2.0])
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    res = spmm_numpy(n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, X=X)
    expected = np.array([[0.0, 0.0], [0.5, 1.0], [6.0, 8.0]])
    np.testing.assert_allclose(res, expected)


def test_spmm_numpy_errors():
    n_nodes = 3
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([0.5, 0.5])

    X_invalid = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="X must have shape"):
        spmm_numpy(n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, X=X_invalid)

    X_invalid_rows = np.zeros((2, 2))
    with pytest.raises(ValueError, match="X must have shape"):
        spmm_numpy(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, X=X_invalid_rows
        )


def test_to_numpy():
    arr = np.array([1, 2, 3])
    assert to_numpy(arr) is arr

    lst = [1, 2, 3]
    assert np.array_equal(to_numpy(lst), np.array(lst))

    if torch is not None:
        t = torch.tensor([1, 2, 3])
        assert np.array_equal(to_numpy(t), np.array([1, 2, 3]))

        t_cpu = t.cpu()
        assert np.array_equal(to_numpy(t_cpu), np.array([1, 2, 3]))


def test_labels_to_onehot_invalid_classes():
    with pytest.raises(ValueError, match="n_classes must be positive"):
        labels_to_onehot([0, 1], n_classes=0)


def test_labels_to_onehot_numpy():
    y = np.array([0, 1, 0], dtype=np.int64)
    out = labels_to_onehot(y, n_classes=2)
    assert out.shape == (3, 2)
    assert out.dtype == np.float32
    assert np.allclose(out[0], np.array([1.0, 0.0], dtype=np.float32))


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_labels_to_onehot_torch_tensor():
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    out = labels_to_onehot(y, n_classes=2)
    assert out.shape == (3, 2)
    assert out.dtype == torch.float32
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0]))


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_normalize_edge_weight_torch():
    n_nodes = 3
    edge_index = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float)

    w_row = normalize_edge_weight_torch(
        n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="rw"
    )
    expected_row = torch.tensor([1.0, 1.0, 1.0])
    assert torch.allclose(w_row, expected_row)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_normalize_edge_weight_torch_errors():
    n_nodes = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.5], dtype=torch.float)

    with pytest.raises(ValueError, match="Unknown norm mode"):
        normalize_edge_weight_torch(
            n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, mode="invalid"
        )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_spmm_torch():
    n_nodes = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 2.0], dtype=torch.float)
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    res = spmm_torch(n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, X=X)
    expected = torch.tensor([[0.0, 0.0], [0.5, 1.0], [6.0, 8.0]])
    assert torch.allclose(res, expected)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_spmm_torch_errors():
    n_nodes = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.5], dtype=torch.float)

    X_invalid = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="X must have shape"):
        spmm_torch(n_nodes=n_nodes, edge_index=edge_index, edge_weight=edge_weight, X=X_invalid)


class FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def test_to_numpy_detach_cpu_numpy() -> None:
    arr = np.array([1, 2, 3], dtype=np.float32)
    out = to_numpy(FakeTensor(arr))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)
