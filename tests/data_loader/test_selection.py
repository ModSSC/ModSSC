from __future__ import annotations

import importlib

import numpy as np
import pytest

import modssc.data_loader.selection as selection
from modssc.data_loader import DatasetSelectionError, select_rows


def test_select_rows_preserves_mapping_metadata_and_container_types() -> None:
    values = {
        "x": np.arange(15).reshape(5, 3),
        "labels": ["a", "b", "c", "d", "e"],
        "schema": {"version": 1},
    }

    selected = select_rows(values, [3, 1])

    np.testing.assert_array_equal(selected["x"], values["x"][[3, 1]])
    assert selected["labels"] == ["d", "b"]
    assert selected["schema"] is values["schema"]


def test_select_rows_rejects_negative_indices_with_native_error() -> None:
    with pytest.raises(DatasetSelectionError) as raised:
        select_rows(np.arange(4), [0, -1])

    assert raised.value.code == "E_DATA_SELECTION_NEGATIVE_INDEX"


def test_select_rows_reports_unsupported_contract_context() -> None:
    with pytest.raises(DatasetSelectionError, match="evaluation payload") as raised:
        select_rows(object(), [0], context="evaluation payload")

    assert raised.value.code == "E_DATA_SELECTION_UNSUPPORTED"


def test_empty_selection_does_not_index_scalar_mapping_metadata() -> None:
    scalar = np.asarray(7)

    selected = select_rows({"x": np.ones((2, 3)), "version": scalar}, [])

    assert selected["x"].shape == (0, 3)
    assert selected["version"] is scalar


def test_graph_selection_applies_the_edge_mask_to_edge_attributes() -> None:
    values = {
        "x": np.arange(12).reshape(4, 3),
        "edge_index": np.asarray([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]], dtype=np.int64),
        "edge_weight": np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
        "edge_attr": np.arange(10).reshape(5, 2),
        "num_nodes": 4,
        "n_edges": 5,
    }

    selected = select_rows(values, [0, 2])

    np.testing.assert_array_equal(selected["edge_index"], np.asarray([[0], [1]]))
    np.testing.assert_array_equal(selected["edge_weight"], np.asarray([0.2]))
    np.testing.assert_array_equal(selected["edge_attr"], np.asarray([[2, 3]]))
    assert selected["num_nodes"] == 2
    assert selected["n_edges"] == 1


def test_mapping_metadata_is_not_selected_merely_because_indices_fit() -> None:
    vocabulary = np.asarray(["zero", "one", "two", "three", "four", "five"])
    selected = select_rows(
        {"x": np.arange(12).reshape(4, 3), "vocabulary": vocabulary},
        [0, 2],
    )

    assert selected["vocabulary"] is vocabulary


def test_graph_selection_rejects_misaligned_edge_attributes() -> None:
    values = {
        "x": np.arange(12).reshape(4, 3),
        "edge_index": np.asarray([[0, 2], [2, 3]], dtype=np.int64),
        "edge_weight": np.asarray([1.0]),
    }

    with pytest.raises(DatasetSelectionError) as raised:
        select_rows(values, [0, 2])

    assert raised.value.code == "E_DATA_SELECTION_EDGE_ALIGNMENT"


def test_selection_rejects_out_of_bounds_indices_before_backend_indexing() -> None:
    with pytest.raises(DatasetSelectionError) as raised:
        select_rows(np.arange(4), [4])

    assert raised.value.code == "E_DATA_SELECTION_INDEX_BOUNDS"


def test_selection_helpers_handle_missing_torch_and_unusual_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def missing_torch(name: str):
        if name == "torch":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", missing_torch)
    assert selection._is_torch_tensor(np.array([1])) is False

    class InvalidShape:
        def __len__(self):
            raise TypeError("shape unavailable")

    class Shaped:
        shape = InvalidShape()

    assert selection._leading_size(Shaped()) is None
    assert selection._leading_size(np.asarray(1)) is None
    assert selection._leading_size([1, 2]) == 2
    assert selection._leading_size(object()) is None
    assert selection._infer_population_size({"num_nodes": "4"}) == 4
    assert selection._infer_population_size({"num_nodes": "bad"}) is None
    assert selection._infer_population_size({}) is None


@pytest.mark.parametrize(
    ("edge_index", "num_nodes", "indices", "code"),
    [
        ([[0, 1, 2]], 3, [0], "E_DATA_SELECTION_GRAPH_SHAPE"),
        (np.array([[0.0], [1.0]]), 2, [0, 1], "E_DATA_SELECTION_GRAPH_DTYPE"),
        (np.array([[-1], [0]], dtype=np.int64), 2, [0], "E_DATA_SELECTION_GRAPH_BOUNDS"),
        (np.array([[0], [2]], dtype=np.int64), 2, [0], "E_DATA_SELECTION_GRAPH_BOUNDS"),
        (np.empty((2, 0), dtype=np.int64), None, [0], "E_DATA_SELECTION_GRAPH_BOUNDS"),
        (np.array([[0], [1]], dtype=np.int64), 2, [0, 0], "E_DATA_SELECTION_GRAPH_DUPLICATE_NODE"),
    ],
)
def test_graph_selection_rejects_invalid_topology_contracts(
    edge_index, num_nodes, indices, code
) -> None:
    values = {"edge_index": edge_index}
    if num_nodes is not None:
        values["num_nodes"] = num_nodes

    with pytest.raises(DatasetSelectionError) as raised:
        select_rows(values, indices)

    assert raised.value.code == code


def test_normalize_edge_index_rejects_unreadable_shape() -> None:
    class InvalidShape:
        def __len__(self):
            raise ValueError("no shape")

    class Edges:
        shape = InvalidShape()

    with pytest.raises(DatasetSelectionError) as raised:
        selection._normalize_edge_index(Edges())
    assert raised.value.code == "E_DATA_SELECTION_GRAPH_SHAPE"


def test_graph_selection_preserves_torch_backend_and_slices_edge_fields() -> None:
    torch = pytest.importorskip("torch")
    values = {
        "x": torch.arange(8).reshape(4, 2),
        "edge_index": torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]], dtype=torch.int32),
        "edge_weight": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "num_nodes": 4,
        "n_edges": 4,
    }

    selected = select_rows(values, [0, 2])

    assert isinstance(selected["edge_index"], torch.Tensor)
    assert selected["edge_index"].dtype == torch.int32
    torch.testing.assert_close(selected["edge_index"], torch.tensor([[0], [1]], dtype=torch.int32))
    torch.testing.assert_close(selected["edge_weight"], torch.tensor([0.2]))
    assert selected["num_nodes"] == 2
    assert selected["n_edges"] == 1


@pytest.mark.parametrize("dtype", ["float32", "bool"])
def test_graph_selection_rejects_noninteger_torch_edges(dtype: str) -> None:
    torch = pytest.importorskip("torch")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=getattr(torch, dtype))
    with pytest.raises(DatasetSelectionError) as raised:
        select_rows({"x": torch.ones((2, 1)), "edge_index": edge_index}, [0, 1])
    assert raised.value.code == "E_DATA_SELECTION_GRAPH_DTYPE"


def test_slice_value_accepts_torch_masks_for_numpy_lists_and_tensors() -> None:
    torch = pytest.importorskip("torch")
    mask = torch.tensor([True, False, True])

    np.testing.assert_array_equal(selection._slice_value(np.arange(3), mask), [0, 2])
    assert selection._slice_value(["a", "b", "c"], mask) == ["a", "c"]
    torch.testing.assert_close(
        selection._slice_value(torch.tensor([3, 4, 5]), [2, 0]),
        torch.tensor([5, 3]),
    )
    torch.testing.assert_close(
        selection._slice_tensor(torch.tensor([3, 4, 5]), torch.tensor([True, False, True])),
        torch.tensor([3, 5]),
    )


def test_selection_handles_none_edge_only_mappings_and_custom_indexers() -> None:
    assert select_rows(None, [0]) is None
    edge_only = select_rows(
        {
            "edge_index": np.array([[0, 1], [1, 2]], dtype=np.int64),
            "edge_custom": ["a", "b"],
        },
        [0, 1],
    )
    np.testing.assert_array_equal(edge_only["edge_index"], np.array([[0], [1]]))
    assert edge_only["edge_custom"] == ["a"]

    class Indexable:
        def __getitem__(self, indices):
            return tuple(np.asarray(indices).tolist())

    assert select_rows(Indexable(), [2, 0]) == (2, 0)


def test_edge_alignment_detection_is_explicit() -> None:
    assert selection._is_edge_aligned("metadata", [1, 2], n_edges=2) is False
    assert selection._is_edge_aligned("edge_custom", [1, 2], n_edges=2) is True
