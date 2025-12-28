from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.data_loader.providers.pyg import (
    PyGProvider,
    _as_edge_index,
    _normalize_filter,
    _pad_labels,
    _subset_graph,
    _to_numpy,
)
from modssc.data_loader.types import DatasetIdentity
from modssc.data_loader.uri import ParsedURI


def test_resolve_with_dataset_name():
    provider = PyGProvider()
    parsed = ParsedURI(provider="pyg", reference="Planetoid/Cora")

    identity = provider.resolve(parsed, options={})

    assert identity.canonical_uri == "pyg:Planetoid/Cora"
    assert identity.dataset_id == "Planetoid"
    assert identity.resolved_kwargs == {
        "dataset_class": "Planetoid",
        "dataset_kwargs": {"name": "Cora"},
    }


def test_resolve_without_name():
    provider = PyGProvider()
    parsed = ParsedURI(provider="pyg", reference="KarateClub")

    identity = provider.resolve(parsed, options={})

    assert identity.canonical_uri == "pyg:KarateClub"
    assert identity.resolved_kwargs["dataset_class"] == "KarateClub"
    assert identity.resolved_kwargs["dataset_kwargs"] == {}


def test_resolve_allows_dataset_override():
    provider = PyGProvider()
    parsed = ParsedURI(provider="pyg", reference="Planetoid/Cora")

    identity = provider.resolve(parsed, options={"dataset_class": "KarateClub"})

    assert identity.canonical_uri == "pyg:KarateClub/Cora"
    assert identity.dataset_id == "KarateClub"
    assert identity.resolved_kwargs["dataset_class"] == "KarateClub"
    assert identity.resolved_kwargs["dataset_kwargs"]["name"] == "Cora"


def test_resolve_with_optional_kwargs():
    provider = PyGProvider()
    parsed = ParsedURI(provider="pyg", reference="Planetoid/Cora")

    identity = provider.resolve(parsed, options={"max_nodes": 10, "class_filter": [1], "seed": 7})

    assert identity.resolved_kwargs["max_nodes"] == 10
    assert identity.resolved_kwargs["class_filter"] == [1]
    assert identity.resolved_kwargs["seed"] == 7


def test_load_canonical_missing_dataset_class(tmp_path):
    provider = PyGProvider()
    identity = DatasetIdentity(
        provider="pyg",
        canonical_uri="pyg:Missing",
        dataset_id="Missing",
        version=None,
        modality="graph",
        task="node_classification",
        resolved_kwargs={"dataset_class": "Missing", "dataset_kwargs": {}},
    )

    with patch("modssc.data_loader.providers.pyg.optional_import") as mock_import:
        mock_import.return_value = object()

        with pytest.raises(ValueError, match="not found"):
            provider.load_canonical(identity, raw_dir=tmp_path)


def test_load_canonical_raises_on_bad_kwargs(tmp_path):
    provider = PyGProvider()
    identity = DatasetIdentity(
        provider="pyg",
        canonical_uri="pyg:NeedsName",
        dataset_id="NeedsName",
        version=None,
        modality="graph",
        task="node_classification",
        resolved_kwargs={"dataset_class": "NeedsName", "dataset_kwargs": {}},
    )

    class NeedsName:
        def __init__(self, name):
            self.name = name

    with patch("modssc.data_loader.providers.pyg.optional_import") as mock_import:
        mock_module = MagicMock()
        mock_module.NeedsName = NeedsName
        mock_import.return_value = mock_module

        with pytest.raises(ValueError, match="Failed to instantiate NeedsName"):
            provider.load_canonical(identity, raw_dir=tmp_path)


def test_load_canonical_empty_dataset(tmp_path):
    provider = PyGProvider()
    identity = DatasetIdentity(
        provider="pyg",
        canonical_uri="pyg:EmptyDataset",
        dataset_id="EmptyDataset",
        version=None,
        modality="graph",
        task="node_classification",
        resolved_kwargs={"dataset_class": "EmptyDataset", "dataset_kwargs": {}},
    )

    class EmptyDataset:
        def __init__(self, root):
            self.root = root

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise IndexError

    with patch("modssc.data_loader.providers.pyg.optional_import") as mock_import:
        mock_module = MagicMock()
        mock_module.EmptyDataset = EmptyDataset
        mock_import.return_value = mock_module

        with pytest.raises(ValueError, match="empty"):
            provider.load_canonical(identity, raw_dir=tmp_path)


def test_load_canonical_with_root_and_masks(tmp_path):
    provider = PyGProvider()
    parsed = ParsedURI(provider="pyg", reference="WithRoot/Cora")
    identity = provider.resolve(parsed, options={})

    class Data:
        def __init__(self):
            self.x = np.array([[1.0, 2.0]])
            self.y = np.array([0])
            self.edge_index = np.array([[0], [0]])
            self.train_mask = np.array([True])
            self.val_mask = np.array([False])
            self.test_mask = np.array([True])

    class WithRoot:
        def __init__(self, root, name=None):
            self.root = root
            self.name = name
            self._items = [Data()]

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx):
            return self._items[idx]

    with patch("modssc.data_loader.providers.pyg.optional_import") as mock_import:
        mock_module = MagicMock()
        mock_module.WithRoot = WithRoot
        mock_import.return_value = mock_module

        ds = provider.load_canonical(identity, raw_dir=tmp_path)

    assert ds.train.masks is not None
    assert set(ds.train.masks) == {"train", "val", "test"}
    assert np.array_equal(ds.train.X, np.array([[1.0, 2.0]]))
    assert ds.meta["dataset_class"] == "WithRoot"
    assert ds.meta["dataset_kwargs"]["name"] == "Cora"


def test_load_canonical_without_root_parameter(tmp_path):
    provider = PyGProvider()
    identity = DatasetIdentity(
        provider="pyg",
        canonical_uri="pyg:NoRoot",
        dataset_id="NoRoot",
        version=None,
        modality="graph",
        task="node_classification",
        resolved_kwargs={"dataset_class": "NoRoot", "dataset_kwargs": {"tag": "demo"}},
    )

    class NoRoot:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._items = [type("Data", (), {"x": [1, 2], "y": [0], "edge_index": None})()]

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx):
            return self._items[idx]

    with patch("modssc.data_loader.providers.pyg.optional_import") as mock_import:
        mock_module = MagicMock()
        mock_module.NoRoot = NoRoot
        mock_import.return_value = mock_module

        ds = provider.load_canonical(identity, raw_dir=tmp_path)

    assert ds.train.edges is None
    assert ds.train.masks is None
    assert np.array_equal(ds.train.X, np.array([1, 2]))
    assert ds.meta["dataset_kwargs"]["tag"] == "demo"


def test_load_canonical_applies_class_filter_and_max_nodes(tmp_path):
    provider = PyGProvider()
    identity = DatasetIdentity(
        provider="pyg",
        canonical_uri="pyg:Filter",
        dataset_id="Filter",
        version=None,
        modality="graph",
        task="node_classification",
        resolved_kwargs={
            "dataset_class": "Filter",
            "dataset_kwargs": {"name": "Demo"},
            "class_filter": [1],
            "max_nodes": 1,
            "seed": 123,
        },
    )

    class Data:
        def __init__(self):
            self.x = np.arange(8, dtype=np.float64).reshape(4, 2)
            self.y = np.array([0, 1, 0, 1])
            self.edge_index = np.array([[0, 1, 2], [1, 2, 3]])

    class Filter:
        def __init__(self, root, name=None):
            self.root = root
            self.name = name
            self._items = [Data()]

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx):
            return self._items[idx]

    with patch("modssc.data_loader.providers.pyg.optional_import") as mock_import:
        mock_module = MagicMock()
        mock_module.Filter = Filter
        mock_import.return_value = mock_module

        ds = provider.load_canonical(identity, raw_dir=tmp_path)

    assert ds.train.X.shape[0] == 1
    assert ds.train.y.shape[0] == 1
    assert int(ds.train.y[0]) == 1


def test_load_canonical_applies_max_nodes_without_seed(tmp_path):
    provider = PyGProvider()
    identity = DatasetIdentity(
        provider="pyg",
        canonical_uri="pyg:NoSeed",
        dataset_id="NoSeed",
        version=None,
        modality="graph",
        task="node_classification",
        resolved_kwargs={
            "dataset_class": "NoSeed",
            "dataset_kwargs": {},
            "max_nodes": 2,
            "seed": None,
        },
    )

    class Data:
        def __init__(self):
            self.x = np.arange(8, dtype=np.float64).reshape(4, 2)
            self.y = np.array([0, 1, 2, 3])
            self.edge_index = None

    class NoSeed:
        def __init__(self, root, **kwargs):
            self.root = root
            self._items = [Data()]

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx):
            return self._items[idx]

    with patch("modssc.data_loader.providers.pyg.optional_import") as mock_import:
        mock_module = MagicMock()
        mock_module.NoSeed = NoSeed
        mock_import.return_value = mock_module

        ds = provider.load_canonical(identity, raw_dir=tmp_path)

    assert np.array_equal(ds.train.X, np.arange(8, dtype=np.float64).reshape(4, 2)[:2])
    assert np.array_equal(ds.train.y, np.array([0, 1]))
    assert ds.train.edges is None


def test_to_numpy_ndarray():
    arr = np.array([1, 2, 3])
    assert _to_numpy(arr) is arr


def test_to_numpy_tensor_like():
    mock_tensor = MagicMock()
    mock_tensor.detach.return_value = mock_tensor
    mock_tensor.cpu.return_value = mock_tensor
    mock_tensor.numpy.return_value = np.array([4, 5, 6])

    res = _to_numpy(mock_tensor)
    assert np.array_equal(res, np.array([4, 5, 6]))
    mock_tensor.detach.assert_called_once()
    mock_tensor.cpu.assert_called_once()
    mock_tensor.numpy.assert_called_once()


def test_to_numpy_fallback():
    obj = [7, 8, 9]
    res = _to_numpy(obj)
    assert np.array_equal(res, np.array([7, 8, 9]))


def test_normalize_filter_variants():
    assert _normalize_filter(None) is None
    assert set(_normalize_filter({1, 2})) == {1, 2}
    assert _normalize_filter("x") == ["x"]


def test_pad_labels_variants():
    y_full = np.array([1, 2], dtype=np.int64)
    assert _pad_labels(y_full, 2) is y_full

    y_float = np.array([1.0, 2.0], dtype=np.float64)
    padded_float = _pad_labels(y_float, 4)
    assert padded_float.shape == (4,)
    assert np.isnan(padded_float[2])

    y_bool = np.array([True, False], dtype=np.bool_)
    padded_bool = _pad_labels(y_bool, 3)
    assert padded_bool.dtype == np.int64
    assert padded_bool.tolist() == [1, 0, -1]

    y_obj = np.array(["a"], dtype=object)
    padded_obj = _pad_labels(y_obj, 2)
    assert padded_obj.dtype == object
    assert padded_obj.tolist() == ["a", None]


def test_as_edge_index_variants():
    one_d = np.array([1, 2, 3])
    assert np.array_equal(_as_edge_index(one_d), one_d)

    ei = np.array([[0, 1], [1, 2]])
    assert np.array_equal(_as_edge_index(ei), ei)

    ei_t = np.array([[0, 1], [2, 3], [4, 5]])
    assert np.array_equal(_as_edge_index(ei_t), ei_t.T)

    ei_other = np.arange(9).reshape(3, 3)
    assert np.array_equal(_as_edge_index(ei_other), ei_other)


def test_subset_graph_empty_keep():
    X = np.array([[1], [2]], dtype=np.float64)
    y = np.array([0, 1])
    X_sub, y_sub, edges_sub, masks = _subset_graph(
        X=X, y=y, edge_index=None, masks={}, keep_idx=np.array([], dtype=np.int64)
    )
    assert X_sub.shape == (0, 1)
    assert y_sub.shape == (0,)
    assert edges_sub is None
    assert masks == {}


def test_subset_graph_mask_padding_and_no_edges():
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    y = np.array([0, 1, 2])
    edge_index = np.array([[0, 1], [1, 0]])
    masks = {"train": np.array([True, False])}
    keep_idx = np.array([2], dtype=np.int64)

    X_sub, y_sub, edges_sub, masks_sub = _subset_graph(
        X=X, y=y, edge_index=edge_index, masks=masks, keep_idx=keep_idx
    )

    assert X_sub.shape == (1, 2)
    assert y_sub.shape == (1,)
    assert edges_sub.shape == (2, 0)
    assert masks_sub["train"].shape == (1,)
