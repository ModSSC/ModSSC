from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import modssc.data_loader.storage.files as storage_files
from modssc.data_loader.storage.files import FileStorage, _is_str_object_array, _jsonable


def test_is_str_object_array_exception():
    mock_arr = MagicMock()
    mock_arr.dtype = object
    mock_arr.tolist.side_effect = RuntimeError("fail")
    assert _is_str_object_array(mock_arr) is False


def test_is_str_object_array_asarray_exception(monkeypatch):
    arr = np.array(["a", "b"], dtype=object)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(storage_files.np, "asarray", boom)
    assert _is_str_object_array(arr) is False


def test_load_array_unknown_format(tmp_path):
    storage = FileStorage()
    with pytest.raises(ValueError, match="Unknown array format"):
        storage._load_array(tmp_path, {"format": "unknown", "path": "foo"})


def test_jsonable_shape_dtype():
    class TensorLike:
        shape = (2, 2)
        dtype = "float32"

    obj = TensorLike()
    res = _jsonable(obj)
    assert res["__type__"] == "TensorLike"
    assert res["shape"] == [2, 2]
    assert res["dtype"] == "float32"


def test_jsonable_shape_exception():
    class BrokenShapeIter:
        def __iter__(self):
            raise ValueError("Iter failed")

    class BrokenShape:
        @property
        def shape(self):
            return BrokenShapeIter()

        dtype = "int"

    obj = BrokenShape()
    res = _jsonable(obj)
    assert res["__type__"] == "BrokenShape"
    assert res["shape"] is None
    assert res["dtype"] == "int"


def test_jsonable_mapping():
    obj = {"a": 1, "b": [2]}
    res = _jsonable(obj)
    assert res == {"a": 1, "b": [2]}


def test_jsonable_list_small():
    obj = [1, 2, 3]
    res = _jsonable(obj)
    assert res == [1, 2, 3]


def test_jsonable_list_large():
    obj = list(range(100))
    res = _jsonable(obj)
    assert res["__type__"] == "list"
    assert res["len"] == 100


def test_jsonable_unknown_type():
    class Unknown:
        pass

    obj = Unknown()
    res = _jsonable(obj)
    assert res["__type__"] == "Unknown"


def test_storage_roundtrip(tmp_path):
    from modssc.data_loader.types import LoadedDataset, Split

    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    edges_train = np.array([[0, 1], [1, 0]])
    masks_train = {"train": np.array([True, False])}

    train = Split(X=X_train, y=y_train, edges=edges_train, masks=masks_train)

    X_test = np.array(["a", "b"], dtype=object)
    y_test = np.array([0, 1])
    test = Split(X=X_test, y=y_test)

    meta = {"foo": "bar", "path": Path("baz")}

    dataset = LoadedDataset(train=train, test=test, meta=meta)

    storage = FileStorage()
    storage.save(tmp_path, dataset)

    loaded = storage.load(tmp_path)

    assert loaded.meta["foo"] == "bar"
    assert loaded.meta["path"] == "baz"

    np.testing.assert_array_equal(loaded.train.X, X_train)
    np.testing.assert_array_equal(loaded.train.y, y_train)
    np.testing.assert_array_equal(loaded.train.edges, edges_train)
    np.testing.assert_array_equal(loaded.train.masks["train"], masks_train["train"])

    np.testing.assert_array_equal(loaded.test.X, X_test)
    np.testing.assert_array_equal(loaded.test.y, y_test)


def test_jsonable_path():
    p = Path("foo/bar")
    res = _jsonable(p)
    assert res == "foo/bar"


def test_jsonable_ndarray():
    arr = np.array([1, 2])
    res = _jsonable(arr)
    assert res["__type__"] == "ndarray"
    assert res["shape"] == [2]
    assert res["dtype"] == str(arr.dtype)


def test_storage_save_minimal(tmp_path):
    from modssc.data_loader.types import LoadedDataset, Split

    train = Split(X=np.array([1]), y=np.array([0]))
    dataset = LoadedDataset(train=train, test=None, meta={})

    storage = FileStorage()
    storage.save(tmp_path, dataset)

    loaded = storage.load(tmp_path)
    assert loaded.test is None
    assert loaded.train.edges is None
    assert loaded.train.masks is None


def test_jsonable_none():
    assert _jsonable(None) is None


def test_load_array_unknown_format_explicit(tmp_path):
    storage = FileStorage()
    with pytest.raises(ValueError, match="Unknown array format: 'invalid_fmt'"):
        storage._load_array(tmp_path, {"format": "invalid_fmt", "path": "test.npy"})


def test_load_array_npy_mmap(tmp_path, monkeypatch):
    storage = FileStorage()
    path = tmp_path / "test.npy"
    np.save(path, np.array([1, 2]))

    monkeypatch.setenv("MODSSC_DATA_LOADER_MMAP_THRESHOLD", "0")

    with patch("modssc.data_loader.storage.files.np.load", wraps=np.load) as mock_load:
        storage._load_array(tmp_path, {"format": "npy", "path": "test.npy"})
        mock_load.assert_called_with(path, allow_pickle=True, mmap_mode="r")
