from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import numpy as np

from modssc.data_loader.manifest import (
    _jsonable,
    build_manifest,
    dataset_summary,
    read_manifest,
    write_manifest,
)
from modssc.data_loader.types import DatasetIdentity, LoadedDataset, Split


def test_manifest_roundtrip_and_jsonable(tmp_path: Path) -> None:
    identity = DatasetIdentity(
        provider="toy",
        canonical_uri="toy:default",
        dataset_id="default",
        version=None,
        modality="tabular",
        task="classification",
        required_extra=None,
        resolved_kwargs={"seed": 0},
    )
    ds = LoadedDataset(
        train=Split(X=np.zeros((2, 3)), y=np.array([0, 1])),
        test=None,
        meta={
            "root": tmp_path,
            "arr": np.array([1, 2, 3]),
            "big_list": list(range(100)),
        },
    )

    man = build_manifest(schema_version=1, fingerprint="abc", identity=identity, dataset=ds)
    assert man.schema_version == 1
    assert man.fingerprint == "abc"
    assert man.identity["canonical_uri"] == "toy:default"

    meta = man.meta
    assert isinstance(meta["big_list"], dict)
    assert meta["big_list"]["__type__"] == "list"
    assert meta["big_list"]["len"] == 100

    path = tmp_path / "manifest.json"
    write_manifest(path, man)
    loaded = read_manifest(path)
    assert loaded.fingerprint == "abc"


def test_dataset_summary_contains_shapes() -> None:
    ds = LoadedDataset(train=Split(X=np.zeros((2, 2)), y=np.array([0, 1])), test=None, meta={})
    summary = dataset_summary(ds)
    assert "train" in summary
    assert summary["train"]["X"]["shape"] == [2, 2]


class MockArrayLike:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class FailsOnIter:
    def __iter__(self):
        raise RuntimeError("Iter failed")


class FailsOnStr:
    def __str__(self):
        raise RuntimeError("Str failed")


class MockShapeIterFails:
    @property
    def shape(self):
        return FailsOnIter()

    @property
    def dtype(self):
        return "float32"


def test_jsonable_large_list():
    large_list = list(range(100))
    res = _jsonable(large_list)
    assert res == {"__type__": "list", "len": 100}


def test_jsonable_small_list():
    small_list = [1, 2, 3]
    res = _jsonable(small_list)
    assert res == [1, 2, 3]


def test_jsonable_unknown_type():
    class Unknown:
        pass

    obj = Unknown()
    res = _jsonable(obj)
    assert res == {"__type__": "Unknown"}


def test_jsonable_array_like():
    obj = MockArrayLike((10, 2), "int32")
    res = _jsonable(obj)
    assert res["__type__"] == "MockArrayLike"
    assert res["shape"] == [10, 2]
    assert res["dtype"] == "int32"


def test_dataset_summary_no_shape_dtype():
    obj = "just a string"
    split = Split(X=obj, y=None)
    ds = LoadedDataset(train=split, test=None)

    summary = dataset_summary(ds)
    x_summary = summary["train"]["X"]
    assert x_summary["type"] == "str"
    assert "shape" not in x_summary
    assert "dtype" not in x_summary


def test_jsonable_array_like_shape_fails():
    obj = MockShapeIterFails()
    res = _jsonable(obj)
    assert res["__type__"] == "MockShapeIterFails"
    assert res["shape"] is None
    assert res["dtype"] == "float32"


def test_dataset_summary_shape_fails():
    mock_X = MagicMock()

    type(mock_X).shape = PropertyMock(return_value=FailsOnIter())
    type(mock_X).dtype = PropertyMock(return_value="float32")

    split = Split(X=mock_X, y=None)
    ds = LoadedDataset(train=split, test=None)

    dataset_summary(ds)
