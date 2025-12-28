from __future__ import annotations

from pathlib import Path

import numpy as np

from modssc.data_loader.storage.files import FileStorage
from modssc.data_loader.types import LoadedDataset, Split


def test_storage_roundtrip_edges_masks_and_meta(tmp_path: Path) -> None:
    store = FileStorage()

    ds = LoadedDataset(
        train=Split(
            X=np.array([[1.0], [2.0]]),
            y=np.array([0, 1]),
            edges=np.array([[0, 1], [1, 0]]),
            masks={"train": np.array([True, False]), "test": np.array([False, True])},
        ),
        test=None,
        meta={"root": tmp_path, "arr": np.array([1, 2, 3])},
    )

    store.save(tmp_path, ds)
    loaded = store.load(tmp_path)

    assert loaded.train.edges is not None
    np.testing.assert_array_equal(loaded.train.edges, ds.train.edges)

    assert loaded.train.masks is not None
    np.testing.assert_array_equal(loaded.train.masks["train"], np.array([True, False]))

    assert isinstance(loaded.meta["root"], str)
    assert isinstance(loaded.meta["arr"], dict)
    assert loaded.meta["arr"]["__type__"] == "ndarray"


def test_storage_roundtrip_object_array_non_string(tmp_path: Path) -> None:
    store = FileStorage()
    X = np.empty((2,), dtype=object)
    X[0] = np.array([1, 2, 3])
    X[1] = np.array([4, 5])

    ds = LoadedDataset(train=Split(X=X, y=np.array([0, 1])), test=None, meta={})
    store.save(tmp_path, ds)
    loaded = store.load(tmp_path)

    assert loaded.train.X.dtype == object
    assert isinstance(loaded.train.X[0], np.ndarray)
    assert loaded.train.X[0].tolist() == [1, 2, 3]
