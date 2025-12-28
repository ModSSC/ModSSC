from __future__ import annotations

import numpy as np

from modssc.data_loader.storage.files import FileStorage
from modssc.data_loader.types import LoadedDataset, Split


def test_storage_numeric_roundtrip(tmp_path) -> None:
    st = FileStorage()
    ds = LoadedDataset(train=Split(X=np.zeros((3, 2)), y=np.array([0, 1, 0])), meta={"a": 1})
    st.save(tmp_path, ds)
    out = st.load(tmp_path)
    assert out.train.X.shape == (3, 2)
    assert out.meta["a"] == 1


def test_storage_text_roundtrip(tmp_path) -> None:
    st = FileStorage()
    X = np.asarray(["a", "b"], dtype=object)
    y = np.asarray([0, 1])
    ds = LoadedDataset(train=Split(X=X, y=y), meta={})
    st.save(tmp_path, ds)
    out = st.load(tmp_path)
    assert out.train.X.dtype == object
    assert out.train.X.tolist() == ["a", "b"]
