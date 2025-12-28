from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset, Split


def to_numpy(value: Any, *, dtype: Any | None = None, allow_object: bool = True) -> np.ndarray:
    """Best effort conversion to numpy without importing heavy frameworks."""
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False) if dtype is not None else value

    if hasattr(value, "to_numpy"):
        arr = value.to_numpy()
        return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)

    obj = value
    if hasattr(obj, "detach"):
        try:
            obj = obj.detach()
        except Exception:
            obj = value
    if hasattr(obj, "cpu"):
        try:
            obj = obj.cpu()
        except Exception:
            obj = obj
    if hasattr(obj, "numpy"):
        try:
            arr = obj.numpy()
            return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)
        except Exception:
            pass

    try:
        return np.asarray(obj, dtype=dtype)
    except Exception:
        if allow_object:
            arr = np.empty((1,), dtype=object)
            arr[0] = obj
            return arr
        raise


def split_to_numpy(split: Split, *, allow_object: bool = True) -> Split:
    edges = None if split.edges is None else to_numpy(split.edges, allow_object=allow_object)
    masks = None
    if split.masks is not None:
        masks = {
            k: to_numpy(v, dtype=bool, allow_object=allow_object) for k, v in split.masks.items()
        }
    return Split(
        X=to_numpy(split.X, allow_object=allow_object),
        y=to_numpy(split.y, allow_object=allow_object),
        edges=edges,
        masks=masks,
    )


def dataset_to_numpy(dataset: LoadedDataset, *, allow_object: bool = True) -> LoadedDataset:
    train = split_to_numpy(dataset.train, allow_object=allow_object)
    test = None if dataset.test is None else split_to_numpy(dataset.test, allow_object=allow_object)
    return replace(dataset, train=train, test=test)
