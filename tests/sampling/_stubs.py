from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Split:
    X: Any
    y: Any
    edges: Any | None = None
    masks: dict[str, Any] | None = None


@dataclass(frozen=True)
class LoadedDataset:
    train: Split
    test: Split | None = None
    meta: dict[str, Any] | None = None


def make_toy_dataset(
    n: int = 100, n_classes: int = 3, seed: int = 0, with_test: bool = False
) -> LoadedDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = rng.integers(0, n_classes, size=(n,), dtype=np.int64)
    train = Split(X=X, y=y)
    test = None
    if with_test:
        Xt = rng.normal(size=(20, 4)).astype(np.float32)
        yt = rng.integers(0, n_classes, size=(20,), dtype=np.int64)
        test = Split(X=Xt, y=yt)
    meta = {"dataset_fingerprint": "deadbeef"}
    return LoadedDataset(train=train, test=test, meta=meta)


def make_graph_dataset(
    n_nodes: int = 50, seed: int = 0, with_official_masks: bool = True
) -> LoadedDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_nodes, 8)).astype(np.float32)
    y = rng.integers(0, 3, size=(n_nodes,), dtype=np.int64)
    edges = np.vstack(
        [rng.integers(0, n_nodes, size=(100,)), rng.integers(0, n_nodes, size=(100,))]
    ).astype(np.int64)

    masks = None
    if with_official_masks:
        train = np.zeros((n_nodes,), dtype=bool)
        val = np.zeros((n_nodes,), dtype=bool)
        test = np.zeros((n_nodes,), dtype=bool)
        train[:20] = True
        val[20:30] = True
        test[30:50] = True
        masks = {"train": train, "val": val, "test": test}

    train_split = Split(X=X, y=y, edges=edges, masks=masks)
    meta = {"dataset_fingerprint": "cafebabe"}
    return LoadedDataset(train=train_split, test=None, meta=meta)
