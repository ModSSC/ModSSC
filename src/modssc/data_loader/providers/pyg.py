from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_loader.optional import optional_import
from modssc.data_loader.providers.base import BaseProvider
from modssc.data_loader.types import DatasetIdentity, LoadedDataset, Split
from modssc.data_loader.uri import ParsedURI


class PyGProvider(BaseProvider):
    name = "pyg"
    required_extra = "graph"

    def resolve(self, parsed: ParsedURI, *, options: Mapping[str, Any]) -> DatasetIdentity:
        ref = parsed.reference.strip()

        # Convention: "ClassName/DatasetName" or just "ClassName"
        if "/" in ref:
            cls_name, arg = ref.split("/", 1)
            kwargs = {"name": arg.strip()}
        else:
            cls_name = ref
            kwargs = {}

        # Allow overriding via options
        if "dataset_class" in options:
            cls_name = options["dataset_class"]

        max_nodes = options.get("max_nodes")
        class_filter = _normalize_filter(options.get("class_filter"))
        seed = options.get("seed")

        # Construct a canonical URI that reflects the class and name
        canonical_suffix = f"{cls_name}/{kwargs.get('name', '')}".rstrip("/")

        resolved_kwargs: dict[str, Any] = {
            "dataset_class": cls_name,
            "dataset_kwargs": kwargs,
        }
        if max_nodes is not None:
            resolved_kwargs["max_nodes"] = max_nodes
        if class_filter is not None:
            resolved_kwargs["class_filter"] = class_filter
        if seed is not None:
            resolved_kwargs["seed"] = seed

        return DatasetIdentity(
            provider=self.name,
            canonical_uri=f"pyg:{canonical_suffix}",
            dataset_id=cls_name,
            version=None,
            modality="graph",
            task="node_classification",  # Default assumption, might need refinement
            required_extra=self.required_extra,
            resolved_kwargs=resolved_kwargs,
        )

    def load_canonical(self, identity: DatasetIdentity, *, raw_dir: Path) -> LoadedDataset:
        pyg_datasets = optional_import(
            "torch_geometric.datasets",
            extra=self.required_extra or "graph",
            purpose="torch_geometric dataset loading",
        )

        cls_name = identity.resolved_kwargs["dataset_class"]
        kwargs = identity.resolved_kwargs["dataset_kwargs"]

        if not hasattr(pyg_datasets, cls_name):
            raise ValueError(
                f"PyG dataset class '{cls_name}' not found in torch_geometric.datasets"
            )

        DatasetClass = getattr(pyg_datasets, cls_name)

        root = raw_dir / "source"
        root.mkdir(parents=True, exist_ok=True)

        # Instantiate the dataset
        # Check if 'root' is in the signature (some datasets like KarateClub don't take root)
        sig = inspect.signature(DatasetClass.__init__)

        try:
            if "root" in sig.parameters:
                dataset = DatasetClass(root=str(root), **kwargs)
            else:
                dataset = DatasetClass(**kwargs)
        except TypeError as e:
            raise ValueError(f"Failed to instantiate {cls_name} with kwargs={kwargs}: {e}") from e

        if len(dataset) == 0:
            raise ValueError(f"PyG dataset {cls_name} is empty.")

        data = dataset[0]

        X = _to_numpy(data.x) if hasattr(data, "x") and data.x is not None else None
        y = _to_numpy(data.y) if hasattr(data, "y") and data.y is not None else None
        edges = (
            _to_numpy(data.edge_index)
            if hasattr(data, "edge_index") and data.edge_index is not None
            else None
        )

        masks: dict[str, Any] = {}
        if hasattr(data, "train_mask"):
            masks["train"] = _to_numpy(data.train_mask)
        if hasattr(data, "val_mask"):
            masks["val"] = _to_numpy(data.val_mask)
        if hasattr(data, "test_mask"):
            masks["test"] = _to_numpy(data.test_mask)

        max_nodes = identity.resolved_kwargs.get("max_nodes")
        class_filter = _normalize_filter(identity.resolved_kwargs.get("class_filter"))
        seed = identity.resolved_kwargs.get("seed")

        n_nodes = int(X.shape[0]) if X is not None else (int(y.shape[0]) if y is not None else 0)
        if y is not None and int(y.shape[0]) < n_nodes:
            y = _pad_labels(y, n_nodes)
        idx = np.arange(n_nodes, dtype=np.int64)
        if class_filter is not None and y is not None:
            idx = np.where(np.isin(y, np.asarray(class_filter)))[0].astype(np.int64)

        if max_nodes is not None and idx.size > int(max_nodes):
            if seed is not None:
                rng = np.random.default_rng(int(seed))
                rng.shuffle(idx)
            idx = np.sort(idx[: int(max_nodes)])

        X, y, edges, masks = _subset_graph(
            X=X,
            y=y,
            edge_index=edges,
            masks=masks,
            keep_idx=idx,
        )

        train = Split(X=X, y=y, edges=edges, masks=masks if masks else None)

        meta = {
            "provider": "pyg",
            "dataset_class": cls_name,
            "dataset_kwargs": kwargs,
            "num_graphs": len(dataset),
        }
        return LoadedDataset(train=train, test=None, meta=meta)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    obj = x
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "numpy"):
        return np.asarray(obj.numpy())
    return np.asarray(obj)


def _normalize_filter(values: Any) -> list[Any] | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple, set, np.ndarray)):
        return list(values)
    return [values]


def _pad_labels(y: np.ndarray, n_nodes: int) -> np.ndarray:
    if int(y.shape[0]) >= n_nodes:
        return y
    y_arr = _to_numpy(y).reshape(-1)
    if np.issubdtype(y_arr.dtype, np.floating):
        pad_value: Any = np.nan
        dtype = y_arr.dtype
    elif np.issubdtype(y_arr.dtype, np.integer) or y_arr.dtype == np.bool_:
        pad_value = -1
        dtype = np.int64 if y_arr.dtype == np.bool_ else y_arr.dtype
        if y_arr.dtype == np.bool_:
            y_arr = y_arr.astype(dtype, copy=False)
    else:
        pad_value = None
        dtype = object
        y_arr = y_arr.astype(dtype, copy=False)
    padded = np.full((n_nodes,), pad_value, dtype=dtype)
    padded[: y_arr.shape[0]] = y_arr
    return padded


def _as_edge_index(edge_index: Any) -> np.ndarray:
    ei = _to_numpy(edge_index)
    if ei.ndim != 2:
        return ei
    if ei.shape[0] == 2:
        return ei
    if ei.shape[1] == 2:
        return ei.T
    return ei


def _subset_graph(
    *,
    X: np.ndarray | None,
    y: np.ndarray | None,
    edge_index: np.ndarray | None,
    masks: dict[str, Any],
    keep_idx: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if keep_idx.size == 0:
        return X[:0] if X is not None else None, y[:0] if y is not None else None, None, {}

    n_nodes = int(
        max(
            int(X.shape[0]) if X is not None else 0,
            int(y.shape[0]) if y is not None else 0,
        )
    )
    mapping = -np.ones((n_nodes,), dtype=np.int64)
    mapping[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)

    new_edge_index = None
    if edge_index is not None and edge_index.size:
        ei = _as_edge_index(edge_index)
        src = ei[0].astype(np.int64, copy=False)
        dst = ei[1].astype(np.int64, copy=False)
        keep_edge = (mapping[src] >= 0) & (mapping[dst] >= 0)
        if bool(keep_edge.any()):
            new_edge_index = np.stack([mapping[src[keep_edge]], mapping[dst[keep_edge]]], axis=0)
        else:
            new_edge_index = np.zeros((2, 0), dtype=np.int64)

    new_masks: dict[str, Any] = {}
    for key, value in masks.items():
        arr = _to_numpy(value).astype(bool, copy=False).reshape(-1)
        if arr.shape[0] < n_nodes:
            padded = np.zeros((n_nodes,), dtype=bool)
            padded[: arr.shape[0]] = arr
            arr = padded
        new_masks[key] = arr[keep_idx]

    X_sub = X[keep_idx] if X is not None else None
    y_sub = y[keep_idx] if y is not None else None
    return X_sub, y_sub, new_edge_index, new_masks
