from __future__ import annotations

import contextlib
import gzip
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_loader.numpy_adapter import to_numpy
from modssc.data_loader.types import LoadedDataset, Split


def _is_str_object_array(arr: np.ndarray) -> bool:
    if arr.dtype != object:
        return False
    try:
        items = arr.tolist()
    except Exception:
        return False
    # conservative: require every element to be str
    try:
        return all(isinstance(x, str) for x in np.asarray(items, dtype=object).flat)
    except Exception:
        return False


def _write_jsonl_gz(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for s in items:
            f.write(json.dumps(s, ensure_ascii=False))
            f.write("\n")


def _read_jsonl_gz(path: Path) -> list[str]:
    out: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


@dataclass(frozen=True)
class FileStorage:
    """Simple processed storage.

    - numeric arrays are stored as .npy
    - text arrays (object arrays of str) are stored as jsonl.gz
    - other object arrays are stored as .npy (uses numpy pickle internally)

    Security note
    Loading object arrays from untrusted cache content is unsafe.
    """

    def save(self, processed_dir: Path, dataset: LoadedDataset) -> None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        layout: dict[str, Any] = {"version": 1, "splits": {}}

        layout["splits"]["train"] = self._save_split(processed_dir, "train", dataset.train)
        if dataset.test is not None:
            layout["splits"]["test"] = self._save_split(processed_dir, "test", dataset.test)

        meta_path = processed_dir / "meta.json"
        meta_path.write_text(
            json.dumps(_jsonable_mapping(dataset.meta), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        layout_path = processed_dir / "layout.json"
        layout_path.write_text(
            json.dumps(layout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def load(self, processed_dir: Path) -> LoadedDataset:
        layout = json.loads((processed_dir / "layout.json").read_text(encoding="utf-8"))
        train = self._load_split(processed_dir, "train", layout["splits"]["train"])
        test = None
        if "test" in layout["splits"]:
            test = self._load_split(processed_dir, "test", layout["splits"]["test"])

        meta = json.loads((processed_dir / "meta.json").read_text(encoding="utf-8"))
        return LoadedDataset(train=train, test=test, meta=meta)

    def _save_split(self, root: Path, name: str, split: Split) -> Mapping[str, Any]:
        info: dict[str, Any] = {}

        X = to_numpy(split.X, allow_object=True)
        y = to_numpy(split.y, allow_object=True)

        info["X"] = self._save_array(root, f"{name}_X", X)
        info["y"] = self._save_array(root, f"{name}_y", y)

        if split.edges is not None:
            edges = to_numpy(split.edges, allow_object=True)
            info["edges"] = self._save_array(root, f"{name}_edges", edges)

        if split.masks is not None:
            masks_info: dict[str, Any] = {}
            for k, v in split.masks.items():
                arr = to_numpy(v, dtype=bool, allow_object=True)
                masks_info[k] = self._save_array(root, f"{name}_mask_{k}", arr)
            info["masks"] = masks_info

        return info

    def _load_split(self, root: Path, name: str, info: Mapping[str, Any]) -> Split:
        X = self._load_array(root, info["X"])
        y = self._load_array(root, info["y"])

        edges = None
        if "edges" in info:
            edges = self._load_array(root, info["edges"])

        masks = None
        if "masks" in info:
            masks = {k: self._load_array(root, v) for k, v in info["masks"].items()}

        return Split(X=X, y=y, edges=edges, masks=masks)

    def _save_array(self, root: Path, stem: str, arr: np.ndarray) -> Mapping[str, Any]:
        if _is_str_object_array(arr):
            path = root / f"{stem}.jsonl.gz"
            _write_jsonl_gz(path, arr.tolist())
            return {"format": "jsonl.gz", "path": path.name}

        path = root / f"{stem}.npy"
        np.save(path, arr, allow_pickle=bool(arr.dtype == object))
        return {"format": "npy", "path": path.name}

    def _load_array(self, root: Path, info: Mapping[str, Any]) -> np.ndarray:
        fmt = info["format"]
        path = root / info["path"]
        if fmt == "jsonl.gz":
            items = _read_jsonl_gz(path)
            return np.asarray(items, dtype=object)
        if fmt == "npy":
            mmap_mode = None
            # Default to 64MB threshold to match preprocess cache
            threshold = int(
                os.environ.get("MODSSC_DATA_LOADER_MMAP_THRESHOLD", str(64 * 1024 * 1024))
            )
            with contextlib.suppress(OSError):
                if path.stat().st_size >= threshold:
                    mmap_mode = "r"
            return np.load(path, allow_pickle=True, mmap_mode=mmap_mode)
        raise ValueError(f"Unknown array format: {fmt!r}")


def _jsonable_mapping(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k): _jsonable(v) for k, v in meta.items()}


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return {"__type__": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        try:
            shape = list(obj.shape)
        except Exception:
            shape = None
        return {
            "__type__": type(obj).__name__,
            "shape": shape,
            "dtype": str(getattr(obj, "dtype", "")),
        }
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if len(obj) > 50:
            return {"__type__": type(obj).__name__, "len": len(obj)}
        return [_jsonable(v) for v in obj]
    return {"__type__": type(obj).__name__}
