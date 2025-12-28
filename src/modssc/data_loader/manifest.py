from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_loader.types import DatasetIdentity, LoadedDataset


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def python_version() -> str:
    return sys.version.split()[0]


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


def dataset_summary(ds: LoadedDataset) -> dict[str, Any]:
    def shape_dtype(x: Any) -> dict[str, Any]:
        if x is None:
            return {"type": "None"}
        out: dict[str, Any] = {"type": type(x).__name__}
        if hasattr(x, "shape"):
            with contextlib.suppress(Exception):
                out["shape"] = [int(s) for s in x.shape]
        if hasattr(x, "dtype"):
            with contextlib.suppress(Exception):
                out["dtype"] = str(x.dtype)
        return out

    def split_summary(split) -> dict[str, Any]:
        return {
            "X": shape_dtype(split.X),
            "y": shape_dtype(split.y),
            "edges": None if split.edges is None else shape_dtype(split.edges),
            "masks": None
            if split.masks is None
            else {k: shape_dtype(v) for k, v in split.masks.items()},
        }

    return {
        "train": split_summary(ds.train),
        "test": None if ds.test is None else split_summary(ds.test),
    }


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    fingerprint: str
    created_at: str
    identity: Mapping[str, Any]
    dataset: Mapping[str, Any]
    meta: Mapping[str, Any]
    environment: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "identity": dict(self.identity),
            "dataset": dict(self.dataset),
            "meta": dict(self.meta),
            "environment": dict(self.environment),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        obj = json.loads(text)
        return cls(
            schema_version=int(obj["schema_version"]),
            fingerprint=str(obj["fingerprint"]),
            created_at=str(obj["created_at"]),
            identity=obj["identity"],
            dataset=obj["dataset"],
            meta=obj.get("meta", {}),
            environment=obj.get("environment", {}),
        )


def build_manifest(
    *,
    schema_version: int,
    fingerprint: str,
    identity: DatasetIdentity,
    dataset: LoadedDataset,
) -> Manifest:
    env = {
        "python": python_version(),
    }
    return Manifest(
        schema_version=schema_version,
        fingerprint=fingerprint,
        created_at=utc_now_iso(),
        identity=identity.as_dict(),
        dataset=dataset_summary(dataset),
        meta=_jsonable(dataset.meta)
        if isinstance(dataset.meta, Mapping)
        else {"__type__": type(dataset.meta).__name__},
        environment=env,
    )


def write_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json() + "\n", encoding="utf-8")


def read_manifest(path: Path) -> Manifest:
    return Manifest.from_json(path.read_text(encoding="utf-8"))
