from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_loader.optional import optional_import
from modssc.data_loader.providers.base import BaseProvider
from modssc.data_loader.types import DatasetIdentity, LoadedDataset, Split
from modssc.data_loader.uri import ParsedURI


def _parse_openml_ref(ref: str) -> dict[str, Any]:
    ref = ref.strip()
    if ref.isdigit():
        return {"data_id": int(ref)}
    # simple "name=adult,version=2" style
    parts = [p.strip() for p in ref.split(",") if p.strip()]
    out: dict[str, Any] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    if "data_id" in out:
        out["data_id"] = int(out["data_id"])
    if "version" in out:
        # keep as str or int, sklearn accepts int or 'active'
        vv = out["version"]
        if vv.isdigit():
            out["version"] = int(vv)
    return out


def _normalize_filter(values: Any) -> list[Any] | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple, set, np.ndarray)):
        return list(values)
    return [values]


def _apply_class_filter(
    X: np.ndarray, y: np.ndarray, *, class_filter: list[Any] | None
) -> tuple[np.ndarray, np.ndarray]:
    if class_filter is None:
        return X, y
    mask = np.isin(y, np.asarray(class_filter))
    return X[mask], y[mask]


def _limit_samples(
    X: np.ndarray, y: np.ndarray, *, max_samples: int | None, seed: int | None
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None:
        return X, y
    n = int(y.shape[0])
    max_n = int(max_samples)
    if max_n <= 0 or n == 0:
        return X[:0], y[:0]
    take = min(n, max_n)
    idx = np.arange(n, dtype=np.int64)
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(idx)
    return X[idx[:take]], y[idx[:take]]


class OpenMLProvider(BaseProvider):
    name = "openml"
    required_extra = "openml"

    def resolve(self, parsed: ParsedURI, *, options: Mapping[str, Any]) -> DatasetIdentity:
        base = _parse_openml_ref(parsed.reference)
        # options override base
        cfg = {**base, **dict(options)}
        # canonical uri normalization
        if "data_id" in cfg:
            canonical_uri = f"openml:{int(cfg['data_id'])}"
            dataset_id = str(int(cfg["data_id"]))
        else:
            name = str(cfg.get("name") or cfg.get("dataset") or cfg.get("data_name") or "")
            if not name:
                # keep reference for better error message later
                name = parsed.reference
            canonical_uri = f"openml:name={name}"
            dataset_id = name

        version = cfg.get("version")
        version_str = str(version) if version is not None else None

        resolved_kwargs = {
            # sklearn fetch_openml supports data_home to control caching
            "data_id": cfg.get("data_id"),
            "name": cfg.get("name"),
            "version": cfg.get("version"),
            "as_frame": bool(cfg.get("as_frame", False)),
            "target_column": cfg.get("target_column"),
            "class_filter": _normalize_filter(cfg.get("class_filter")),
            "max_samples": cfg.get("max_samples"),
            "seed": cfg.get("seed"),
        }

        return DatasetIdentity(
            provider=self.name,
            canonical_uri=canonical_uri,
            dataset_id=str(dataset_id),
            version=version_str,
            modality="tabular",
            task=str(cfg.get("task", "classification")),
            required_extra=self.required_extra,
            resolved_kwargs=resolved_kwargs,
        )

    def load_canonical(self, identity: DatasetIdentity, *, raw_dir: Path) -> LoadedDataset:
        sklearn_datasets = optional_import(
            "sklearn.datasets",
            extra=self.required_extra or "openml",
            purpose="OpenML dataset loading via scikit-learn",
        )
        fetch_openml = sklearn_datasets.fetch_openml

        cfg = dict(identity.resolved_kwargs)
        kwargs: dict[str, Any] = {
            "return_X_y": True,
            "as_frame": bool(cfg.get("as_frame", False)),
            "data_home": str(raw_dir),
        }

        if cfg.get("data_id") is not None:
            kwargs["data_id"] = int(cfg["data_id"])
        else:
            name = cfg.get("name") or identity.dataset_id
            kwargs["name"] = str(name)

        # scikit-learn does not accept version=None
        if cfg.get("version") is not None:
            kwargs["version"] = cfg["version"]

        X, y = fetch_openml(**kwargs)

        X_np = _to_numpy(X)
        y_np = _to_numpy(y)

        class_filter = _normalize_filter(cfg.get("class_filter"))
        X_np, y_np = _apply_class_filter(X_np, y_np, class_filter=class_filter)
        X_np, y_np = _limit_samples(
            X_np, y_np, max_samples=cfg.get("max_samples"), seed=cfg.get("seed")
        )

        meta = {
            "provider": "openml",
            "dataset_id": identity.dataset_id,
            "version": identity.version,
        }
        return LoadedDataset(train=Split(X=X_np, y=y_np), test=None, meta=meta)


def _to_numpy(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)
