from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_loader.optional import optional_import
from modssc.data_loader.providers.base import BaseProvider
from modssc.data_loader.types import DatasetIdentity, LoadedDataset, Split
from modssc.data_loader.uri import ParsedURI


def _split_name_version(ref: str) -> tuple[str, str | None]:
    ref = ref.strip()
    if "/" in ref:
        name, version = ref.split("/", 1)
        name = name.strip()
        version = version.strip() or None
        return name, version
    return ref, None


class TFDSProvider(BaseProvider):
    name = "tfds"
    required_extra = "tfds"

    def resolve(self, parsed: ParsedURI, *, options: Mapping[str, Any]) -> DatasetIdentity:
        name, version = _split_name_version(parsed.reference)
        resolved_kwargs = {
            "name": name,
            "version": version,
            "as_supervised": bool(options.get("as_supervised", True)),
        }
        canonical_uri = f"tfds:{name}" + (f"/{version}" if version else "")
        return DatasetIdentity(
            provider=self.name,
            canonical_uri=canonical_uri,
            dataset_id=name,
            version=version,
            modality=str(options.get("modality", "unknown")),
            task=str(options.get("task", "classification")),
            required_extra=self.required_extra,
            resolved_kwargs=resolved_kwargs,
        )

    def load_canonical(self, identity: DatasetIdentity, *, raw_dir: Path) -> LoadedDataset:
        tfds = optional_import(
            "tensorflow_datasets",
            extra=self.required_extra or "tfds",
            purpose="TFDS dataset loading",
        )

        cfg = dict(identity.resolved_kwargs)
        name = str(cfg["name"])
        version = cfg.get("version")
        as_supervised = bool(cfg.get("as_supervised", True))

        data_dir = raw_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        full_name = f"{name}:{version}" if version else name

        train_ds = tfds.load(
            full_name, split="train", as_supervised=as_supervised, data_dir=str(data_dir)
        )
        # test split might not exist
        try:
            test_ds = tfds.load(
                full_name, split="test", as_supervised=as_supervised, data_dir=str(data_dir)
            )
        except Exception:
            test_ds = None

        X_train, y_train = _materialize_tfds(train_ds)
        train = Split(X=X_train, y=y_train)

        test = None
        if test_ds is not None:
            X_test, y_test = _materialize_tfds(test_ds)
            test = Split(X=X_test, y=y_test)

        meta = {"provider": "tfds", "name": name, "version": version}
        return LoadedDataset(train=train, test=test, meta=meta)


def _materialize_tfds(ds: Any) -> tuple[np.ndarray, np.ndarray]:
    # tfds.as_numpy is the standard path, but keep it optional for stubs.
    # In real TFDS, iter yields tf.Tensors, but tests will stub simple tuples.
    items = list(ds.as_numpy()) if hasattr(ds, "as_numpy") else list(ds)

    X_list: list[Any] = []
    y_list: list[Any] = []
    for x, y in items:
        X_list.append(x)
        y_list.append(y)
    return np.asarray(X_list, dtype=object), np.asarray(y_list)
