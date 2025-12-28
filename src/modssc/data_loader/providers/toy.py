from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_loader.providers.base import BaseProvider
from modssc.data_loader.types import DatasetIdentity, LoadedDataset, Split
from modssc.data_loader.uri import ParsedURI


class ToyProvider(BaseProvider):
    name = "toy"
    required_extra = None

    def resolve(self, parsed: ParsedURI, *, options: Mapping[str, Any]) -> DatasetIdentity:
        # canonical toy dataset, options can adjust sizes (still canonical in the sense of deterministic generation)
        resolved = {
            "n_samples": int(options.get("n_samples", 64)),
            "n_features": int(options.get("n_features", 4)),
            "n_classes": int(options.get("n_classes", 3)),
            "seed": int(options.get("seed", 0)),
            "test": bool(options.get("official_test", True)),
            "test_size": float(options.get("test_size", 0.25)),
        }
        return DatasetIdentity(
            provider=self.name,
            canonical_uri="toy:default",
            dataset_id="toy",
            version="1",
            modality="tabular",
            task="classification",
            required_extra=None,
            resolved_kwargs=resolved,
        )

    def load_canonical(self, identity: DatasetIdentity, *, raw_dir: Path) -> LoadedDataset:
        cfg = dict(identity.resolved_kwargs)
        n_samples = int(cfg["n_samples"])
        n_features = int(cfg["n_features"])
        n_classes = int(cfg["n_classes"])
        seed = int(cfg["seed"])
        has_test = bool(cfg["test"])
        test_size = float(cfg["test_size"])

        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
        y = rng.integers(low=0, high=n_classes, size=(n_samples,), dtype=np.int64)

        if has_test:
            n_test = int(round(n_samples * test_size))
            n_test = max(0, min(n_samples, n_test))
        else:
            n_test = 0

        n_train = n_samples - n_test
        train = Split(X=X[:n_train], y=y[:n_train])
        test = Split(X=X[n_train:], y=y[n_train:]) if n_test else None

        meta = {
            "toy": True,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
        }
        return LoadedDataset(train=train, test=test, meta=meta)
