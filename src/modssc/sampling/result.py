from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modssc.sampling.errors import SamplingValidationError


@dataclass(frozen=True)
class SamplingResult:
    """Sampling result with indices (inductive) or masks (graph transductive).

    Indices keys (typical):
    - train, val, test
    - train_labeled, train_unlabeled

    Refs indicate the base split each index array refers to:
    - "train" means indices are relative to dataset.train
    - "test" means indices are relative to dataset.test
    - "nodes" means graph nodes

    Masks keys (graph):
    - train, val, test, labeled, unlabeled
    """

    schema_version: int
    created_at: str
    dataset_fingerprint: str
    split_fingerprint: str
    plan: Mapping[str, Any]

    indices: Mapping[str, np.ndarray] = field(default_factory=dict)
    refs: Mapping[str, str] = field(default_factory=dict)
    masks: Mapping[str, np.ndarray] = field(default_factory=dict)

    stats: Mapping[str, Any] = field(default_factory=dict)

    def is_graph(self) -> bool:
        return bool(self.masks)

    def validate(self, *, n_train: int, n_test: int | None, n_nodes: int | None) -> None:
        if self.is_graph():
            self._validate_graph(n_nodes=n_nodes)
        else:
            self._validate_inductive(n_train=n_train, n_test=n_test)

    def _validate_inductive(self, *, n_train: int, n_test: int | None) -> None:
        def _check_idx(name: str, base: str, idx: np.ndarray) -> None:
            if idx.dtype.kind not in ("i", "u"):
                raise SamplingValidationError(f"{name} indices must be integers, got {idx.dtype}")
            if idx.size == 0:
                return
            max_ok = n_train if base == "train" else n_test
            if max_ok is None:
                raise SamplingValidationError("n_test is required when base='test'")
            if idx.min() < 0 or idx.max() >= max_ok:
                raise SamplingValidationError(f"{name} has out-of-range indices for base={base!r}")
            if np.unique(idx).size != idx.size:
                raise SamplingValidationError(f"{name} contains duplicate indices")

        for name, idx in self.indices.items():
            base = self.refs.get(name, "train")
            _check_idx(name, base, idx)

        # core invariants
        train = self.indices.get("train")
        val = self.indices.get("val")
        test = self.indices.get("test")
        if train is None:
            raise SamplingValidationError("Missing train indices")
        if val is None:
            raise SamplingValidationError("Missing val indices")
        if test is None:
            raise SamplingValidationError("Missing test indices (may be empty array)")

        if self.refs.get("train", "train") != "train" or self.refs.get("val", "train") != "train":
            raise SamplingValidationError("train and val indices must be relative to dataset.train")

        # disjointness of train/val in same base
        if np.intersect1d(train, val).size:
            raise SamplingValidationError("train and val overlap")

        # test disjointness only if same base
        if self.refs.get("test", "train") == "train" and (
            np.intersect1d(train, test).size or np.intersect1d(val, test).size
        ):
            raise SamplingValidationError("test overlaps with train or val")

        labeled = self.indices.get("train_labeled")
        unlabeled = self.indices.get("train_unlabeled")
        if labeled is None or unlabeled is None:
            raise SamplingValidationError("Missing labeled/unlabeled indices")
        if np.intersect1d(labeled, unlabeled).size:
            raise SamplingValidationError("labeled and unlabeled overlap")
        if not np.array_equal(np.sort(np.concatenate([labeled, unlabeled])), np.sort(train)):
            raise SamplingValidationError("labeled + unlabeled must cover train exactly")

    def _validate_graph(self, *, n_nodes: int | None) -> None:
        if n_nodes is None:
            raise SamplingValidationError("n_nodes is required to validate graph masks")
        required = {"train", "val", "test", "labeled", "unlabeled"}
        if set(self.masks.keys()) != required:
            raise SamplingValidationError(f"Graph masks must have keys {sorted(required)}")

        for k, m in self.masks.items():
            if m.dtype != bool:
                raise SamplingValidationError(f"Mask {k!r} must be bool, got {m.dtype}")
            if m.shape != (n_nodes,):
                raise SamplingValidationError(
                    f"Mask {k!r} must have shape ({n_nodes},), got {m.shape}"
                )

        train = self.masks["train"]
        val = self.masks["val"]
        test = self.masks["test"]
        labeled = self.masks["labeled"]
        unlabeled = self.masks["unlabeled"]

        if (labeled & ~train).any():
            raise SamplingValidationError("labeled_mask must be subset of train_mask")
        if not np.array_equal(unlabeled, train & ~labeled):
            raise SamplingValidationError("unlabeled_mask must equal train_mask & ~labeled_mask")

        # train/val/test should be disjoint
        if (train & val).any() or (train & test).any() or (val & test).any():
            raise SamplingValidationError("train/val/test masks must be disjoint")

    @property
    def train_idx(self) -> np.ndarray:
        if self.is_graph():
            return np.where(self.masks["train"])[0]
        return self.indices.get("train", np.array([]))

    @property
    def val_idx(self) -> np.ndarray:
        if self.is_graph():
            return np.where(self.masks["val"])[0]
        return self.indices.get("val", np.array([]))

    @property
    def test_idx(self) -> np.ndarray:
        if self.is_graph():
            return np.where(self.masks["test"])[0]
        return self.indices.get("test", np.array([]))

    @property
    def labeled_idx(self) -> np.ndarray:
        if self.is_graph():
            return np.where(self.masks["labeled"])[0]
        return self.indices.get("train_labeled", np.array([]))

    @property
    def unlabeled_idx(self) -> np.ndarray:
        if self.is_graph():
            return np.where(self.masks["unlabeled"])[0]
        return self.indices.get("train_unlabeled", np.array([]))
