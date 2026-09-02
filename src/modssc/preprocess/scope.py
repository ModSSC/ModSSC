"""Native population scopes for fitting preprocessing transformations."""

from __future__ import annotations

from typing import Literal

import numpy as np

from modssc.data_loader.types import LoadedDataset
from modssc.sampling.result import SamplingResult

FitScope = Literal["train", "train_labeled", "train_unlabeled", "val"]


def resolve_fit_indices(
    *,
    dataset: LoadedDataset,
    sampling: SamplingResult,
    fit_on: FitScope | None,
) -> np.ndarray | None:
    """Resolve a preprocessing fit scope in the sampling result's index space."""

    del dataset  # retained in the public contract for future provider-aware scopes
    if fit_on is None:
        return None
    if sampling.is_graph():
        mask_names = {
            "train": "train",
            "train_labeled": "labeled",
            "train_unlabeled": "unlabeled",
            "val": "val",
        }
        try:
            mask = sampling.masks[mask_names[fit_on]]
        except KeyError as exc:
            raise ValueError(f"Unsupported fit_on for graph sampling: {fit_on!r}") from exc
        return np.where(np.asarray(mask, dtype=bool))[0].astype(np.int64, copy=False)
    try:
        return np.asarray(sampling.indices[fit_on], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unsupported fit_on: {fit_on!r}") from exc


__all__ = ["FitScope", "resolve_fit_indices"]
