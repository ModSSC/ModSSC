from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset

from .plan import ViewsPlan


@dataclass(frozen=True)
class ViewsResult:
    """Result of `generate_views`.

    Attributes
    ----------
    views:
        Mapping of view name -> dataset where each split's `.X` is the view-specific feature matrix.
    columns:
        Mapping of view name -> selected column indices (sorted, unique).
    seed:
        Global seed used for any stochastic view operations (e.g. random column selection).
    plan:
        The input plan (validated).
    meta:
        Arbitrary metadata.
    """

    views: dict[str, LoadedDataset]
    columns: dict[str, np.ndarray]
    seed: int
    plan: ViewsPlan
    meta: dict[str, Any]
