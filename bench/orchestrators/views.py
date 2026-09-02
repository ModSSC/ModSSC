from __future__ import annotations

import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset
from modssc.views.api import generate_views
from modssc.views.plan import ViewsPlan
from modssc.views.types import ViewsResult

_LOGGER = logging.getLogger(__name__)


def run(
    dataset: LoadedDataset,
    *,
    plan_dict: Mapping[str, Any],
    seed: int,
    fit_indices: np.ndarray | None,
    cache: bool,
) -> ViewsResult:
    start = perf_counter()
    plan = ViewsPlan.from_dict(plan_dict)
    view_names = [view.name for view in plan.views]
    _LOGGER.info(
        "Views start: seed=%s cache=%s n_views=%s",
        int(seed),
        bool(cache),
        len(view_names),
    )
    _LOGGER.debug(
        "Views plan: names=%s fit_indices=%s",
        view_names,
        None if fit_indices is None else int(fit_indices.shape[0]),
    )
    result = generate_views(
        dataset,
        plan=plan,
        seed=int(seed),
        cache=bool(cache),
        fit_indices=fit_indices,
    )
    cols = {name: int(arr.shape[0]) for name, arr in result.columns.items()}
    _LOGGER.info("Views result: n_views=%s", len(result.views))
    _LOGGER.debug("Views columns: %s", cols)
    _LOGGER.debug("Views meta: %s", dict(result.meta))
    _LOGGER.info("Views stage done: duration_s=%.3f", perf_counter() - start)
    return result
