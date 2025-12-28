from __future__ import annotations

import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset
from modssc.views.api import generate_views
from modssc.views.plan import ColumnSelectSpec, ViewSpec, ViewsPlan
from modssc.views.types import ViewsResult

from .preprocess import _plan_from_dict as _preprocess_plan_from_dict

_LOGGER = logging.getLogger(__name__)


def _column_spec_from_dict(obj: Mapping[str, Any]) -> ColumnSelectSpec:
    return ColumnSelectSpec(
        mode=str(obj.get("mode", "all")),
        indices=tuple(int(i) for i in obj.get("indices", []) or ()),
        fraction=float(obj.get("fraction", 0.5)),
        complement_of=str(obj.get("complement_of")) if obj.get("complement_of") else None,
        seed_offset=int(obj.get("seed_offset", 0)),
    )


def _view_spec_from_dict(obj: Mapping[str, Any]) -> ViewSpec:
    name = str(obj.get("name", ""))
    if not name:
        raise ValueError("Each view must define 'name'")

    preprocess = None
    if obj.get("preprocess") is not None:
        preprocess = _preprocess_plan_from_dict(obj["preprocess"])

    columns = None
    if obj.get("columns") is not None:
        columns = _column_spec_from_dict(obj["columns"])

    meta = obj.get("meta")
    if meta is not None and not isinstance(meta, Mapping):
        raise ValueError("view.meta must be a mapping when provided")

    return ViewSpec(
        name=name, preprocess=preprocess, columns=columns, meta=dict(meta) if meta else None
    )


def _plan_from_dict(obj: Mapping[str, Any]) -> ViewsPlan:
    views_raw = obj.get("views", [])
    if not isinstance(views_raw, list):
        raise ValueError("views.plan.views must be a list")
    views = [_view_spec_from_dict(v) for v in views_raw]
    plan = ViewsPlan(views=tuple(views))
    plan.validate()
    return plan


def run(
    dataset: LoadedDataset,
    *,
    plan_dict: Mapping[str, Any],
    seed: int,
    fit_indices: np.ndarray | None,
    cache: bool,
) -> ViewsResult:
    start = perf_counter()
    plan = _plan_from_dict(plan_dict)
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
