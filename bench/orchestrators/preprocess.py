from __future__ import annotations

import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset
from modssc.preprocess import preprocess
from modssc.preprocess.plan import PreprocessPlan
from modssc.preprocess.types import PreprocessResult

_LOGGER = logging.getLogger(__name__)


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(s) for s in shape)
    except Exception:
        return None


def run(
    dataset: LoadedDataset,
    *,
    plan_dict: Mapping[str, Any],
    seed: int,
    fit_indices: np.ndarray | None,
    cache: bool,
    cache_dir: str | None = None,
) -> PreprocessResult:
    start = perf_counter()
    plan = PreprocessPlan.from_dict(plan_dict)
    step_ids = [step.step_id for step in plan.steps if step.enabled]
    _LOGGER.info(
        "Preprocess start: seed=%s cache=%s n_steps=%s fit_indices=%s",
        int(seed),
        bool(cache),
        len(step_ids),
        None if fit_indices is None else int(fit_indices.shape[0]),
    )
    _LOGGER.debug(
        "Preprocess plan: output_key=%s steps=%s",
        plan.output_key,
        step_ids,
    )
    result = preprocess(
        dataset,
        plan,
        seed=int(seed),
        fit_indices=fit_indices,
        cache=bool(cache),
        cache_dir=cache_dir,
    )
    _LOGGER.info(
        "Preprocess result: fingerprint=%s plan_fingerprint=%s cache_dir=%s skipped=%s",
        result.preprocess_fingerprint,
        result.plan.fingerprint,
        result.cache_dir,
        [step.step_id for step in result.plan.skipped],
    )
    _LOGGER.debug(
        "Preprocess output shapes: train_X=%s test_X=%s",
        _shape_of(result.dataset.train.X),
        _shape_of(result.dataset.test.X) if result.dataset.test is not None else None,
    )
    _LOGGER.info("Preprocess stage done: duration_s=%.3f", perf_counter() - start)
    return result
