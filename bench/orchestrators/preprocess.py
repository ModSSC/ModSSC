from __future__ import annotations

import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np

from modssc.data_loader.types import LoadedDataset
from modssc.preprocess import preprocess
from modssc.preprocess.plan import PreprocessPlan, StepConfig
from modssc.preprocess.types import PreprocessResult
from modssc.sampling.result import SamplingResult

_LOGGER = logging.getLogger(__name__)


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(s) for s in shape)
    except Exception:
        return None


def _plan_from_dict(obj: Mapping[str, Any]) -> PreprocessPlan:
    output_key = str(obj.get("output_key", "features.X"))
    steps_raw = obj.get("steps", [])
    if not isinstance(steps_raw, list):
        raise ValueError("preprocess.plan.steps must be a list")

    steps: list[StepConfig] = []
    for item in steps_raw:
        if not isinstance(item, Mapping):
            raise ValueError("Each preprocess step must be a mapping")
        step_id = str(item.get("id") or item.get("step_id") or "")
        if not step_id:
            raise ValueError("Each preprocess step must define 'id'")
        params = item.get("params", {}) or {}
        if not isinstance(params, Mapping):
            raise ValueError(f"params for step {step_id!r} must be a mapping")
        modalities = tuple(str(m) for m in (item.get("modalities") or ()))
        requires_fields = tuple(str(k) for k in (item.get("requires_fields") or ()))
        enabled = bool(item.get("enabled", True))
        steps.append(
            StepConfig(
                step_id=step_id,
                params=dict(params),
                modalities=modalities,
                requires_fields=requires_fields,
                enabled=enabled,
            )
        )

    return PreprocessPlan(steps=tuple(steps), output_key=output_key)


def resolve_fit_indices(
    *,
    dataset: LoadedDataset,
    sampling: SamplingResult,
    fit_on: str | None,
) -> np.ndarray | None:
    if fit_on is None:
        return None

    if sampling.is_graph():
        masks = sampling.masks
        if fit_on == "train":
            return np.where(np.asarray(masks["train"]))[0]
        if fit_on == "train_labeled":
            return np.where(np.asarray(masks["labeled"]))[0]
        if fit_on == "train_unlabeled":
            return np.where(np.asarray(masks["unlabeled"]))[0]
        if fit_on == "val":
            return np.where(np.asarray(masks["val"]))[0]
        raise ValueError(f"Unsupported fit_on for graph sampling: {fit_on!r}")

    if fit_on == "train":
        return np.asarray(sampling.indices["train"], dtype=np.int64)
    if fit_on == "train_labeled":
        return np.asarray(sampling.indices["train_labeled"], dtype=np.int64)
    if fit_on == "train_unlabeled":
        return np.asarray(sampling.indices["train_unlabeled"], dtype=np.int64)
    if fit_on == "val":
        return np.asarray(sampling.indices["val"], dtype=np.int64)

    raise ValueError(f"Unsupported fit_on: {fit_on!r}")


def run(
    dataset: LoadedDataset,
    *,
    plan_dict: Mapping[str, Any],
    seed: int,
    fit_indices: np.ndarray | None,
    cache: bool,
) -> PreprocessResult:
    start = perf_counter()
    plan = _plan_from_dict(plan_dict)
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
