"""Thin benchmark adapter for the native evaluation brick."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from time import perf_counter
from typing import Any

from modssc.evaluation import (
    EvaluationError,
    evaluate_inductive_method,
    evaluate_transductive_method,
    make_inductive_split_provider,
)
from modssc.preprocess.types import PreprocessResult
from modssc.sampling.result import SamplingResult
from modssc.transductive.data import PreparedNodeData
from modssc.views.types import ViewsResult

from ..errors import BenchRuntimeError

_LOGGER = logging.getLogger(__name__)

_BENCH_ERROR_CODES = {
    "contract": "E_BENCH_EVAL_CONTRACT",
    "split": "E_BENCH_EVAL_SPLIT_INVALID",
    "torch_required": "E_BENCH_PREPROCESS_TO_TORCH_REQUIRED",
    "shape": "E_BENCH_SHAPE_CONTRACT",
}


def _bench_evaluation_error(exc: EvaluationError) -> BenchRuntimeError:
    return BenchRuntimeError(_BENCH_ERROR_CODES[exc.kind], str(exc))


def evaluate_inductive(
    *,
    method: Any,
    pre: PreprocessResult,
    sampling: SamplingResult,
    report_splits: Iterable[str],
    metrics: Iterable[str],
    views: ViewsResult | None,
    strict: bool = False,
) -> dict[str, dict[str, Any]]:
    """Adapt benchmark artifacts to the native evaluation API."""

    start = perf_counter()
    split_names = tuple(report_splits)
    metric_names = tuple(metrics)
    _LOGGER.info(
        "Evaluation (inductive): splits=%s metrics=%s strict=%s",
        list(split_names),
        list(metric_names),
        bool(strict),
    )
    try:
        results = evaluate_inductive_method(
            method=method,
            split_provider=make_inductive_split_provider(
                preprocess=pre,
                sampling=sampling,
                views=views,
            ),
            report_splits=split_names,
            metrics=metric_names,
            strict=bool(strict),
        )
    except EvaluationError as exc:
        raise _bench_evaluation_error(exc) from exc
    _LOGGER.info("Evaluation (inductive) done: duration_s=%.3f", perf_counter() - start)
    return results


def evaluate_transductive(
    *,
    method: Any,
    data: PreparedNodeData,
    report_splits: Iterable[str],
    metrics: Iterable[str],
    masks: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Delegate evaluation while keeping runner masks as an integrity check."""

    start = perf_counter()
    split_names = tuple(report_splits)
    metric_names = tuple(metrics)
    _LOGGER.info(
        "Evaluation (transductive): splits=%s metrics=%s",
        list(split_names),
        list(metric_names),
    )
    try:
        results = evaluate_transductive_method(
            method=method,
            data=data,
            report_splits=split_names,
            metrics=metric_names,
            declared_masks=masks,
        )
    except EvaluationError as exc:
        raise _bench_evaluation_error(exc) from exc
    _LOGGER.info("Evaluation (transductive) done: duration_s=%.3f", perf_counter() - start)
    return results


__all__ = ["evaluate_inductive", "evaluate_transductive"]
