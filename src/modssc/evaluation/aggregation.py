"""Generic aggregation primitives for repeated evaluations.

The benchmark runner decides *when* several runs form a sweep. The numerical
meaning of their aggregation belongs to the evaluation brick and is reusable
independently of YAML files, schedulers, or research articles.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean, pstdev
from typing import Any

MetricPath = tuple[str, ...]


def iter_numeric_leaves(
    value: Any,
    *,
    path: MetricPath = (),
) -> list[tuple[MetricPath, float]]:
    """Return numeric leaves from a nested metric mapping."""

    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [(path, float(value))]
    if isinstance(value, Mapping):
        leaves: list[tuple[MetricPath, float]] = []
        for key, child in value.items():
            leaves.extend(iter_numeric_leaves(child, path=path + (str(key),)))
        return leaves
    return []


def summarize_numeric(values: Sequence[float]) -> dict[str, Any]:
    """Summarize a non-empty finite series with a two-sided Student CI95."""

    if not values:
        raise ValueError("values must be non-empty")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("values must contain only finite numbers")
    count = len(normalized)
    mean = float(fmean(normalized))
    population_std = float(pstdev(normalized)) if count > 1 else 0.0
    sample_std = (
        math.sqrt(sum((value - mean) ** 2 for value in normalized) / (count - 1))
        if count > 1
        else 0.0
    )
    if count > 1:
        half_width = _student_critical_95(count - 1) * sample_std / math.sqrt(count)
        ci95_low: float | None = mean - half_width
        ci95_high: float | None = mean + half_width
    else:
        ci95_low = None
        ci95_high = None
    return {
        "count": count,
        "mean": mean,
        "std": sample_std,
        "std_ddof": 1,
        "population_std": population_std,
        "min": float(min(normalized)),
        "max": float(max(normalized)),
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "values": normalized,
    }


def _student_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df=degrees_of_freedom))
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for exact two-sided Student confidence intervals"
        ) from exc


def _set_nested(mapping: dict[str, Any], path: MetricPath, value: Any) -> None:
    if not path:
        raise ValueError("metric path must be non-empty")
    current = mapping
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def aggregate_metric_records(
    records: Iterable[Mapping[str, Any]],
    *,
    require_same_schema: bool = True,
) -> dict[str, Any]:
    """Aggregate corresponding numeric leaves from repeated metric records.

    A repeated experiment is not scientifically comparable when successful
    runs emit different metric leaves, so schema equality is strict by default.
    """

    values_by_path: dict[MetricPath, list[float]] = {}
    expected_paths: set[MetricPath] | None = None
    for record in records:
        leaves = [(path, value) for path, value in iter_numeric_leaves(record) if path]
        paths = {path for path, _value in leaves}
        if require_same_schema and expected_paths is not None and paths != expected_paths:
            raise ValueError("metric schema differs between repeated evaluations")
        if expected_paths is None:
            expected_paths = paths
        for path, value in leaves:
            values_by_path.setdefault(path, []).append(value)

    aggregate: dict[str, Any] = {}
    for path, values in sorted(values_by_path.items()):
        _set_nested(aggregate, path, summarize_numeric(values))
    return aggregate


__all__ = [
    "MetricPath",
    "aggregate_metric_records",
    "iter_numeric_leaves",
    "summarize_numeric",
]
