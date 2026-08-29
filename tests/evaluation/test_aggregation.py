from __future__ import annotations

import pytest

from modssc.evaluation import (
    aggregate_metric_records,
    iter_numeric_leaves,
    summarize_numeric,
)
from modssc.evaluation.aggregation import _student_critical_95


@pytest.mark.parametrize(
    ("degrees_of_freedom", "expected"),
    [
        (1, 12.706204736432095),
        (4, 2.7764451051977987),
        (9, 2.2621571628540993),
        (99, 1.9842169515086827),
    ],
)
def test_student_critical_values_are_exact_scipy_quantiles(
    degrees_of_freedom: int,
    expected: float,
) -> None:
    assert _student_critical_95(degrees_of_freedom) == pytest.approx(expected)


def test_summarize_numeric_uses_population_standard_deviation() -> None:
    assert summarize_numeric([1.0, 3.0]) == {
        "count": 2,
        "mean": 2.0,
        "std": pytest.approx(2**0.5),
        "std_ddof": 1,
        "population_std": 1.0,
        "min": 1.0,
        "max": 3.0,
        "ci95_low": pytest.approx(-10.706204736432095),
        "ci95_high": pytest.approx(14.706204736432095),
        "values": [1.0, 3.0],
    }
    assert summarize_numeric([2]) == {
        "count": 1,
        "mean": 2.0,
        "std": 0.0,
        "std_ddof": 1,
        "population_std": 0.0,
        "min": 2.0,
        "max": 2.0,
        "ci95_low": None,
        "ci95_high": None,
        "values": [2.0],
    }


def test_summarize_numeric_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        summarize_numeric([])
    with pytest.raises(ValueError, match="finite"):
        summarize_numeric([float("nan")])


def test_iter_numeric_leaves_ignores_booleans_lists_and_text() -> None:
    leaves = iter_numeric_leaves(
        {
            "test": {"accuracy": 0.75, "enabled": True},
            "diagnostics": [1, 2],
            "status": "ok",
        }
    )

    assert leaves == [(("test", "accuracy"), 0.75)]


def test_aggregate_metric_records_handles_nested_metrics() -> None:
    aggregate = aggregate_metric_records(
        [
            {"test": {"accuracy": 0.8, "macro_f1": 0.7}},
            {"test": {"accuracy": 1.0, "macro_f1": 0.9}},
        ]
    )

    assert aggregate["test"]["accuracy"] == {
        "count": 2,
        "mean": 0.9,
        "std": pytest.approx(2**0.5 / 10),
        "std_ddof": 1,
        "population_std": pytest.approx(0.1),
        "min": 0.8,
        "max": 1.0,
        "ci95_low": pytest.approx(-0.3706204736432095),
        "ci95_high": pytest.approx(2.1706204736432095),
        "values": [0.8, 1.0],
    }
    assert aggregate["test"]["macro_f1"]["count"] == 2
    assert aggregate_metric_records([]) == {}


def test_aggregate_metric_records_rejects_mismatched_metric_schemas() -> None:
    with pytest.raises(ValueError, match="schema differs"):
        aggregate_metric_records([{"test": {"accuracy": 1.0}}, {"test": {"macro_f1": 1.0}}])

    aggregate = aggregate_metric_records(
        [{"test": {"accuracy": 1.0}}, {"test": {"macro_f1": 1.0}}],
        require_same_schema=False,
    )
    assert set(aggregate["test"]) == {"accuracy", "macro_f1"}
