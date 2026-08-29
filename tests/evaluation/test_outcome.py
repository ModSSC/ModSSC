from __future__ import annotations

import math

import numpy as np

from modssc.evaluation import assess_evaluation_metrics


def test_evaluation_outcome_is_success_when_all_metrics_are_finite() -> None:
    outcome = assess_evaluation_metrics(
        {"test": {"accuracy": np.float64(0.75), "benchmark_eligible": True}}
    )

    assert outcome.status == "success"
    assert outcome.code is None
    assert outcome.non_finite_paths == ()
    assert outcome.metrics == {"test": {"accuracy": 0.75, "benchmark_eligible": True}}


def test_evaluation_outcome_preserves_paths_and_uses_json_null_for_non_finite() -> None:
    outcome = assess_evaluation_metrics(
        {
            "test": {"accuracy": math.nan},
            "reported": {"history": [0.5, math.inf]},
        }
    )

    assert outcome.status == "not_evaluable"
    assert outcome.code == "E_EVALUATION_NOT_EVALUABLE"
    assert outcome.reason == "non_finite_metrics"
    assert outcome.non_finite_paths == (
        "test.accuracy",
        "reported.history.[1]",
    )
    assert outcome.metrics == {
        "test": {"accuracy": None},
        "reported": {"history": [0.5, None]},
    }
