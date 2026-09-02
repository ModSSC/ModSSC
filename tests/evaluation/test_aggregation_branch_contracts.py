from __future__ import annotations

import builtins

import pytest

from modssc.evaluation import assess_evaluation_metrics
from modssc.evaluation.aggregation import _set_nested, _student_critical_95


def test_private_aggregation_guards_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        _student_critical_95(0)
    with pytest.raises(ValueError, match="metric path"):
        _set_nested({}, (), 1.0)


def test_student_interval_fails_closed_when_scipy_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def without_scipy(name: str, *args: object, **kwargs: object) -> object:
        if name == "scipy.stats":
            raise ImportError("scipy unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)

    with pytest.raises(RuntimeError, match="scipy is required"):
        _student_critical_95(4)


def test_evaluation_outcome_preserves_non_numeric_leaf() -> None:
    outcome = assess_evaluation_metrics({"status": "complete"})

    assert outcome.status == "success"
    assert outcome.metrics == {"status": "complete"}
