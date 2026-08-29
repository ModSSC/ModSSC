from __future__ import annotations

from types import SimpleNamespace

import pytest

from modssc.runtime import (
    MethodExecutionOutcome,
    MethodNotEvaluableError,
    assess_method_execution,
    enforce_method_execution,
)


def _method(*, spec: object | None = None, diagnostics: object | None = None) -> object:
    method = SimpleNamespace()
    if spec is not None:
        method.spec = spec
    if diagnostics is not None:
        method.diagnostics_ = diagnostics
    return method


def test_method_outcome_is_opt_in_and_preserves_diagnostics() -> None:
    without_contract = assess_method_execution(_method(diagnostics={"rounds": 2}))
    assert without_contract == MethodExecutionOutcome(
        status="success",
        reason=None,
        diagnostics={"rounds": 2},
    )
    assert without_contract.code is None

    converged = assess_method_execution(
        _method(
            spec=SimpleNamespace(require_convergence=True),
            diagnostics={"converged": True},
        )
    )
    assert converged.status == "success"
    assert enforce_method_execution(_method()) == MethodExecutionOutcome(
        status="success",
        reason=None,
        diagnostics={},
    )


def test_declared_non_convergence_is_typed_not_evaluable() -> None:
    method = _method(
        spec=SimpleNamespace(require_convergence=True),
        diagnostics={"converged": False, "iterations": 1},
    )

    outcome = assess_method_execution(method)

    assert outcome.status == "not_evaluable"
    assert outcome.code == "E_METHOD_NOT_EVALUABLE"
    with pytest.raises(MethodNotEvaluableError, match="convergence") as raised:
        enforce_method_execution(method)
    assert raised.value.code == "E_METHOD_NOT_EVALUABLE"
    assert raised.value.reason == outcome.reason
    assert raised.value.diagnostics == {"converged": False, "iterations": 1}


@pytest.mark.parametrize(
    ("diagnostics", "minimum", "expected_status"),
    [
        ({"pseudo_labels_added_total": 3}, 3, "success"),
        ({"pseudo_labels_selected_total": 4}, 3, "success"),
        ({"pseudo_labels_added_total": 2}, 3, "not_evaluable"),
        ({"rounds": 1}, 1, "not_evaluable"),
    ],
)
def test_minimum_pseudo_label_requirement_is_generic(
    diagnostics: dict[str, int],
    minimum: int,
    expected_status: str,
) -> None:
    outcome = assess_method_execution(
        _method(
            spec=SimpleNamespace(min_pseudo_labels_added=minimum),
            diagnostics=diagnostics,
        )
    )

    assert outcome.status == expected_status


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (SimpleNamespace(require_convergence="yes"), "must be a boolean"),
        (SimpleNamespace(min_pseudo_labels_added=True), "non-negative integer"),
        (SimpleNamespace(min_pseudo_labels_added=-1), "non-negative integer"),
        (SimpleNamespace(min_pseudo_labels_added="1"), "non-negative integer"),
    ],
)
def test_method_outcome_rejects_invalid_requirement_types(
    spec: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assess_method_execution(_method(spec=spec, diagnostics=[]))


def test_not_evaluable_error_rejects_a_success_outcome() -> None:
    with pytest.raises(ValueError, match="requires a non-evaluable outcome"):
        MethodNotEvaluableError(
            MethodExecutionOutcome(status="success", reason=None, diagnostics={})
        )
