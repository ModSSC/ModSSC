"""Generic scientific outcome gates for fitted methods.

Methods remain free to expose arbitrary diagnostics.  Reproduction cards may
opt into the two cross-method guarantees defined here through native dataclass
spec fields: convergence must be explicit, and a minimum number of pseudo-label
events may be required.  The benchmark runner does not know which method uses
either contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

MethodExecutionStatus = Literal["success", "not_evaluable"]


@dataclass(frozen=True)
class MethodExecutionOutcome:
    """Typed result of native post-fit scientific requirements."""

    status: MethodExecutionStatus
    reason: str | None
    diagnostics: Mapping[str, Any]

    @property
    def code(self) -> str | None:
        """Return the stable error code for a non-evaluable outcome."""

        return "E_METHOD_NOT_EVALUABLE" if self.status == "not_evaluable" else None


class MethodNotEvaluableError(RuntimeError):
    """Raised when a declared post-fit scientific requirement is unmet."""

    code = "E_METHOD_NOT_EVALUABLE"
    status: MethodExecutionStatus = "not_evaluable"

    def __init__(self, outcome: MethodExecutionOutcome) -> None:
        if outcome.status != "not_evaluable" or not outcome.reason:
            raise ValueError("MethodNotEvaluableError requires a non-evaluable outcome")
        super().__init__(outcome.reason)
        self.outcome = outcome
        self.reason = outcome.reason
        self.diagnostics = dict(outcome.diagnostics)


def _minimum_pseudo_labels(spec: Any) -> int | None:
    value = getattr(spec, "min_pseudo_labels_added", None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError("min_pseudo_labels_added must be a non-negative integer or null")
    return int(value)


def _pseudo_label_count(diagnostics: Mapping[str, Any]) -> int | None:
    for key in ("pseudo_labels_added_total", "pseudo_labels_selected_total"):
        value = diagnostics.get(key)
        if isinstance(value, Integral) and not isinstance(value, bool):
            return int(value)
    return None


def assess_method_execution(method: Any) -> MethodExecutionOutcome:
    """Assess optional, method-independent post-fit scientific requirements.

    The requirements are deliberately opt-in.  Existing programmatic uses keep
    their historical behaviour unless the native method specification declares
    ``require_convergence=True`` or ``min_pseudo_labels_added``.
    """

    diagnostics_value = getattr(method, "diagnostics_", None)
    diagnostics = dict(diagnostics_value) if isinstance(diagnostics_value, Mapping) else {}
    spec = getattr(method, "spec", None)

    require_convergence = getattr(spec, "require_convergence", False)
    if not isinstance(require_convergence, bool):
        raise ValueError("require_convergence must be a boolean")
    if require_convergence and diagnostics.get("converged") is not True:
        return MethodExecutionOutcome(
            status="not_evaluable",
            reason="the method did not satisfy its declared convergence requirement",
            diagnostics=diagnostics,
        )

    minimum = _minimum_pseudo_labels(spec)
    if minimum is not None:
        observed = _pseudo_label_count(diagnostics)
        if observed is None:
            return MethodExecutionOutcome(
                status="not_evaluable",
                reason="the method did not expose a pseudo-label count required by its spec",
                diagnostics=diagnostics,
            )
        if observed < minimum:
            return MethodExecutionOutcome(
                status="not_evaluable",
                reason=(
                    "the method produced fewer pseudo-label events than required: "
                    f"observed={observed} minimum={minimum}"
                ),
                diagnostics=diagnostics,
            )

    return MethodExecutionOutcome(
        status="success",
        reason=None,
        diagnostics=diagnostics,
    )


def enforce_method_execution(method: Any) -> MethodExecutionOutcome:
    """Return a successful outcome or raise a typed non-evaluable error."""

    outcome = assess_method_execution(method)
    if outcome.status == "not_evaluable":
        raise MethodNotEvaluableError(outcome)
    return outcome


__all__ = [
    "MethodExecutionOutcome",
    "MethodExecutionStatus",
    "MethodNotEvaluableError",
    "assess_method_execution",
    "enforce_method_execution",
]
