"""Typed, JSON-safe outcomes for native evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

EvaluationStatus = Literal["success", "not_evaluable"]


@dataclass(frozen=True)
class EvaluationOutcome:
    """Evaluation metrics plus an explicit scientific evaluability status."""

    metrics: dict[str, Any]
    status: EvaluationStatus
    reason: str | None = None
    non_finite_paths: tuple[str, ...] = ()

    @property
    def code(self) -> str | None:
        if self.status == "not_evaluable":
            return "E_EVALUATION_NOT_EVALUABLE"
        return None


def assess_evaluation_metrics(metrics: Mapping[str, Any]) -> EvaluationOutcome:
    """Replace non-finite metric numbers with ``None`` and classify the result.

    Non-finite values are never dropped or averaged away. Their exact paths are
    retained so runners can report a non-evaluable outcome using strict JSON.
    """

    non_finite_paths: list[str] = []

    def normalize(value: Any, path: tuple[str, ...]) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, Real):
            numeric = float(value)
            if math.isfinite(numeric):
                return value.item() if hasattr(value, "item") else value
            non_finite_paths.append(".".join(path) if path else "<root>")
            return None
        if isinstance(value, Mapping):
            return {str(key): normalize(child, path + (str(key),)) for key, child in value.items()}
        if isinstance(value, tuple | list):
            return [normalize(child, path + (f"[{index}]",)) for index, child in enumerate(value)]
        return value

    normalized = normalize(metrics, ())
    if not isinstance(normalized, dict):  # pragma: no cover - Mapping guarantees this
        raise TypeError("metrics must normalize to a dictionary")
    paths = tuple(non_finite_paths)
    if paths:
        return EvaluationOutcome(
            metrics=normalized,
            status="not_evaluable",
            reason="non_finite_metrics",
            non_finite_paths=paths,
        )
    return EvaluationOutcome(metrics=normalized, status="success")


__all__ = [
    "EvaluationOutcome",
    "EvaluationStatus",
    "assess_evaluation_metrics",
]
