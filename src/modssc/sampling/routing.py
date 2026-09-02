"""Native routing policy for sampling artifacts.

Sampling produces either row indices or graph masks.  Choosing to reinterpret
graph-node masks as inductive row indices is a scientific policy decision, not
a convenience conversion for a YAML runner.  This module makes that decision
explicit and records it in a portable trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from .errors import SamplingValidationError
from .result import SamplingResult

LearningRegime = Literal["inductive", "transductive"]


class InductiveGraphSamplingPolicy(StrEnum):
    """Policy for presenting graph-mask partitions to an inductive method."""

    REJECT = "reject"
    MASKS_TO_INDICES = "masks_to_indices"


@dataclass(frozen=True)
class SamplingRoutingEvent:
    """One machine-readable scientific transformation applied while routing."""

    code: str
    message: str
    source_representation: str
    target_representation: str
    policy: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "source_representation": self.source_representation,
            "target_representation": self.target_representation,
            "policy": self.policy,
        }


@dataclass(frozen=True)
class SamplingRoutingResult:
    """Sampling artifact actually delivered to a learning regime."""

    sampling: SamplingResult
    events: tuple[SamplingRoutingEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": "graph_masks" if self.sampling.is_graph() else "indices",
            "events": [event.to_dict() for event in self.events],
        }


def _inductive_policy(value: InductiveGraphSamplingPolicy | str) -> InductiveGraphSamplingPolicy:
    try:
        return InductiveGraphSamplingPolicy(value)
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in InductiveGraphSamplingPolicy)
        raise SamplingValidationError(
            f"inductive graph sampling policy must be one of: {allowed}"
        ) from exc


def route_sampling_for_regime(
    sampling: SamplingResult,
    *,
    regime: LearningRegime,
    inductive_graph_policy: InductiveGraphSamplingPolicy | str = (
        InductiveGraphSamplingPolicy.REJECT
    ),
) -> SamplingRoutingResult:
    """Return the exact sampling representation delivered to ``regime``.

    Transductive consumers retain graph masks.  Inductive consumers retain row
    indices; graph masks can only be converted when the caller declares the
    :attr:`~InductiveGraphSamplingPolicy.MASKS_TO_INDICES` policy explicitly.
    """

    if not isinstance(sampling, SamplingResult):
        raise TypeError("sampling must be a SamplingResult")
    if regime not in {"inductive", "transductive"}:
        raise ValueError("regime must be 'inductive' or 'transductive'")
    if regime == "transductive" or not sampling.is_graph():
        return SamplingRoutingResult(sampling=sampling)

    policy = _inductive_policy(inductive_graph_policy)
    if policy is InductiveGraphSamplingPolicy.REJECT:
        raise SamplingValidationError(
            "graph-mask sampling cannot be delivered to an inductive method unless "
            "inductive_graph_policy='masks_to_indices' is declared"
        )

    converted = sampling.as_inductive_indices()
    return SamplingRoutingResult(
        sampling=converted,
        events=(
            SamplingRoutingEvent(
                code="sampling.graph_masks_to_inductive_indices",
                message=(
                    "Converted graph-node partition masks to row indices in the shared "
                    "dataset.train node space"
                ),
                source_representation="graph_masks",
                target_representation="indices",
                policy=policy.value,
            ),
        ),
    )


__all__ = [
    "InductiveGraphSamplingPolicy",
    "SamplingRoutingEvent",
    "SamplingRoutingResult",
    "route_sampling_for_regime",
]
