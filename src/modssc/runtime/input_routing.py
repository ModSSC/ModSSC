"""Native construction of regime-specific scientific execution inputs.

This is the boundary between materialized upstream bricks and method
execution.  It deliberately knows nothing about YAML: adapters report which
artifacts were configured, and this module decides what the selected learning
regime can actually receive.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from modssc.data_augmentation import (
    UnlabeledAugmentationResult,
    validate_augmentation_regime,
)
from modssc.data_augmentation.errors import DataAugmentationValidationError
from modssc.inductive.execution import InductiveExecutionInput
from modssc.preprocess.types import PreprocessResult
from modssc.runtime.execution import ExecutionContext
from modssc.sampling.errors import SamplingValidationError
from modssc.sampling.result import SamplingResult
from modssc.sampling.routing import (
    InductiveGraphSamplingPolicy,
    route_sampling_for_regime,
)
from modssc.transductive.data import masks_from_sampling
from modssc.transductive.errors import TransductiveDataError
from modssc.transductive.execution import TransductiveExecutionInput
from modssc.views.types import ViewsResult

LearningRegime = Literal["inductive", "transductive"]
InputRoutingErrorKind = Literal[
    "augmentation_regime",
    "augmentation_missing",
    "augmentation_alignment",
    "sampling_policy",
    "graph_missing",
    "mask_contract",
]

_ERROR_CODES: dict[InputRoutingErrorKind, str] = {
    "augmentation_regime": "E_INPUT_AUGMENTATION_REGIME",
    "augmentation_missing": "E_INPUT_AUGMENTATION_NOT_MATERIALIZED",
    "augmentation_alignment": "E_INPUT_AUGMENTATION_ALIGNMENT",
    "sampling_policy": "E_INPUT_SAMPLING_POLICY",
    "graph_missing": "E_INPUT_GRAPH_REQUIRED",
    "mask_contract": "E_INPUT_MASK_CONTRACT",
}


class InputRoutingError(ValueError):
    """Raised before a method receives an incomplete or unsupported input."""

    def __init__(self, kind: InputRoutingErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.code = _ERROR_CODES[kind]


@dataclass(frozen=True)
class ScientificInputRequest:
    """Materialized upstream artifacts and explicit regime-routing policy."""

    regime: LearningRegime
    preprocess: PreprocessResult
    sampling: SamplingResult
    graph: Any | None = None
    views: ViewsResult | None = None
    augmentation: UnlabeledAugmentationResult | None = None
    augmentation_configured: bool = False
    inductive_graph_policy: InductiveGraphSamplingPolicy | str = InductiveGraphSamplingPolicy.REJECT
    use_test_split: bool = False
    execution_context: ExecutionContext | None = None

    def __post_init__(self) -> None:
        if self.regime not in {"inductive", "transductive"}:
            raise ValueError("regime must be 'inductive' or 'transductive'")
        if not isinstance(self.preprocess, PreprocessResult):
            raise TypeError("preprocess must be a PreprocessResult")
        if not isinstance(self.sampling, SamplingResult):
            raise TypeError("sampling must be a SamplingResult")
        object.__setattr__(self, "augmentation_configured", bool(self.augmentation_configured))
        object.__setattr__(self, "use_test_split", bool(self.use_test_split))


@dataclass(frozen=True)
class RoutedScientificInput:
    """Exact regime-specific execution input plus auditable routing facts."""

    regime: LearningRegime
    execution_input: InductiveExecutionInput | TransductiveExecutionInput
    sampling: SamplingResult
    masks: Mapping[str, np.ndarray] | None
    expected_labeled_count: int | None
    events: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "sampling_representation": ("graph_masks" if self.sampling.is_graph() else "indices"),
            "has_graph": getattr(self.execution_input, "graph", None) is not None,
            "augmentation_delivered": (
                self.regime == "inductive"
                and any(
                    getattr(self.execution_input, name, None) is not None
                    for name in ("X_u_w", "X_u_s", "X_u_s_1", "online_augmentation")
                )
            ),
            "events": [dict(event) for event in self.events],
        }


def _augmentation_alignment(
    augmentation: UnlabeledAugmentationResult,
    sampling: SamplingResult,
) -> None:
    expected = np.asarray(sampling.indices["train_unlabeled"], dtype=np.int64).reshape(-1)
    delivered = np.asarray(augmentation.sample_ids, dtype=np.int64).reshape(-1)
    if not np.array_equal(delivered, expected):
        raise InputRoutingError(
            "augmentation_alignment",
            "augmentation sample_ids must exactly match the routed train_unlabeled indices",
        )


def route_scientific_input(request: ScientificInputRequest) -> RoutedScientificInput:
    """Build the exact native execution input for one learning regime.

    The returned ``sampling`` is the artifact actually used downstream.  In
    particular, an inductive graph-mask conversion appears both in that object
    and in ``events``.  A configured transductive augmentation fails here before
    method lookup or fitting.
    """

    if not isinstance(request, ScientificInputRequest):
        raise TypeError("request must be a ScientificInputRequest")

    augmentation_configured = request.augmentation_configured or request.augmentation is not None
    try:
        validate_augmentation_regime(
            regime=request.regime,
            configured=augmentation_configured,
        )
    except DataAugmentationValidationError as exc:
        raise InputRoutingError("augmentation_regime", str(exc)) from exc

    try:
        sampling_route = route_sampling_for_regime(
            request.sampling,
            regime=request.regime,
            inductive_graph_policy=request.inductive_graph_policy,
        )
    except SamplingValidationError as exc:
        raise InputRoutingError("sampling_policy", str(exc)) from exc

    events: list[Mapping[str, Any]] = [event.to_dict() for event in sampling_route.events]
    sampling = sampling_route.sampling
    if request.regime == "inductive":
        if augmentation_configured and request.augmentation is None:
            raise InputRoutingError(
                "augmentation_missing",
                "configured inductive augmentation has no materialized native result",
            )
        augmentation = request.augmentation
        if augmentation is not None:
            _augmentation_alignment(augmentation, sampling)
            events.append(
                {
                    "code": "augmentation.inductive_delivered",
                    "message": "Delivered native unlabeled augmentation views",
                    "sample_count": int(augmentation.sample_ids.size),
                    "online": augmentation.online is not None,
                }
            )
        if request.graph is not None:
            events.append(
                {
                    "code": "graph.inductive_delivered",
                    "message": "Attached the native graph artifact to InductiveDataset",
                }
            )
        execution_input = InductiveExecutionInput(
            preprocess=request.preprocess,
            sampling=sampling,
            views=request.views,
            X_u_w=None if augmentation is None else augmentation.weak,
            X_u_s=None if augmentation is None else augmentation.strong,
            X_u_s_1=None if augmentation is None else augmentation.second_strong,
            online_augmentation=None if augmentation is None else augmentation.online,
            graph=request.graph,
            routing_events=tuple(sampling_route.events),
            execution_context=request.execution_context,
        )
        return RoutedScientificInput(
            regime="inductive",
            execution_input=execution_input,
            sampling=sampling,
            masks=None,
            expected_labeled_count=None,
            events=tuple(events),
        )

    train_y = request.preprocess.dataset.train.y
    n_train = int(train_y.shape[0])
    test = request.preprocess.dataset.test
    n_test = int(test.y.shape[0]) if request.use_test_split and test is not None else None
    try:
        masks = masks_from_sampling(sampling, n_train=n_train, n_test=n_test)
    except TransductiveDataError as exc:
        raise InputRoutingError("mask_contract", str(exc)) from exc
    execution_input = TransductiveExecutionInput(
        dataset=request.preprocess.dataset,
        graph=request.graph,
        masks=masks,
        augmentation_configured=augmentation_configured,
        routing_events=tuple(events),
        execution_context=request.execution_context,
    )
    return RoutedScientificInput(
        regime="transductive",
        execution_input=execution_input,
        sampling=sampling,
        masks=masks,
        expected_labeled_count=sampling.expected_labeled_count(),
        events=tuple(events),
    )


__all__ = [
    "InputRoutingError",
    "InputRoutingErrorKind",
    "RoutedScientificInput",
    "ScientificInputRequest",
    "route_scientific_input",
]
