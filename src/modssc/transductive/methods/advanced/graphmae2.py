from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.errors import TransductiveNotImplementedError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphMAE2Spec:
    """Configuration for GraphMAE2 (placeholder)."""

    # Intentionally empty in this wave.
    pass


class GraphMAE2Method(TransductiveMethod):
    info = MethodInfo(
        method_id="graphmae2",
        name="GraphMAE2",
        year=2023,
        family="advanced",
        supports_gpu=True,
        required_extra="transductive-advanced",
        paper_title="GraphMAE2: A Decoding-Enhanced Masked Graph Autoencoder for Structural Information",
        paper_pdf="https://arxiv.org/abs/2304.12283",
        official_code="",
    )

    def __init__(self, spec: GraphMAE2Spec | None = None) -> None:
        self.spec = spec or GraphMAE2Spec()

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> GraphMAE2Method:
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        raise TransductiveNotImplementedError(
            "graphmae2",
            hint="This ID tracks a planned advanced method; contact the maintainers if you need it.",
        )

    def predict_proba(self, data: Any) -> np.ndarray:
        raise TransductiveNotImplementedError("graphmae2")
