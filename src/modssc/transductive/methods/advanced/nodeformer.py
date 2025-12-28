from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.errors import TransductiveNotImplementedError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeFormerSpec:
    """Configuration for NodeFormer (placeholder)."""

    # Intentionally empty in this wave.
    pass


class NodeFormerMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="nodeformer",
        name="NodeFormer",
        year=2022,
        family="advanced",
        supports_gpu=True,
        required_extra="transductive-advanced",
        paper_title="NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification",
        paper_pdf="https://arxiv.org/abs/2206.06615",
        official_code="",
    )

    def __init__(self, spec: NodeFormerSpec | None = None) -> None:
        self.spec = spec or NodeFormerSpec()

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> NodeFormerMethod:
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        raise TransductiveNotImplementedError(
            "nodeformer",
            hint="This advanced method is planned; run `modssc transductive methods list --all` "
            "to inspect planned entries or reach out if you need it prioritized.",
        )

    def predict_proba(self, data: Any) -> np.ndarray:
        raise TransductiveNotImplementedError("nodeformer")
