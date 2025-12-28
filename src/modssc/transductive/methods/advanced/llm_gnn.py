from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.errors import TransductiveNotImplementedError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMGNNSpec:
    """Configuration for LLM-GNN (placeholder)."""

    # Intentionally empty in this wave.
    pass


class LLMGNNMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="llm_gnn",
        name="LLM-GNN",
        year=2023,
        family="advanced",
        supports_gpu=True,
        required_extra="transductive-advanced",
        paper_title="LLM-GNN (placeholder)",
        paper_pdf="",
        official_code="",
    )

    def __init__(self, spec: LLMGNNSpec | None = None) -> None:
        self.spec = spec or LLMGNNSpec()

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> LLMGNNMethod:
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        raise TransductiveNotImplementedError(
            "llm_gnn",
            hint="This hybrid method is on the roadmap; contact us if you need early integration.",
        )

    def predict_proba(self, data: Any) -> np.ndarray:
        raise TransductiveNotImplementedError("llm_gnn")
