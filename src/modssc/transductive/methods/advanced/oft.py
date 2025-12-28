from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.errors import TransductiveNotImplementedError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OFTSpec:
    """Configuration for OFT (placeholder)."""

    # Intentionally empty in this wave.
    pass


class OFTMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="oft",
        name="OFT",
        year=2024,
        family="advanced",
        supports_gpu=True,
        required_extra="transductive-advanced",
        paper_title="OFT (placeholder)",
        paper_pdf="",
        official_code="",
    )

    def __init__(self, spec: OFTSpec | None = None) -> None:
        self.spec = spec or OFTSpec()

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> OFTMethod:
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        raise TransductiveNotImplementedError(
            "oft",
            hint="OFT remains planned; reach out or install the advanced extra once available.",
        )

    def predict_proba(self, data: Any) -> np.ndarray:
        raise TransductiveNotImplementedError("oft")
