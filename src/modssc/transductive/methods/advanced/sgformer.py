from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.transductive.base import MethodInfo, TransductiveMethod
from modssc.transductive.errors import TransductiveNotImplementedError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SGFormerSpec:
    """Configuration for SGFormer (placeholder)."""

    # Intentionally empty in this wave.
    pass


class SGFormerMethod(TransductiveMethod):
    info = MethodInfo(
        method_id="sgformer",
        name="SGFormer",
        year=2023,
        family="advanced",
        supports_gpu=True,
        required_extra="transductive-advanced",
        paper_title="SGFormer (placeholder)",
        paper_pdf="",
        official_code="",
    )

    def __init__(self, spec: SGFormerSpec | None = None) -> None:
        self.spec = spec or SGFormerSpec()

    def fit(self, data: Any, *, device: str | None = None, seed: int = 0) -> SGFormerMethod:
        logger.info("Starting %s.fit", self.info.method_id)
        logger.debug("spec=%s device=%s seed=%s", self.spec, device, seed)
        raise TransductiveNotImplementedError(
            "sgformer",
            hint="This entry is a roadmap placeholder; use `--all` when listing methods to view it.",
        )

    def predict_proba(self, data: Any) -> np.ndarray:
        raise TransductiveNotImplementedError("sgformer")
