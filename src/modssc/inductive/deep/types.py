from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TorchModelBundle:
    """Torch model bundle provided by the deep-model brick."""

    model: Any
    optimizer: Any
    ema_model: Any | None = None
    scheduler: Any | None = None
    scaler: Any | None = None
    meta: Mapping[str, Any] | None = None
