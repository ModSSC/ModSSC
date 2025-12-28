from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Encoder(Protocol):
    """Minimal encoder interface used by embedding steps."""

    def encode(
        self, X: Any, *, batch_size: int = 32, rng: np.random.Generator | None = None
    ) -> np.ndarray: ...
