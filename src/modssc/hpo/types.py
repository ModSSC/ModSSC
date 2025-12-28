from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class HpoError(ValueError):
    """Raised when an HPO space or distribution is invalid."""


@dataclass(frozen=True)
class Trial:
    index: int
    patch: dict[str, Any]
    params: dict[str, object]


@dataclass(frozen=True)
class DistributionSpec:
    dist: str
    low: float | None = None
    high: float | None = None
    values: list[Any] | None = None
