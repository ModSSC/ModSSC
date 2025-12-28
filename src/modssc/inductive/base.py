from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .types import DeviceSpec


@runtime_checkable
class InductiveDatasetLike(Protocol):
    """Minimum dataset interface expected by inductive methods.

    All fields are read-only from the inductive brick perspective.
    """

    X_l: Any
    y_l: Any
    X_u: Any | None
    X_u_w: Any | None
    X_u_s: Any | None
    views: Mapping[str, Any] | None
    meta: Mapping[str, Any] | None


@dataclass(frozen=True)
class MethodInfo:
    """Metadata for an inductive method."""

    method_id: str
    name: str
    year: int | None = None
    family: str | None = None  # pseudo-label, consistency, mixup, teacher, agreement
    supports_gpu: bool = True
    required_extra: str | None = None
    paper_title: str | None = None
    paper_pdf: str | None = None
    official_code: str | None = None


class InductiveMethod(Protocol):
    """Common interface for inductive methods."""

    info: MethodInfo

    def fit(
        self, data: InductiveDatasetLike, *, device: DeviceSpec, seed: int = 0
    ) -> InductiveMethod: ...

    def predict_proba(self, X: Any) -> Any: ...
