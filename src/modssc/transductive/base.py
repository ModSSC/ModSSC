from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .types import DeviceSpec


@runtime_checkable
class GraphLike(Protocol):
    """Minimum graph interface expected by the transductive core."""

    edge_index: Any
    edge_weight: Any | None


@runtime_checkable
class NodeDatasetLike(Protocol):
    """Minimum node dataset interface expected by the transductive core."""

    X: Any
    y: Any
    graph: GraphLike
    masks: Mapping[str, Any] | None
    meta: Mapping[str, Any] | None


@dataclass(frozen=True)
class MethodInfo:
    """Metadata for a transductive method."""

    method_id: str
    name: str
    year: int | None = None
    family: str | None = None  # propagation, pde, gnn, cut, embedding
    supports_gpu: bool = False
    required_extra: str | None = None
    paper_title: str | None = None
    paper_pdf: str | None = None
    official_code: str | None = None


class TransductiveMethod(Protocol):
    """Common interface for transductive methods (future waves)."""

    info: MethodInfo

    def fit(
        self, data: NodeDatasetLike, *, device: DeviceSpec, seed: int = 0
    ) -> TransductiveMethod: ...

    def predict_proba(self, data: NodeDatasetLike) -> np.ndarray: ...
