from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

DeviceName = Literal["cpu", "cuda", "mps", "auto"]
DTypeName = Literal["float32", "float64"]


@dataclass(frozen=True)
class DeviceSpec:
    """Device and dtype settings.

    device:
      - cpu: always CPU
      - cuda: use CUDA (error if unavailable)
      - mps: use Apple MPS (error if unavailable)
      - auto: pick cuda if available, else mps, else cpu

    dtype: numeric precision to use for math operators
    """

    device: DeviceName = "cpu"
    dtype: DTypeName = "float32"


@dataclass(frozen=True)
class InductiveDataset:
    """Read-only input bundle for inductive SSL methods.

    The inductive brick must not mutate these arrays; any data changes should
    be handled by upstream bricks (sampling, preprocess, augmentation, views).
    """

    X_l: Any
    y_l: Any
    X_u: Any | None = None
    X_u_w: Any | None = None
    X_u_s: Any | None = None
    views: Mapping[str, Any] | None = None
    meta: Mapping[str, Any] | None = None
