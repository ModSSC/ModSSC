from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
