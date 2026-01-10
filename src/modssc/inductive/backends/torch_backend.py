from __future__ import annotations

import numpy as np

from modssc.device import mps_is_available, resolve_device_name

from ..errors import OptionalDependencyError
from ..optional import optional_import
from ..types import DeviceSpec


def _torch():
    return optional_import("torch", extra="inductive-torch")


def resolve_device(spec: DeviceSpec):
    torch = _torch()
    requested = spec.device
    resolved = resolve_device_name(requested, torch=torch)
    if resolved == "cpu":
        return torch.device("cpu")
    if resolved == "cuda":
        if not torch.cuda.is_available():  # type: ignore[attr-defined]
            raise OptionalDependencyError("torch", "inductive-torch", message="CUDA not available")
        return torch.device("cuda")
    if resolved == "mps":
        if not mps_is_available(torch):
            raise OptionalDependencyError("torch", "inductive-torch", message="MPS not available")
        return torch.device("mps")
    raise ValueError(f"Unknown device: {requested!r}")


def dtype_from_spec(spec: DeviceSpec):
    torch = _torch()
    if spec.dtype == "float32":
        return torch.float32
    if spec.dtype == "float64":
        return torch.float64
    raise ValueError(f"Unknown dtype: {spec.dtype!r}")


def to_tensor(x, *, device, dtype=None):
    torch = _torch()
    t = torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(device=device)
