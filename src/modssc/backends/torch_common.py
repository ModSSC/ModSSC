from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from modssc.runtime.device import mps_is_available, resolve_device_name


def make_resolve_device(
    *,
    torch_getter: Callable[[], Any],
    optional_dependency_error_cls: type[Exception],
    extra: str,
) -> Callable[[Any], Any]:
    def resolve_device(spec: Any):
        torch = torch_getter()
        requested = spec.device
        resolved = resolve_device_name(requested, torch=torch)
        if resolved == "cpu":
            return torch.device("cpu")
        if resolved == "cuda":
            if not torch.cuda.is_available():  # type: ignore[attr-defined]
                raise optional_dependency_error_cls("torch", extra, message="CUDA not available")
            return torch.device("cuda")
        if resolved == "mps":
            if not mps_is_available(torch):
                raise optional_dependency_error_cls("torch", extra, message="MPS not available")
            return torch.device("mps")
        raise ValueError(f"Unknown device: {requested!r}")

    return resolve_device


def make_dtype_from_spec(*, torch_getter: Callable[[], Any]) -> Callable[[Any], Any]:
    def dtype_from_spec(spec: Any):
        torch = torch_getter()
        if spec.dtype == "float32":
            return torch.float32
        if spec.dtype == "float64":
            return torch.float64
        raise ValueError(f"Unknown dtype: {spec.dtype!r}")

    return dtype_from_spec


def make_to_tensor(*, torch_getter: Callable[[], Any]) -> Callable[..., Any]:
    def to_tensor(x, *, device, dtype=None):
        torch = torch_getter()
        t = torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.as_tensor(x)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t.to(device=device)

    return to_tensor
