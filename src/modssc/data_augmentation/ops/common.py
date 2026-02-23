from __future__ import annotations

from typing import Any

import numpy as np


def apply_gaussian_noise(
    x: Any,
    *,
    std: float,
    rng: np.random.Generator,
    is_torch_tensor_fn: Any,
) -> Any:
    std = float(std)
    if std < 0:
        raise ValueError("std must be >= 0")
    if std == 0:
        return x
    if is_torch_tensor_fn(x):
        import importlib

        torch = importlib.import_module("torch")
        seed = int(rng.integers(0, 1 << 31))
        gen = torch.Generator(device=x.device).manual_seed(seed)
        noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * std
        return x + noise
    arr = np.asarray(x)
    noise = rng.normal(0.0, std, size=arr.shape).astype(arr.dtype, copy=False)
    return arr + noise
