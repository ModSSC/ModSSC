from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .types import Backend


def is_torch_tensor(x: Any) -> bool:
    """Heuristic check for torch tensors without importing torch eagerly."""
    if isinstance(x, dict):
        return False

    mod = type(x).__module__
    if not mod.startswith("torch"):
        return False
    # Typical tensor attributes
    return hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "device")


def to_numpy(x: Any) -> np.ndarray:
    """Best-effort conversion to numpy (detaches torch tensors)."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _u64_from_bytes(b: bytes) -> int:
    return int.from_bytes(b[:8], "little", signed=False)


def seed_to_rng_seed(*, seed: int, epoch: int, sample_id: int) -> int:
    """Derive a stable 32-bit RNG seed from (seed, epoch, sample_id)."""
    payload = f"{int(seed)}|{int(epoch)}|{int(sample_id)}".encode()
    h = hashlib.blake2b(payload, digest_size=16).digest()
    # numpy wants uint32-ish; keep within 0..2**32-1
    return int(_u64_from_bytes(h) % (2**32))


def make_numpy_rng(*, seed: int, epoch: int = 0, sample_id: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed_to_rng_seed(seed=seed, epoch=epoch, sample_id=sample_id))


def resolve_backend(x: Any, backend: Backend) -> Backend:
    if backend == "auto":
        return "torch" if is_torch_tensor(x) else "numpy"
    return backend


def ensure_edge_index_2xE(edge_index: Any) -> Any:
    if is_torch_tensor(edge_index):
        ei = edge_index
        if ei.ndim != 2:
            raise ValueError(f"edge_index must be 2D, got shape={ei.shape}")
        if ei.shape[0] == 2:
            return ei
        if ei.shape[1] == 2:
            return ei.t()
        raise ValueError(f"edge_index must have shape (2, E) or (E, 2), got {ei.shape}")

    ei = to_numpy(edge_index).astype(np.int64, copy=False)
    if ei.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape={ei.shape}")
    if ei.shape[0] == 2:
        return ei
    if ei.shape[1] == 2:
        return ei.T
    raise ValueError(f"edge_index must have shape (2, E) or (E, 2), got {ei.shape}")


def copy_like(x: Any) -> Any:
    """Shallow copy for numpy/torch tensors."""
    if isinstance(x, np.ndarray):
        return x.copy()
    if is_torch_tensor(x):
        return x.clone()
    # best effort
    try:
        return x.copy()
    except Exception:
        return x


def split_image_channels_last(x: np.ndarray) -> tuple[np.ndarray, str]:
    """Return image array and a layout string ('hw', 'hwc', 'chw')."""
    if x.ndim == 2:
        return x, "hw"
    if x.ndim == 3:
        # Decide between HWC and CHW heuristically (small channel count)
        if x.shape[0] in (1, 3, 4) and x.shape[2] not in (1, 3, 4):
            return x, "chw"
        return x, "hwc"
    raise ValueError(f"Expected image with ndim 2 or 3, got shape={x.shape}")
