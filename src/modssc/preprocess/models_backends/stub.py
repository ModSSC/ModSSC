from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.preprocess.numpy_adapter import to_numpy


def _sample_bytes(x: Any) -> bytes:
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, str):
        return x.encode("utf-8", errors="ignore")
    arr = to_numpy(x)
    # Use shape + dtype + bytes for stable representation (avoid huge copies).
    head = arr.tobytes()[:1024]
    meta = f"{arr.shape}|{arr.dtype}".encode()
    return meta + b"|" + head


@dataclass
class StubEncoder:
    """A deterministic encoder used for tests and offline examples."""

    dim: int = 8

    def encode(
        self, X: Any, *, batch_size: int = 32, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        # Ensure iterable of samples.
        if isinstance(X, np.ndarray) and X.ndim >= 2:
            samples = [X[i] for i in range(X.shape[0])]
        elif isinstance(X, list):
            samples = X
        else:
            samples = list(X)

        out = np.empty((len(samples), int(self.dim)), dtype=np.float32)
        for i, s in enumerate(samples):
            h = hashlib.sha256(_sample_bytes(s) + f"|{i}".encode()).digest()
            seed = int.from_bytes(h[:4], "big", signed=False)
            r = np.random.default_rng(seed)
            out[i] = r.standard_normal(int(self.dim), dtype=np.float32)
        return out
