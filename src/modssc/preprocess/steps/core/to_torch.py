from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modssc.device import mps_is_available, resolve_device_name
from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.optional import require
from modssc.preprocess.store import ArtifactStore


def _resolve_device(torch, device: str) -> Any:
    requested = device
    resolved = resolve_device_name(device, torch=torch)
    if resolved == "cpu":
        return torch.device("cpu")
    if resolved == "cuda":
        if not torch.cuda.is_available():
            raise PreprocessValidationError("CUDA not available for core.to_torch")
        return torch.device("cuda")
    if resolved == "mps":
        if not mps_is_available(torch):
            raise PreprocessValidationError("MPS not available for core.to_torch")
        return torch.device("mps")
    raise PreprocessValidationError(f"Unknown device: {requested!r}")


def _resolve_dtype(torch, dtype: str | None):
    if dtype is None or dtype == "auto":
        return None
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise PreprocessValidationError(f"Unknown dtype: {dtype!r}")


@dataclass
class ToTorchStep:
    """Convert features.X to a torch tensor on a target device/dtype."""

    device: str = "cpu"
    dtype: str | None = "float32"
    input_key: str = "features.X"

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        torch = require(module="torch", extra="inductive-torch", purpose="core.to_torch")
        x = store.require(self.input_key)
        dev = _resolve_device(torch, self.device)
        dt = _resolve_dtype(torch, self.dtype)

        if isinstance(x, dict):
            # Recurse for dictionary values (e.g. for graph structures)
            res = {}
            for k, v in x.items():
                if isinstance(v, (np.ndarray, list)):
                    # Edge index is long, features are float usually.
                    # Heuristic: if key is edge_index, use long
                    local_dt = torch.long if (k == "edge_index" or "index" in k) else dt
                    res[k] = torch.as_tensor(v, device=dev, dtype=local_dt)
                else:
                    res[k] = v
            return {"features.X": res}

        return {"features.X": torch.as_tensor(x, device=dev, dtype=dt)}
