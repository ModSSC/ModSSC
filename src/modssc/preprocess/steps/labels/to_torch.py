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
            raise PreprocessValidationError("CUDA not available for labels.to_torch")
        return torch.device("cuda")
    if resolved == "mps":
        if not mps_is_available(torch):
            raise PreprocessValidationError("MPS not available for labels.to_torch")
        return torch.device("mps")
    raise PreprocessValidationError(f"Unknown device: {requested!r}")


def _resolve_dtype(torch, dtype: str):
    if dtype == "int64":
        return torch.int64
    if dtype == "int32":
        return torch.int32
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise PreprocessValidationError(f"Unknown dtype: {dtype!r}")


@dataclass
class LabelsToTorchStep:
    """Convert raw.y to a torch tensor and store as labels.y."""

    device: str = "cpu"
    dtype: str = "int64"

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        torch = require(module="torch", extra="inductive-torch", purpose="labels.to_torch")
        y = store.get("labels.y", store.require("raw.y"))
        dev = _resolve_device(torch, self.device)
        dt = _resolve_dtype(torch, self.dtype)
        tensor_type = getattr(torch, "Tensor", None)
        if tensor_type is not None and isinstance(y, tensor_type):
            return {"labels.y": y.to(device=dev, dtype=dt)}
        return {"labels.y": torch.as_tensor(y, device=dev, dtype=dt)}
