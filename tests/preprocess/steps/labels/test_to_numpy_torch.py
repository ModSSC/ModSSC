from __future__ import annotations

import numpy as np
import pytest

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.labels import to_torch as labels_to_torch
from modssc.preprocess.steps.labels.to_numpy import LabelsToNumpyStep
from modssc.preprocess.store import ArtifactStore


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeMPS:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = _FakeMPS(mps_available)


class FakeTorch:
    float32 = "float32"
    float64 = "float64"
    int64 = "int64"
    int32 = "int32"

    def __init__(self, *, cuda_available: bool = False, mps_available: bool = False) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self.backends = _FakeBackends(mps_available)

    def device(self, name: str) -> str:
        return f"device:{name}"

    def as_tensor(self, x, device=None, dtype=None):
        return {"value": np.asarray(x), "device": device, "dtype": dtype}


def test_labels_to_numpy_steps() -> None:
    store = ArtifactStore()
    store.set("raw.y", [0, 1, 1])
    out = LabelsToNumpyStep().transform(store, rng=np.random.default_rng(0))
    assert isinstance(out["labels.y"], np.ndarray)


def test_labels_resolve_device_and_dtype() -> None:
    torch = FakeTorch()
    assert labels_to_torch._resolve_device(torch, "cpu") == "device:cpu"

    with pytest.raises(PreprocessValidationError, match="CUDA not available"):
        labels_to_torch._resolve_device(FakeTorch(cuda_available=False), "cuda")
    assert labels_to_torch._resolve_device(FakeTorch(cuda_available=True), "cuda") == "device:cuda"

    with pytest.raises(PreprocessValidationError, match="MPS not available"):
        labels_to_torch._resolve_device(FakeTorch(mps_available=False), "mps")
    assert labels_to_torch._resolve_device(FakeTorch(mps_available=True), "mps") == "device:mps"

    assert labels_to_torch._resolve_device(FakeTorch(cuda_available=True), "auto") == "device:cuda"
    assert labels_to_torch._resolve_device(FakeTorch(mps_available=True), "auto") == "device:mps"
    assert labels_to_torch._resolve_device(FakeTorch(), "auto") == "device:cpu"

    with pytest.raises(PreprocessValidationError, match="Unknown device"):
        labels_to_torch._resolve_device(torch, "weird")

    assert labels_to_torch._resolve_dtype(torch, "int64") == torch.int64
    assert labels_to_torch._resolve_dtype(torch, "int32") == torch.int32
    assert labels_to_torch._resolve_dtype(torch, "float32") == torch.float32
    assert labels_to_torch._resolve_dtype(torch, "float64") == torch.float64

    with pytest.raises(PreprocessValidationError, match="Unknown dtype"):
        labels_to_torch._resolve_dtype(torch, "bad")


def test_labels_to_torch_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = FakeTorch()
    monkeypatch.setattr(labels_to_torch, "require", lambda **_: fake_torch)

    store = ArtifactStore()
    store.set("raw.y", [0, 1, 1])
    step = labels_to_torch.LabelsToTorchStep(device="cpu", dtype="int64")
    out = step.transform(store, rng=np.random.default_rng(0))

    val = out["labels.y"]
    assert val["device"] == "device:cpu"
    assert val["dtype"] == fake_torch.int64
