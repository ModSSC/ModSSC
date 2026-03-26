from __future__ import annotations

import numpy as np
import pytest

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.steps.core import to_torch as core_to_torch
from modssc.preprocess.steps.core.to_numpy import ToNumpyStep
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
    long = "int64"

    def __init__(self, *, cuda_available: bool = False, mps_available: bool = False) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self.backends = _FakeBackends(mps_available)

    def device(self, name: str) -> str:
        return f"device:{name}"

    def zeros(self, *args, **kwargs):
        return object()

    def as_tensor(self, x, device=None, dtype=None):
        return {"value": np.asarray(x), "device": device, "dtype": dtype}


def test_to_numpy_steps() -> None:
    store = ArtifactStore()
    store.set("features.X", [[1.0, 2.0], [3.0, 4.0]])
    out = ToNumpyStep().transform(store, rng=np.random.default_rng(0))
    assert isinstance(out["features.X"], np.ndarray)


def test_core_resolve_device_and_dtype() -> None:
    torch = FakeTorch()
    assert core_to_torch._resolve_device(torch, "cpu") == "device:cpu"

    with pytest.raises(PreprocessValidationError, match="CUDA not available"):
        core_to_torch._resolve_device(FakeTorch(cuda_available=False), "cuda")
    assert core_to_torch._resolve_device(FakeTorch(cuda_available=True), "cuda") == "device:cuda"

    with pytest.raises(PreprocessValidationError, match="MPS not available"):
        core_to_torch._resolve_device(FakeTorch(mps_available=False), "mps")
    assert core_to_torch._resolve_device(FakeTorch(mps_available=True), "mps") == "device:mps"

    assert core_to_torch._resolve_device(FakeTorch(cuda_available=True), "auto") == "device:cuda"
    assert core_to_torch._resolve_device(FakeTorch(mps_available=True), "auto") == "device:mps"
    assert core_to_torch._resolve_device(FakeTorch(), "auto") == "device:cpu"

    with pytest.raises(PreprocessValidationError, match="Unknown device"):
        core_to_torch._resolve_device(torch, "weird")

    assert core_to_torch._resolve_dtype(torch, None) is None
    assert core_to_torch._resolve_dtype(torch, "auto") is None
    assert core_to_torch._resolve_dtype(torch, "float32") == torch.float32
    assert core_to_torch._resolve_dtype(torch, "float64") == torch.float64

    with pytest.raises(PreprocessValidationError, match="Unknown dtype"):
        core_to_torch._resolve_dtype(torch, "bad")


def test_core_to_torch_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = FakeTorch()
    monkeypatch.setattr(core_to_torch, "require", lambda **_: fake_torch)

    store = ArtifactStore()
    store.set("features.X", [[1.0, 2.0], [3.0, 4.0]])
    step = core_to_torch.ToTorchStep(device="cpu", dtype="float32")
    out = step.transform(store, rng=np.random.default_rng(0))

    val = out["features.X"]
    assert val["device"] == "device:cpu"
    assert val["dtype"] == fake_torch.float32


def test_core_to_torch_transform_custom_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = FakeTorch()
    monkeypatch.setattr(core_to_torch, "require", lambda **_: fake_torch)

    store = ArtifactStore()
    store.set("my_custom_key", [[1.0, 2.0], [3.0, 4.0]])

    step = core_to_torch.ToTorchStep(device="cpu", dtype="float32", input_key="my_custom_key")
    out = step.transform(store, rng=np.random.default_rng(0))

    val = out["features.X"]
    assert val["device"] == "device:cpu"
    assert val["dtype"] == fake_torch.float32
    np.testing.assert_array_equal(val["value"], [[1.0, 2.0], [3.0, 4.0]])


def test_core_to_torch_transform_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = FakeTorch()
    monkeypatch.setattr(core_to_torch, "require", lambda **_: fake_torch)

    store = ArtifactStore()
    store.set(
        "features.X",
        {
            "x": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "edge_index": np.array([[0, 1], [1, 0]]),
            "meta": "keep",
        },
    )

    step = core_to_torch.ToTorchStep(device="cpu", dtype="float32")
    out = step.transform(store, rng=np.random.default_rng(0))

    res = out["features.X"]
    assert res["x"]["dtype"] == fake_torch.float32
    assert res["edge_index"]["dtype"] == fake_torch.long
    assert res["meta"] == "keep"
