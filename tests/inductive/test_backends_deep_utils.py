from __future__ import annotations

import numpy as np
import pytest
import torch

from modssc import device as device_mod
from modssc.inductive.backends import torch_backend
from modssc.inductive.deep import TorchModelBundle, validate_torch_model_bundle
from modssc.inductive.errors import InductiveValidationError, OptionalDependencyError
from modssc.inductive.methods import deep_utils
from modssc.inductive.types import DeviceSpec

from .conftest import SimpleNet


def test_resolve_device_cpu():
    dev = torch_backend.resolve_device(DeviceSpec(device="cpu"))
    assert dev.type == "cpu"


def test_resolve_device_cuda_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    dev = torch_backend.resolve_device(DeviceSpec(device="cuda"))
    assert dev.type == "cuda"


def test_resolve_device_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(OptionalDependencyError):
        torch_backend.resolve_device(DeviceSpec(device="cuda"))


def test_resolve_device_mps_available(monkeypatch):
    if not hasattr(torch.backends, "mps"):
        pytest.skip("torch.backends.mps not available")
    device_mod.mps_is_available.cache_clear()
    if hasattr(torch.backends.mps, "is_built"):
        monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    orig_empty = torch.empty
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: orig_empty(*args))
    dev = torch_backend.resolve_device(DeviceSpec(device="mps"))
    assert dev.type == "mps"


def test_resolve_device_mps_unavailable(monkeypatch):
    if not hasattr(torch.backends, "mps"):
        pytest.skip("torch.backends.mps not available")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    device_mod.mps_is_available.cache_clear()
    with pytest.raises(OptionalDependencyError):
        torch_backend.resolve_device(DeviceSpec(device="mps"))


def test_resolve_device_auto_paths(monkeypatch):
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    device_mod.mps_is_available.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert torch_backend.resolve_device(DeviceSpec(device="auto")).type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        if hasattr(torch.backends.mps, "is_built"):
            monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        device_mod.mps_is_available.cache_clear()
        orig_empty = torch.empty
        monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: orig_empty(*args))
        assert torch_backend.resolve_device(DeviceSpec(device="auto")).type == "mps"

    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    device_mod.mps_is_available.cache_clear()
    assert torch_backend.resolve_device(DeviceSpec(device="auto")).type == "cpu"


def test_resolve_device_unknown():
    with pytest.raises(ValueError, match="Unknown device"):
        torch_backend.resolve_device(DeviceSpec(device="quantum"))


def test_dtype_from_spec_and_to_tensor():
    assert torch_backend.dtype_from_spec(DeviceSpec(dtype="float32")) == torch.float32
    assert torch_backend.dtype_from_spec(DeviceSpec(dtype="float64")) == torch.float64
    with pytest.raises(ValueError):
        torch_backend.dtype_from_spec(DeviceSpec(dtype="float16"))  # type: ignore[arg-type]

    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    t = torch_backend.to_tensor(arr, device=torch.device("cpu"), dtype=torch.float64)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float64

    t2 = torch_backend.to_tensor([[1.0, 2.0]], device=torch.device("cpu"))
    assert isinstance(t2, torch.Tensor)


def test_ensure_model_device_checks():
    class _EmptyModel:
        def parameters(self):
            return []

    with pytest.raises(InductiveValidationError):
        deep_utils.ensure_model_device(_EmptyModel(), device=torch.device("cpu"))

    class _Param:
        def __init__(self, device):
            self.device = device

    class _Mixed:
        def parameters(self):
            return [_Param(torch.device("cpu")), _Param(torch.device("cuda"))]

    with pytest.raises(InductiveValidationError):
        deep_utils.ensure_model_device(_Mixed(), device=torch.device("cpu"))

    model = SimpleNet()
    with pytest.raises(InductiveValidationError):
        deep_utils.ensure_model_device(model, device=torch.device("cuda"))

    deep_utils.ensure_model_device(model, device=torch.device("cpu"))


def test_extract_logits_and_features():
    logits = torch.randn(2, 3)
    assert deep_utils.extract_logits(logits) is logits

    out = {"logits": logits}
    assert deep_utils.extract_logits(out) is logits

    out_tup = (logits,)
    assert deep_utils.extract_logits(out_tup) is logits

    with pytest.raises(InductiveValidationError):
        deep_utils.extract_logits({"logits": "nope"})
    with pytest.raises(InductiveValidationError):
        deep_utils.extract_logits(("nope",))
    with pytest.raises(InductiveValidationError):
        deep_utils.extract_logits({"feat": logits})

    feat = torch.randn(2, 2)
    assert deep_utils.extract_features({"feat": feat}) is feat
    with pytest.raises(InductiveValidationError):
        deep_utils.extract_features({"feat": "nope"})
    with pytest.raises(InductiveValidationError):
        deep_utils.extract_features({"logits": logits})


def test_ensure_float_tensor():
    deep_utils.ensure_float_tensor(torch.zeros((1, 2), dtype=torch.float32), name="X")
    with pytest.raises(InductiveValidationError):
        deep_utils.ensure_float_tensor(torch.zeros((1, 2), dtype=torch.int64), name="X")
    with pytest.raises(InductiveValidationError):
        deep_utils.ensure_float_tensor([[1, 2]], name="X")


def test_freeze_batchnorm_and_num_batches():
    model = SimpleNet()
    model.train()
    bn = model.bn
    assert bn.training is True

    with deep_utils.freeze_batchnorm(model, enabled=True):
        assert bn.training is False
    assert bn.training is True

    with deep_utils.freeze_batchnorm(model, enabled=False):
        assert bn.training is True

    assert deep_utils.num_batches(0, 4) == 1
    assert deep_utils.num_batches(5, 2) == 3


def test_cycle_batch_indices_and_batches():
    gen = torch.Generator().manual_seed(0)
    idx = list(deep_utils.cycle_batch_indices(5, batch_size=2, generator=gen, device=None, steps=4))
    assert len(idx) == 4
    idx_meta = list(
        deep_utils.cycle_batch_indices(
            4, batch_size=2, generator=gen, device=torch.device("meta"), steps=1
        )
    )
    assert idx_meta[0].device.type == "meta"

    X = torch.arange(10, dtype=torch.float32).view(5, 2)
    y = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int64)
    batches = list(deep_utils.cycle_batches(X, y, batch_size=2, generator=gen, steps=3))
    assert len(batches) == 3

    batches_no_y = list(deep_utils.cycle_batches(X, None, batch_size=2, generator=gen, steps=2))
    assert batches_no_y[0][1] is None

    with pytest.raises(InductiveValidationError):
        list(deep_utils.cycle_batches(torch.empty((0, 2)), y, batch_size=2, generator=gen, steps=1))
    with pytest.raises(InductiveValidationError):
        list(deep_utils.cycle_batches(np.zeros((2, 2)), y, batch_size=2, generator=gen, steps=1))


def test_validate_torch_model_bundle_errors():
    with pytest.raises(InductiveValidationError):
        validate_torch_model_bundle("bad")  # type: ignore[arg-type]

    base = SimpleNet()
    opt = torch.optim.SGD(base.parameters(), lr=0.1)

    with pytest.raises(InductiveValidationError):
        validate_torch_model_bundle(TorchModelBundle(model="bad", optimizer=opt))  # type: ignore[arg-type]

    with pytest.raises(InductiveValidationError):
        validate_torch_model_bundle(TorchModelBundle(model=base, optimizer="bad"))  # type: ignore[arg-type]

    frozen = SimpleNet()
    for p in frozen.parameters():
        p.requires_grad = False
    opt_frozen = torch.optim.SGD(frozen.parameters(), lr=0.1)
    with pytest.raises(InductiveValidationError):
        validate_torch_model_bundle(TorchModelBundle(model=frozen, optimizer=opt_frozen))

    other = SimpleNet()
    opt_other = torch.optim.SGD(other.parameters(), lr=0.1)
    with pytest.raises(InductiveValidationError):
        validate_torch_model_bundle(TorchModelBundle(model=base, optimizer=opt_other))

    with pytest.raises(InductiveValidationError):
        validate_torch_model_bundle(TorchModelBundle(model=base, optimizer=opt, ema_model="bad"))
