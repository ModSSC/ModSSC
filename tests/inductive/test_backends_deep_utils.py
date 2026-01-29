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


def test_ensure_float_tensor_dict_paths():
    x_ok = {"x": torch.zeros((2, 2), dtype=torch.float32), "meta": "keep"}
    deep_utils.ensure_float_tensor(x_ok, name="X")

    x_no_x = {"feat": torch.zeros((1, 2), dtype=torch.float32)}
    deep_utils.ensure_float_tensor(x_no_x, name="X")

    x_bad = {"x": torch.zeros((2, 2), dtype=torch.int64)}
    with pytest.raises(InductiveValidationError, match="float32 or float64"):
        deep_utils.ensure_float_tensor(x_bad, name="X")

    with pytest.raises(InductiveValidationError, match="must be a torch.Tensor"):
        deep_utils.ensure_float_tensor({"meta": "no-tensor"}, name="X")


def test_get_torch_helpers_dict_and_tensor():
    x = torch.zeros((3, 4))
    assert deep_utils.get_torch_len(x) == 3
    assert deep_utils.get_torch_device(x) == x.device
    assert deep_utils.get_torch_feature_dim(x) == 4
    assert deep_utils.get_torch_ndim(x) == 2

    d = {"x": torch.zeros((2, 5))}
    assert deep_utils.get_torch_len(d) == 2
    assert deep_utils.get_torch_device(d) == d["x"].device
    assert deep_utils.get_torch_feature_dim(d) == 5
    assert deep_utils.get_torch_ndim(d) == 2


def _install_fake_tg_utils(monkeypatch, *, raise_import: bool):
    import sys
    import types

    utils = types.ModuleType("torch_geometric.utils")
    if raise_import:

        def subgraph(*_args, **_kwargs):
            raise ImportError("no pyg")
    else:

        def subgraph(idx, edge_index, relabel_nodes=True, num_nodes=None):
            return edge_index, None

    utils.subgraph = subgraph

    tg = types.ModuleType("torch_geometric")
    tg.utils = utils
    monkeypatch.setitem(sys.modules, "torch_geometric", tg)
    monkeypatch.setitem(sys.modules, "torch_geometric.utils", utils)


def _install_fake_tg_data(monkeypatch, *, with_batch: bool):
    import sys
    import types

    data_mod = types.ModuleType("torch_geometric.data")
    if with_batch:

        class Data:
            def __init__(self, **kwargs):
                self._data = kwargs

        class Batch:
            def __init__(self, data_list):
                self._data_list = data_list

            @classmethod
            def from_data_list(cls, data_list):
                return cls(data_list)

            def to_dict(self):
                x = torch.cat([d._data["x"] for d in self._data_list], dim=0)
                edge_index = torch.cat([d._data["edge_index"] for d in self._data_list], dim=1)
                return {"x": x, "edge_index": edge_index}

        data_mod.Data = Data
        data_mod.Batch = Batch

    tg = types.ModuleType("torch_geometric")
    tg.data = data_mod
    monkeypatch.setitem(sys.modules, "torch_geometric", tg)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_mod)


def test_slice_data_dict_with_and_without_pyg(monkeypatch):
    X = {
        "x": torch.arange(6, dtype=torch.float32).view(3, 2),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "mask": torch.tensor([1, 0, 1]),
        "meta": "keep",
    }
    idx = torch.tensor([0, 2], dtype=torch.long)

    _install_fake_tg_utils(monkeypatch, raise_import=False)
    out = deep_utils.slice_data(X, idx)
    assert out["x"].shape == (2, 2)
    assert "edge_index" in out
    assert out["mask"].shape == (2,)
    assert out["meta"] == "keep"

    _install_fake_tg_utils(monkeypatch, raise_import=True)
    out2 = deep_utils.slice_data(X, idx)
    assert "edge_index" not in out2


def test_slice_data_dict_with_slice_idx(monkeypatch):
    import sys
    import types

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    X = {"x": torch.arange(12, dtype=torch.float32).view(4, 3), "edge_index": edge_index}
    idx = slice(1, 3)
    expected_idx = torch.arange(1, 3, device=edge_index.device)

    def subgraph(sub_idx, edge_idx, relabel_nodes=True, num_nodes=None):
        assert torch.equal(sub_idx, expected_idx)
        assert edge_idx is edge_index
        return edge_idx, None

    utils = types.ModuleType("torch_geometric.utils")
    utils.subgraph = subgraph
    tg = types.ModuleType("torch_geometric")
    tg.utils = utils
    monkeypatch.setitem(sys.modules, "torch_geometric", tg)
    monkeypatch.setitem(sys.modules, "torch_geometric.utils", utils)

    out = deep_utils.slice_data(X, idx)
    assert out["x"].shape == (2, 3)
    assert "edge_index" in out


def test_slice_data_dict_without_x():
    X = {"mask": torch.zeros((0, 2)), "meta": "keep"}
    out = deep_utils.slice_data(X, torch.tensor([], dtype=torch.long))
    assert out["mask"].shape == (0, 2)
    assert out["meta"] == "keep"


def test_cat_data_dict_paths(monkeypatch):
    assert deep_utils.cat_data([]) is None

    _install_fake_tg_data(monkeypatch, with_batch=True)
    d1 = {"x": torch.zeros((1, 2)), "edge_index": torch.tensor([[0], [0]])}
    d2 = {"x": torch.ones((1, 2)), "edge_index": torch.tensor([[0], [0]])}
    out = deep_utils.cat_data([d1, d2])
    assert out["x"].shape[0] == 2
    assert out["edge_index"].shape[1] == 2

    # ImportError path -> fallback concat
    _install_fake_tg_data(monkeypatch, with_batch=False)
    d3 = {"x": torch.zeros((1, 2)), "edge_index": torch.tensor([[0], [0]]), "meta": "keep"}
    d4 = {"x": torch.ones((1, 2)), "edge_index": torch.tensor([[0], [0]]), "meta": "keep"}
    out2 = deep_utils.cat_data([d3, d4])
    assert out2["x"].shape[0] == 2
    assert out2["edge_index"].shape[0] == d3["edge_index"].shape[0] * 2
    assert out2["edge_index"].shape[1] == d3["edge_index"].shape[1]
    assert out2["meta"] == "keep"


def test_concat_data_paths():
    assert deep_utils.concat_data([]) == []

    out_tensor = deep_utils.concat_data([torch.zeros((1, 2)), torch.ones((1, 2))])
    assert out_tensor.shape == (2, 2)

    g1 = {
        "x": torch.zeros((2, 2)),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_weight": torch.tensor([0.5, 0.5]),
        "mask": torch.tensor([1, 0], dtype=torch.bool),
        "graph_feat": torch.tensor([1.0]),
        "meta": "keep",
    }
    g2 = {
        "x": torch.ones((1, 2)),
        "edge_index": [[0], [0]],
        "edge_weight": [1.0],
        "mask": torch.tensor([0], dtype=torch.bool),
        "graph_feat": torch.tensor([2.0]),
        "meta": "keep2",
    }
    out_graph = deep_utils.concat_data([g1, g2])
    assert out_graph["x"].shape == (3, 2)
    assert out_graph["edge_index"].shape[1] == 3
    assert out_graph["edge_weight"].shape[0] == 3
    assert out_graph["mask"].shape[0] == 3
    assert out_graph["graph_feat"].shape == g1["graph_feat"].shape
    assert out_graph["meta"] == "keep"

    out_mixed = deep_utils.concat_data([g1, torch.zeros((1, 2))])
    assert out_mixed["x"].shape == (3, 2)

    g3 = {
        "x": torch.zeros((1, 2)),
        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
    }
    g4 = {
        "x": torch.ones((1, 2)),
        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
    }
    out_no_weights = deep_utils.concat_data([g3, g4])
    assert out_no_weights["x"].shape == (2, 2)
    assert "edge_weight" not in out_no_weights


def test_cycle_batches_with_dict():
    gen = torch.Generator().manual_seed(0)
    X = {"x": torch.arange(8, dtype=torch.float32).view(4, 2), "meta": "keep"}
    y = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    batches = list(deep_utils.cycle_batches(X, y, batch_size=2, generator=gen, steps=1))
    batch_x, batch_y = batches[0]
    assert batch_x["x"].shape == (2, 2)
    assert batch_y.shape == (2,)


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
