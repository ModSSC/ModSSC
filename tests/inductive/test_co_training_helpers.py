from __future__ import annotations

import sys
import types

import pytest
import torch

from modssc.inductive.errors import InductiveValidationError
from modssc.inductive.methods import co_training as ct


def _install_fake_tg_utils(monkeypatch, *, with_subgraph: bool):
    utils = types.ModuleType("torch_geometric.utils")
    if with_subgraph:

        def subgraph(idx, edge_index, relabel_nodes=True, num_nodes=None):
            return edge_index, None

        utils.subgraph = subgraph

    tg = types.ModuleType("torch_geometric")
    tg.utils = utils
    monkeypatch.setitem(sys.modules, "torch_geometric", tg)
    monkeypatch.setitem(sys.modules, "torch_geometric.utils", utils)


def test_get_torch_helpers_dict():
    x = {"x": torch.zeros((3, 2))}
    assert ct._get_torch_tensor(x).shape == (3, 2)
    assert ct._get_torch_len(x) == 3
    assert ct._get_torch_device(x) == x["x"].device


def test_same_device_branch_index_none():
    dev = torch.device("cuda")
    dev0 = torch.device("cuda:0")
    assert dev != dev0
    assert ct._same_device(dev, dev0)


def test_index_torch_requires_pyg(monkeypatch):
    _install_fake_tg_utils(monkeypatch, with_subgraph=False)
    x = {
        "x": torch.zeros((3, 2)),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    }
    with pytest.raises(InductiveValidationError, match="PyG is required"):
        ct._index_torch(x, torch.tensor([0, 1]))


def test_index_torch_with_subgraph_and_slice(monkeypatch):
    _install_fake_tg_utils(monkeypatch, with_subgraph=True)
    x = {
        "x": torch.arange(6, dtype=torch.float32).view(3, 2),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "extra": torch.ones((3, 1)),
    }
    out = ct._index_torch(x, slice(0, 2))
    assert out["x"].shape == (2, 2)
    assert out["extra"].shape == (3, 1)
    assert "edge_index" in out


def test_index_torch_with_subgraph_and_tensor_idx(monkeypatch):
    _install_fake_tg_utils(monkeypatch, with_subgraph=True)
    x = {
        "x": torch.arange(6, dtype=torch.float32).view(3, 2),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    }
    out = ct._index_torch(x, torch.tensor([0, 2], dtype=torch.long))
    assert out["x"].shape == (2, 2)
    assert "edge_index" in out


def test_index_torch_without_edge_index():
    x = {"x": torch.zeros((3, 2)), "meta": "keep"}
    out = ct._index_torch(x, torch.tensor([0, 1], dtype=torch.long))
    assert out["x"].shape == (2, 2)
    assert out["meta"] == "keep"


def test_cat_torch_with_edge_index():
    d1 = {
        "x": torch.zeros((2, 2)),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    }
    d2 = {
        "x": torch.ones((1, 2)),
        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
    }
    out = ct._cat_torch([d1, d2])
    assert out["x"].shape[0] == 3
    # edge_index should be concatenated along dim=1
    assert out["edge_index"].shape[1] == d1["edge_index"].shape[1] + d2["edge_index"].shape[1]


def test_cat_torch_without_edge_index():
    d1 = {"x": torch.zeros((2, 2))}
    d2 = {"x": torch.ones((1, 2))}
    out = ct._cat_torch([d1, d2])
    assert out["x"].shape[0] == 3
