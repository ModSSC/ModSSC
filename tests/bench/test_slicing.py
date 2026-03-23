from __future__ import annotations

import sys
import types

import numpy as np
import torch

from bench.orchestrators.slicing import select_rows


def test_select_rows_graph_passes_num_nodes_for_empty_edge_index(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def subgraph(subset, edge_index, relabel_nodes=True, num_nodes=None):
        calls["subset"] = subset.detach().cpu().tolist()
        calls["edge_shape"] = tuple(edge_index.shape)
        calls["relabel_nodes"] = relabel_nodes
        calls["num_nodes"] = num_nodes
        return edge_index, None

    utils = types.ModuleType("torch_geometric.utils")
    utils.subgraph = subgraph
    tg = types.ModuleType("torch_geometric")
    tg.utils = utils
    monkeypatch.setitem(sys.modules, "torch_geometric", tg)
    monkeypatch.setitem(sys.modules, "torch_geometric.utils", utils)

    X = {
        "x": torch.randn(7, 3),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }

    out = select_rows(
        X,
        np.array([0], dtype=np.int64),
        context="test_select_rows_graph_passes_num_nodes_for_empty_edge_index",
    )

    assert out["x"].shape == (1, 3)
    assert tuple(out["edge_index"].shape) == (2, 0)
    assert calls == {
        "subset": [0],
        "edge_shape": (2, 0),
        "relabel_nodes": True,
        "num_nodes": 7,
    }
