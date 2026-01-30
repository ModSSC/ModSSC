from __future__ import annotations

import sys
import types

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from modssc.supervised.backends.torch.graphsage_inductive import TorchGraphSAGEClassifier


def _install_fake_pyg(monkeypatch, *, data_to_raises: bool = False):
    data_mod = types.ModuleType("torch_geometric.data")
    nn_mod = types.ModuleType("torch_geometric.nn")
    loader_mod = types.ModuleType("torch_geometric.loader")

    class Data:
        def __init__(self, x, edge_index):
            self.x = x
            self.edge_index = edge_index
            self.num_features = int(x.shape[1])

        def to(self, device):
            if data_to_raises:
                raise RuntimeError("stop")
            return self

    class SAGEConv(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)

        def forward(self, x, edge_index):
            return self.lin(x)

    class GraphSAGE:
        pass

    class NeighborLoader:
        pass

    data_mod.Data = Data
    nn_mod.SAGEConv = SAGEConv
    nn_mod.GraphSAGE = GraphSAGE
    loader_mod.NeighborLoader = NeighborLoader

    pyg = types.ModuleType("torch_geometric")
    pyg.data = data_mod
    pyg.nn = nn_mod
    pyg.loader = loader_mod

    monkeypatch.setitem(sys.modules, "torch_geometric", pyg)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_mod)
    monkeypatch.setitem(sys.modules, "torch_geometric.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "torch_geometric.loader", loader_mod)


def test_graphsage_fit_predict_dict_cpu(monkeypatch):
    _install_fake_pyg(monkeypatch)
    X = {
        "x": np.random.randn(3, 4).astype(np.float32),
        "edge_index": np.array([[0, 1], [1, 2]], dtype=np.int64),
        "edge_weight": np.array([1.0, 1.0], dtype=np.float32),
    }
    y = np.array([0, 1, 0], dtype=np.int64)
    clf = TorchGraphSAGEClassifier(max_epochs=1, batch_size=2, n_jobs=0, seed=0, num_layers=3)
    assert clf.supports_proba
    fit = clf.fit(X, y)
    assert fit.n_samples == 3

    proba = clf.predict_proba(X)
    assert proba.shape[0] == 3

    proba2 = clf.predict_proba(np.random.randn(2, 4).astype(np.float32))
    assert proba2.shape[0] == 2

    pred = clf.predict(X)
    assert pred.shape[0] == 3


def test_graphsage_init_hidden_sizes_sets_layers():
    clf = TorchGraphSAGEClassifier(hidden_sizes=[32, 16])
    assert clf.hidden_channels == 32
    assert clf.num_layers == 3


def test_graphsage_init_hidden_sizes_with_matching_num_layers():
    clf = TorchGraphSAGEClassifier(hidden_sizes=[32, 16], num_layers=3)
    assert clf.hidden_channels == 32
    assert clf.num_layers == 3


def test_graphsage_init_defaults_when_none():
    clf = TorchGraphSAGEClassifier(hidden_channels=None, num_layers=None)
    assert clf.hidden_channels == 128
    assert clf.num_layers == 2


def test_graphsage_init_hidden_sizes_int():
    clf = TorchGraphSAGEClassifier(hidden_sizes=32)
    assert clf.hidden_channels == 32
    assert clf.num_layers == 2


def test_graphsage_init_hidden_sizes_invalid_type():
    with pytest.raises(ValueError, match="hidden_sizes must be an int"):
        TorchGraphSAGEClassifier(hidden_sizes="bad")


def test_graphsage_fit_requires_dict(monkeypatch):
    _install_fake_pyg(monkeypatch)
    clf = TorchGraphSAGEClassifier(max_epochs=1, batch_size=1, n_jobs=0)
    with pytest.raises(ValueError, match="requires a dictionary"):
        clf.fit(np.zeros((2, 2), dtype=np.float32), np.array([0, 1], dtype=np.int64))


def test_graphsage_hidden_sizes_multi_layer(monkeypatch):
    _install_fake_pyg(monkeypatch)
    X = {"x": np.random.randn(3, 4).astype(np.float32), "edge_index": np.array([[0], [1]])}
    y = np.array([0, 1, 0], dtype=np.int64)
    clf = TorchGraphSAGEClassifier(hidden_sizes=[8, 4], max_epochs=1, batch_size=1, n_jobs=0)
    clf.fit(X, y)
    assert len(clf._model.convs) == 3
    assert clf._model.convs[0].lin.out_features == 8
    assert clf._model.convs[1].lin.out_features == 4


def test_graphsage_hidden_sizes_num_layers_mismatch(monkeypatch):
    _install_fake_pyg(monkeypatch)
    with pytest.raises(ValueError, match="num_layers must equal"):
        TorchGraphSAGEClassifier(hidden_sizes=[8, 4], num_layers=2)


def test_graphsage_hidden_sizes_negative(monkeypatch):
    _install_fake_pyg(monkeypatch)
    with pytest.raises(ValueError, match="hidden_sizes must be positive"):
        TorchGraphSAGEClassifier(hidden_sizes=[8, -1])


def test_graphsage_activation_param(monkeypatch):
    _install_fake_pyg(monkeypatch)
    X = {"x": np.random.randn(2, 3).astype(np.float32), "edge_index": np.array([[0], [1]])}
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchGraphSAGEClassifier(max_epochs=1, batch_size=1, n_jobs=0, activation="gelu")
    clf.fit(X, y)


def test_graphsage_activation_tanh(monkeypatch):
    _install_fake_pyg(monkeypatch)
    X = {"x": np.random.randn(2, 3).astype(np.float32), "edge_index": np.array([[0], [1]])}
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchGraphSAGEClassifier(max_epochs=1, batch_size=1, n_jobs=0, activation="tanh")
    clf.fit(X, y)


def test_graphsage_activation_unknown(monkeypatch):
    _install_fake_pyg(monkeypatch)
    X = {"x": np.random.randn(2, 3).astype(np.float32), "edge_index": np.array([[0], [1]])}
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchGraphSAGEClassifier(max_epochs=1, batch_size=1, n_jobs=0, activation="nope")
    with pytest.raises(ValueError, match="Unknown activation"):
        clf.fit(X, y)


def test_graphsage_predict_proba_invalid_input(monkeypatch):
    _install_fake_pyg(monkeypatch)
    X = {"x": np.random.randn(2, 3).astype(np.float32), "edge_index": np.array([[0], [1]])}
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchGraphSAGEClassifier(max_epochs=1, batch_size=1, n_jobs=0)
    clf.fit(X, y)
    with pytest.raises(ValueError, match="Invalid input X"):
        clf.predict_proba(object())


def test_graphsage_predict_numpy_branch(monkeypatch):
    _install_fake_pyg(monkeypatch)
    X = {"x": np.random.randn(2, 3).astype(np.float32), "edge_index": np.array([[0], [1]])}
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchGraphSAGEClassifier(max_epochs=1, batch_size=1, n_jobs=0)
    clf.fit(X, y)
    monkeypatch.setattr(clf, "predict_proba", lambda _x: np.array([[0.2, 0.8]]))
    pred = clf.predict(X)
    assert pred.shape[0] == 1


def test_graphsage_seed_none_branch(monkeypatch):
    _install_fake_pyg(monkeypatch)
    X = {"x": np.random.randn(2, 3).astype(np.float32), "edge_index": np.array([[0], [1]])}
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchGraphSAGEClassifier(max_epochs=1, batch_size=1, n_jobs=0, seed=None)
    clf.fit(X, y)


def test_graphsage_torch_geometric_import_error(monkeypatch):
    import modssc.supervised.backends.torch.graphsage_inductive as gs

    tg = types.ModuleType("torch_geometric")
    loader_mod = types.ModuleType("torch_geometric.loader")
    nn_mod = types.ModuleType("torch_geometric.nn")
    monkeypatch.setitem(sys.modules, "torch_geometric", tg)
    monkeypatch.setitem(sys.modules, "torch_geometric.loader", loader_mod)
    monkeypatch.setitem(sys.modules, "torch_geometric.nn", nn_mod)

    with pytest.raises(ImportError, match="torch_geometric is required"):
        gs._torch_geometric()


def test_graphsage_cuda_branch(monkeypatch):
    _install_fake_pyg(monkeypatch, data_to_raises=True)
    X = {"x": np.random.randn(2, 3).astype(np.float32), "edge_index": np.array([[0], [1]])}
    y = np.array([0, 1], dtype=np.int64)
    clf = TorchGraphSAGEClassifier(max_epochs=1, batch_size=1, n_jobs=1, seed=0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="stop"):
        clf.fit(X, y)
