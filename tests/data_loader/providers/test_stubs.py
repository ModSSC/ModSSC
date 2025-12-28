from __future__ import annotations

import sys
import types

import numpy as np

from modssc.data_loader.providers.hf import HuggingFaceDatasetsProvider
from modssc.data_loader.providers.openml import OpenMLProvider
from modssc.data_loader.providers.pyg import PyGProvider
from modssc.data_loader.providers.tfds import TFDSProvider
from modssc.data_loader.providers.torchaudio import TorchaudioProvider
from modssc.data_loader.providers.torchvision import TorchvisionProvider
from modssc.data_loader.uri import ParsedURI


def test_openml_provider_stub(monkeypatch, tmp_path) -> None:
    skl = types.ModuleType("sklearn")
    datasets = types.ModuleType("sklearn.datasets")

    def fetch_openml(**kwargs):
        assert kwargs.get("data_home") is not None
        return np.array([[1.0, 2.0]]), np.array([0])

    datasets.fetch_openml = fetch_openml
    skl.datasets = datasets

    monkeypatch.setitem(sys.modules, "sklearn", skl)
    monkeypatch.setitem(sys.modules, "sklearn.datasets", datasets)

    p = OpenMLProvider()
    ident = p.resolve(ParsedURI("openml", "61"), options={})
    ds = p.load_canonical(ident, raw_dir=tmp_path)
    assert ds.train.X.shape == (1, 2)


def test_hf_provider_stub(monkeypatch, tmp_path) -> None:
    ds_mod = types.ModuleType("datasets")

    def load_dataset(name, *args, **kwargs):
        return {
            "train": {"text": ["a", "b"], "label": [0, 1]},
            "test": {"text": ["c"], "label": [1]},
        }

    ds_mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", ds_mod)

    p = HuggingFaceDatasetsProvider()
    ident = p.resolve(ParsedURI("hf", "ag_news"), options={})
    ds = p.load_canonical(ident, raw_dir=tmp_path)
    assert ds.test is not None
    assert ds.train.X.tolist() == ["a", "b"]


def test_tfds_provider_stub(monkeypatch, tmp_path) -> None:
    tfds_mod = types.ModuleType("tensorflow_datasets")

    class FakeDS:
        def __init__(self, items):
            self._items = items

        def __iter__(self):
            return iter(self._items)

    def load(name, split, as_supervised, data_dir):
        if split == "train":
            return FakeDS([(1, 0), (2, 1)])
        if split == "test":
            return FakeDS([(3, 1)])
        raise ValueError

    tfds_mod.load = load
    monkeypatch.setitem(sys.modules, "tensorflow_datasets", tfds_mod)

    p = TFDSProvider()
    ident = p.resolve(ParsedURI("tfds", "mnist/1.0.0"), options={})
    ds = p.load_canonical(ident, raw_dir=tmp_path)
    assert ds.test is not None
    assert ds.train.X.shape[0] == 2


def test_torchvision_provider_stub(monkeypatch, tmp_path) -> None:
    tv = types.ModuleType("torchvision")
    tvds = types.ModuleType("torchvision.datasets")

    class FakeDataset:
        def __init__(self, root, train=True, download=True):
            self.data = np.array([[1], [2]]) if train else np.array([[3]])
            self.targets = np.array([0, 1]) if train else np.array([1])

    tvds.MNIST = FakeDataset
    tv.datasets = tvds
    monkeypatch.setitem(sys.modules, "torchvision", tv)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", tvds)

    p = TorchvisionProvider()
    ident = p.resolve(ParsedURI("torchvision", "MNIST"), options={})
    ds = p.load_canonical(ident, raw_dir=tmp_path)
    assert ds.test is not None
    assert ds.train.X.shape == (2, 1)


def test_torchaudio_provider_stub(monkeypatch, tmp_path) -> None:
    ta = types.ModuleType("torchaudio")
    tads = types.ModuleType("torchaudio.datasets")

    class WithSubset:
        def __init__(self, root, subset=None, download=True):
            self._items = (
                [(np.zeros((1,)), 16000, "yes")]
                if subset == "training"
                else [(np.ones((1,)), 16000, "no")]
            )

        def __iter__(self):
            return iter(self._items)

    class NoSubset:
        def __init__(self, root, download=True):
            self._items = [(np.zeros((1,)), 16000, "yes")]

        def __iter__(self):
            return iter(self._items)

    tads.SPEECHCOMMANDS = WithSubset
    tads.YESNO = NoSubset
    ta.datasets = tads

    monkeypatch.setitem(sys.modules, "torchaudio", ta)
    monkeypatch.setitem(sys.modules, "torchaudio.datasets", tads)

    p = TorchaudioProvider()
    ident = p.resolve(ParsedURI("torchaudio", "SPEECHCOMMANDS"), options={})
    ds = p.load_canonical(ident, raw_dir=tmp_path)
    assert ds.test is not None

    ident2 = p.resolve(ParsedURI("torchaudio", "YESNO"), options={})
    ds2 = p.load_canonical(ident2, raw_dir=tmp_path)
    assert ds2.test is None


def test_pyg_provider_stub(monkeypatch, tmp_path) -> None:
    tg = types.ModuleType("torch_geometric")
    tgds = types.ModuleType("torch_geometric.datasets")

    class FakeTensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class Data:
        def __init__(self):
            self.x = FakeTensor([[1.0], [2.0]])
            self.y = FakeTensor([0, 1])
            self.edge_index = FakeTensor([[0, 1], [1, 0]])
            self.train_mask = FakeTensor([True, False])
            self.val_mask = FakeTensor([False, True])
            self.test_mask = FakeTensor([False, True])

    class Planetoid:
        def __init__(self, root, name):
            self.root = root
            self.name = name
            self._items = [Data()]

        def __getitem__(self, idx):
            return self._items[idx]

        def __len__(self):
            return len(self._items)

    tgds.Planetoid = Planetoid
    tg.datasets = tgds

    monkeypatch.setitem(sys.modules, "torch_geometric", tg)
    monkeypatch.setitem(sys.modules, "torch_geometric.datasets", tgds)

    p = PyGProvider()
    ident = p.resolve(ParsedURI("pyg", "Planetoid/Cora"), options={})
    ds = p.load_canonical(ident, raw_dir=tmp_path)
    assert ds.train.masks is not None
    assert set(ds.train.masks.keys()) == {"train", "val", "test"}
