from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.data_augmentation.ops.audio import AddNoise, TimeShift
from modssc.data_augmentation.ops.core import EnsureFloat32
from modssc.data_augmentation.ops.graph import EdgeDropout, FeatureMask
from modssc.data_augmentation.ops.tabular import FeatureDropout, GaussianNoise
from modssc.data_augmentation.ops.vision import (
    Cutout,
    RandomCropPad,
    RandomHorizontalFlip,
)
from modssc.data_augmentation.ops.vision import (
    GaussianNoise as VisionGaussianNoise,
)
from modssc.data_augmentation.types import AugmentationContext


@pytest.fixture
def ctx():
    return AugmentationContext(seed=0, epoch=0, sample_id=0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def mock_torch_module():
    torch = MagicMock()
    torch.float32 = "float32"

    torch.from_numpy.return_value = MagicMock()
    torch.as_tensor.return_value = MagicMock()

    return torch


@pytest.fixture
def mock_tensor():
    tensor = MagicMock()
    tensor.to.return_value = tensor
    tensor.clone.return_value = tensor
    tensor.detach.return_value = tensor
    tensor.cpu.return_value = tensor
    tensor.numpy.return_value = np.array([1])
    tensor.device = "cpu"
    tensor.dtype = "float32"

    tensor.__add__.return_value = tensor
    tensor.__mul__.return_value = tensor

    return tensor


def test_vision_ops_torch(ctx, rng, mock_torch_module, mock_tensor):
    with (
        patch("modssc.data_augmentation.ops.vision.is_torch_tensor", return_value=True),
        patch("importlib.import_module", return_value=mock_torch_module),
    ):
        op = RandomHorizontalFlip(p=1.0)

        mock_tensor.shape = (10, 10)
        mock_tensor.ndim = 2
        op.apply(mock_tensor, rng=rng, ctx=ctx)
        mock_tensor.flip.assert_called_with((-1,))

        mock_tensor.shape = (3, 10, 10)
        mock_tensor.ndim = 3
        op.apply(mock_tensor, rng=rng, ctx=ctx)
        mock_tensor.flip.assert_called_with((-1,))

        mock_tensor.shape = (10, 10, 3)
        mock_tensor.ndim = 3
        op.apply(mock_tensor, rng=rng, ctx=ctx)
        mock_tensor.flip.assert_called_with((-2,))

        op = VisionGaussianNoise(std=0.1)
        mock_tensor.shape = (10, 10, 3)
        op.apply(mock_tensor, rng=rng, ctx=ctx)
        assert mock_tensor.add.called

        op = Cutout(frac=0.5)

        mock_tensor.shape = (10, 10)
        mock_tensor.ndim = 2
        op.apply(mock_tensor, rng=rng, ctx=ctx)
        assert mock_tensor.clone.called

        mock_tensor.shape = (3, 10, 10)
        mock_tensor.ndim = 3
        op.apply(mock_tensor, rng=rng, ctx=ctx)

        mock_tensor.shape = (10, 10, 3)
        mock_tensor.ndim = 3
        op.apply(mock_tensor, rng=rng, ctx=ctx)

        op = RandomCropPad(pad=2)

        mock_torch_module.nn.functional.pad.return_value = mock_tensor
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.squeeze.return_value = mock_tensor
        mock_tensor.permute.return_value = mock_tensor

        mock_tensor.shape = (1, 14, 14)
        mock_tensor.ndim = 3

        input_tensor = MagicMock()
        input_tensor.shape = (10, 10)
        input_tensor.ndim = 2
        input_tensor.unsqueeze.return_value = input_tensor
        input_tensor.device = "cpu"

        mock_torch_module.nn.functional.pad.return_value = mock_tensor

        op.apply(input_tensor, rng=rng, ctx=ctx)

        input_tensor.shape = (3, 10, 10)
        input_tensor.ndim = 3
        mock_tensor.shape = (3, 14, 14)
        op.apply(input_tensor, rng=rng, ctx=ctx)

        input_tensor.shape = (10, 10, 3)
        input_tensor.ndim = 3
        input_tensor.permute.return_value = input_tensor
        mock_tensor.permute.return_value = mock_tensor
        mock_tensor.shape = (14, 14, 3)

        op.apply(input_tensor, rng=rng, ctx=ctx)


def test_audio_ops_torch(ctx, rng, mock_torch_module, mock_tensor):
    with (
        patch("modssc.data_augmentation.ops.audio.is_torch_tensor", return_value=True),
        patch("importlib.import_module", return_value=mock_torch_module),
    ):
        op = AddNoise(std=0.1)
        mock_tensor.shape = (100,)
        op.apply(mock_tensor, rng=rng, ctx=ctx)
        assert mock_tensor.__add__.called

        op = TimeShift(max_frac=0.1)
        mock_tensor.shape = (100,)
        mock_tensor.roll.return_value = mock_tensor
        op.apply(mock_tensor, rng=rng, ctx=ctx)
        mock_tensor.roll.assert_called()


def test_tabular_ops_torch(ctx, rng, mock_torch_module, mock_tensor):
    with (
        patch("modssc.data_augmentation.ops.tabular.is_torch_tensor", return_value=True),
        patch("importlib.import_module", return_value=mock_torch_module),
    ):
        op = GaussianNoise(std=0.1)
        mock_tensor.shape = (10,)
        op.apply(mock_tensor, rng=rng, ctx=ctx)
        assert mock_tensor.__add__.called

        op = FeatureDropout(p=0.5)
        op.apply(mock_tensor, rng=rng, ctx=ctx)
        assert mock_tensor.__mul__.called


def test_core_ops_torch(ctx, rng, mock_torch_module, mock_tensor):
    with (
        patch("modssc.data_augmentation.ops.core.is_torch_tensor", return_value=True),
        patch("importlib.import_module", return_value=mock_torch_module),
    ):
        op = EnsureFloat32()
        op.apply(mock_tensor, rng=rng, ctx=ctx)
        mock_tensor.to.assert_called_with(dtype="float32")


def test_graph_ops_torch(ctx, rng, mock_torch_module, mock_tensor):
    with (
        patch("modssc.data_augmentation.ops.graph.is_torch_tensor", return_value=True),
        patch("importlib.import_module", return_value=mock_torch_module),
    ):

        class MockGraph:
            def __init__(self):
                self.x = mock_tensor
                self.edge_index = "edge_index"
                self.edge_weight = "edge_weight"

        g = MockGraph()

        op = FeatureMask(p=0.5)
        mock_tensor.shape = (10, 10)

        res = op.apply(g, rng=rng, ctx=ctx)

        assert mock_tensor.__mul__.called

        assert res.x is mock_tensor


def test_graph_edge_dropout_torch_preserves_backend(ctx, rng):
    from modssc.data_augmentation.types import GraphSample

    class FakeTensor:
        __module__ = "torch.tensor"

        def __init__(self, arr):
            self._arr = np.asarray(arr)
            self.shape = self._arr.shape
            self.ndim = self._arr.ndim
            self.dtype = self._arr.dtype
            self.device = "cpu"

        def t(self):
            return FakeTensor(self._arr.T)

        def reshape(self, *shape):
            return FakeTensor(self._arr.reshape(*shape))

        def index_select(self, dim, index):
            idx = np.asarray(index._arr, dtype=np.int64)
            if dim == 0:
                data = self._arr[idx]
            elif dim == 1:
                data = self._arr[:, idx]
            else:
                raise ValueError("dim must be 0 or 1")
            return FakeTensor(data)

        def __getitem__(self, item):
            return FakeTensor(self._arr[item])

    class FakeTorch:
        long = "long"

        def as_tensor(self, data, device=None, dtype=None):
            return FakeTensor(np.asarray(data))

    x = FakeTensor(np.ones((4, 2), dtype=np.float32))
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    edge_weight = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    g = GraphSample(x=x, edge_index=edge_index, edge_weight=edge_weight)

    with patch("importlib.import_module", return_value=FakeTorch()):
        op = EdgeDropout(p=0.5)
        out = op.apply(g, rng=rng, ctx=ctx)

    assert isinstance(out.edge_index, FakeTensor)
    assert out.edge_index.shape[0] == 2
    if out.edge_weight is not None:
        assert isinstance(out.edge_weight, FakeTensor)


def test_graph_edge_dropout_torch_without_edge_weight(ctx, rng):
    from modssc.data_augmentation.types import GraphSample

    class FakeTensor:
        __module__ = "torch.tensor"

        def __init__(self, arr):
            self._arr = np.asarray(arr)
            self.shape = self._arr.shape
            self.ndim = self._arr.ndim
            self.dtype = self._arr.dtype
            self.device = "cpu"

        def t(self):
            return FakeTensor(self._arr.T)

        def reshape(self, *shape):
            return FakeTensor(self._arr.reshape(*shape))

        def index_select(self, dim, index):
            idx = np.asarray(index._arr, dtype=np.int64)
            if dim == 0:
                data = self._arr[idx]
            elif dim == 1:
                data = self._arr[:, idx]
            else:
                raise ValueError("dim must be 0 or 1")
            return FakeTensor(data)

        def __getitem__(self, item):
            return FakeTensor(self._arr[item])

    class FakeTorch:
        long = "long"

        def as_tensor(self, data, device=None, dtype=None):
            return FakeTensor(np.asarray(data))

    x = FakeTensor(np.ones((4, 2), dtype=np.float32))
    edge_index = FakeTensor(np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64))
    g = GraphSample(x=x, edge_index=edge_index, edge_weight=None)

    with patch("importlib.import_module", return_value=FakeTorch()):
        out = EdgeDropout(p=0.5).apply(g, rng=rng, ctx=ctx)

    assert isinstance(out.edge_index, FakeTensor)
    assert out.edge_weight is None


def test_graph_edge_dropout_torch_device_loop_no_match(ctx, rng):
    from modssc.data_augmentation.types import GraphSample

    class FakeTensor:
        __module__ = "torch.tensor"

        def __init__(self, arr):
            self._arr = np.asarray(arr)
            self.shape = self._arr.shape
            self.ndim = self._arr.ndim
            self.dtype = self._arr.dtype
            self.device = "cpu"

        def t(self):
            return FakeTensor(self._arr.T)

        def reshape(self, *shape):
            return FakeTensor(self._arr.reshape(*shape))

        def index_select(self, dim, index):
            idx = np.asarray(index._arr, dtype=np.int64)
            if dim == 0:
                data = self._arr[idx]
            elif dim == 1:
                data = self._arr[:, idx]
            else:
                raise ValueError("dim must be 0 or 1")
            return FakeTensor(data)

        def __getitem__(self, item):
            return FakeTensor(self._arr[item])

    class FakeTorch:
        long = "long"

        def as_tensor(self, data, device=None, dtype=None):
            return FakeTensor(np.asarray(data))

    call_state = {"count": 0}

    def fake_is_torch_tensor(_obj):
        call_state["count"] += 1
        return call_state["count"] == 1

    x = np.ones((4, 2), dtype=np.float32)
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    edge_weight = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    g = GraphSample(x=x, edge_index=edge_index, edge_weight=edge_weight)

    with (
        patch(
            "modssc.data_augmentation.ops.graph.is_torch_tensor", side_effect=fake_is_torch_tensor
        ),
        patch("importlib.import_module", return_value=FakeTorch()),
    ):
        out = EdgeDropout(p=0.5).apply(g, rng=rng, ctx=ctx)

    assert isinstance(out.edge_index, FakeTensor)
