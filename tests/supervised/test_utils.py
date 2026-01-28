from unittest.mock import MagicMock

import numpy as np
import pytest

from modssc.supervised.errors import SupervisedValidationError
from modssc.supervised.utils import as_numpy, encode_labels, ensure_2d, onehot, seed_everything


def test_as_numpy_torch_like():
    mock_tensor = MagicMock()
    mock_tensor.detach.return_value = mock_tensor
    mock_tensor.cpu.return_value = mock_tensor
    mock_tensor.numpy.return_value = np.array([1, 2, 3])

    res = as_numpy(mock_tensor)
    assert isinstance(res, np.ndarray)
    assert np.array_equal(res, np.array([1, 2, 3]))
    mock_tensor.detach.assert_called_once()
    mock_tensor.cpu.assert_called_once()
    mock_tensor.numpy.assert_called_once()


def test_as_numpy_others():
    arr = np.array([1, 2])
    assert as_numpy(arr) is arr

    lst = [1, 2]
    res = as_numpy(lst)
    assert isinstance(res, np.ndarray)
    assert np.array_equal(res, np.array([1, 2]))


def test_ensure_2d_high_dim():
    arr = np.zeros((2, 3, 4))
    res = ensure_2d(arr)
    assert res.shape == (2, 12)


def test_ensure_2d_low_dim():
    arr = np.array([1, 2, 3])
    res = ensure_2d(arr)
    assert res.shape == (3, 1)

    arr2 = np.zeros((2, 3))
    res2 = ensure_2d(arr2)
    assert res2 is arr2


def test_ensure_2d_invalid():
    with pytest.raises(SupervisedValidationError, match="got ndim=0"):
        ensure_2d(np.array(1.0))


def test_onehot_empty():
    res = onehot(np.array([]), n_classes=3)
    assert res.shape == (0, 3)
    assert res.size == 0


def test_encode_labels():
    y = ["b", "a", "b", "c"]
    y_enc, classes = encode_labels(y)
    np.testing.assert_array_equal(classes, np.array(["a", "b", "c"]))
    np.testing.assert_array_equal(y_enc, np.array([1, 0, 1, 2]))


def test_seed_everything_optional_import_failure(monkeypatch):
    monkeypatch.setattr(
        "modssc.supervised.utils.optional_import",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no torch")),
    )
    seed_everything(123)


def test_seed_everything_torch_paths(monkeypatch):
    class _FakeCuda:
        def __init__(self):
            self.seeded = None

        def is_available(self):
            return True

        def manual_seed_all(self, seed):
            self.seeded = seed

    class _FakeCudnn:
        deterministic = None
        benchmark = None

    class _FakeBackends:
        def __init__(self):
            self.cudnn = _FakeCudnn()

    class _FakeTorch:
        def __init__(self):
            self.cuda = _FakeCuda()
            self.backends = _FakeBackends()
            self.manual_seed_called = None
            self.det_used = None

        def manual_seed(self, seed):
            self.manual_seed_called = seed

        def use_deterministic_algorithms(self, flag):
            self.det_used = flag

    fake_torch = _FakeTorch()
    monkeypatch.setattr(
        "modssc.supervised.utils.optional_import", lambda *_args, **_kwargs: fake_torch
    )

    seed_everything(42, deterministic=True)
    assert fake_torch.manual_seed_called == 42
    assert fake_torch.cuda.seeded == 42
    assert fake_torch.det_used is True
    assert fake_torch.backends.cudnn.deterministic is True
    assert fake_torch.backends.cudnn.benchmark is False


def test_seed_everything_deterministic_false(monkeypatch):
    class _FakeTorch:
        def __init__(self):
            self.cuda = type("Cuda", (), {"is_available": lambda self: False})()

        def manual_seed(self, _seed):
            return None

    monkeypatch.setattr(
        "modssc.supervised.utils.optional_import", lambda *_args, **_kwargs: _FakeTorch()
    )
    seed_everything(123, deterministic=False)


def test_seed_everything_no_deterministic_attrs(monkeypatch):
    class _FakeTorch:
        def __init__(self):
            self.cuda = type("Cuda", (), {"is_available": lambda self: False})()
            self.backends = object()

        def manual_seed(self, _seed):
            return None

    monkeypatch.setattr(
        "modssc.supervised.utils.optional_import", lambda *_args, **_kwargs: _FakeTorch()
    )
    seed_everything(123, deterministic=True)
