from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from modssc.data_loader.providers.torchvision import (
    TorchvisionProvider,
    _apply_class_filter,
    _extract_xy,
    _limit_samples,
    _make_train_test,
    _normalize_filter,
)
from modssc.data_loader.types import DatasetIdentity
from modssc.data_loader.uri import ParsedURI


def test_make_train_test_train_arg():
    mock_cls = MagicMock()
    mock_cls.return_value = "ds_instance"

    train, test = _make_train_test(mock_cls, Path("/tmp"))

    assert train == "ds_instance"
    assert test == "ds_instance"
    assert mock_cls.call_count == 2
    calls = mock_cls.call_args_list
    assert calls[0][1]["train"] is True
    assert calls[1][1]["train"] is False


def test_make_train_test_split_arg():
    mock_cls = MagicMock()

    def side_effect(*args, **kwargs):
        if "train" in kwargs:
            raise TypeError("unexpected keyword argument 'train'")
        return "ds_instance"

    mock_cls.side_effect = side_effect

    train, test = _make_train_test(mock_cls, Path("/tmp"))

    assert train == "ds_instance"
    assert test == "ds_instance"

    assert mock_cls.call_count == 3

    calls = mock_cls.call_args_list

    assert "train" in calls[0][1]
    assert calls[1][1]["split"] == "train"
    assert calls[2][1]["split"] == "test"


def test_extract_xy_attributes():
    ds = MagicMock()
    ds.data = [1, 2]
    ds.targets = [0, 1]

    X, y = _extract_xy(ds)

    assert np.array_equal(X, np.array([1, 2]))
    assert np.array_equal(y, np.array([0, 1]))


def test_extract_xy_fallback():
    ds = [(10, 0), (20, 1)]

    X, y = _extract_xy(ds)

    assert np.array_equal(X, np.array([10, 20], dtype=object))
    assert np.array_equal(y, np.array([0, 1]))


def test_resolve():
    provider = TorchvisionProvider()
    uri = ParsedURI(provider="torchvision", reference="MNIST")
    identity = provider.resolve(uri, options={"task": "classification"})

    assert identity.canonical_uri == "torchvision:MNIST"
    assert identity.dataset_id == "MNIST"
    assert identity.task == "classification"


def test_load_canonical(tmp_path):
    provider = TorchvisionProvider()
    identity = DatasetIdentity(
        provider="torchvision",
        canonical_uri="torchvision:TEST",
        dataset_id="TEST",
        version=None,
        modality="vision",
        task="classification",
        resolved_kwargs={"dataset_class": "TEST"},
    )

    with patch("modssc.data_loader.providers.torchvision.optional_import") as mock_import:
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        mock_module.TEST = MagicMock()

        with patch("modssc.data_loader.providers.torchvision._make_train_test") as mock_mtt:
            mock_train = MagicMock()
            mock_test = MagicMock()
            mock_mtt.return_value = (mock_train, mock_test)

            with patch("modssc.data_loader.providers.torchvision._extract_xy") as mock_ex:
                mock_ex.side_effect = [
                    (np.array(["t1"]), np.array([0])),
                    (np.array(["t2"]), np.array([1])),
                ]

                ds = provider.load_canonical(identity, raw_dir=tmp_path)

                assert ds.train.X[0] == "t1"
                assert ds.test.X[0] == "t2"
                assert ds.meta["provider"] == "torchvision"


def test_normalize_filter_and_limits():
    assert _normalize_filter(None) is None
    assert set(_normalize_filter({1, 2})) == {1, 2}
    assert _normalize_filter("x") == ["x"]

    X = np.array(["a", "b", "c"])
    y = np.array([0, 1, 0])
    X_f, y_f = _apply_class_filter(X, y, class_filter=[0])
    assert X_f.tolist() == ["a", "c"]
    assert y_f.tolist() == [0, 0]

    X_empty, y_empty = _limit_samples(X, y, max_samples=0, seed=None)
    assert X_empty.size == 0
    assert y_empty.size == 0

    X_take_no_seed, y_take_no_seed = _limit_samples(X, y, max_samples=2, seed=None)
    assert X_take_no_seed.shape == (2,)
    assert y_take_no_seed.shape == (2,)

    X_take, y_take = _limit_samples(X, y, max_samples=2, seed=123)
    assert X_take.shape == (2,)
    assert y_take.shape == (2,)
