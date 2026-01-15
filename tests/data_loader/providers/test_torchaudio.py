from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.data_loader.providers.torchaudio import (
    TorchaudioProvider,
    _apply_class_filter,
    _extract_xy,
    _labels_from_paths,
    _limit_samples,
    _normalize_filter,
    _official_or_single,
    _paths_from_walker,
)
from modssc.data_loader.types import DatasetIdentity
from modssc.data_loader.uri import ParsedURI


def test_official_or_single_subsets():
    mock_cls = MagicMock()
    mock_cls.return_value = "ds_instance"

    train, test = _official_or_single(mock_cls, Path("/tmp"))

    assert train == "ds_instance"
    assert test == "ds_instance"
    assert mock_cls.call_count == 2

    calls = mock_cls.call_args_list
    assert calls[0][1]["subset"] == "training"
    assert calls[1][1]["subset"] == "testing"


def test_official_or_single_no_subsets():
    mock_cls = MagicMock()

    def side_effect(*args, **kwargs):
        if "subset" in kwargs:
            raise TypeError("unexpected keyword argument 'subset'")
        return "ds_instance"

    mock_cls.side_effect = side_effect

    train, test = _official_or_single(mock_cls, Path("/tmp"))

    assert train == "ds_instance"
    assert test is None

    assert mock_cls.call_count == 2


def test_paths_from_walker_missing_attrs():
    ds = MagicMock()
    del ds._path
    del ds._walker
    assert _paths_from_walker(ds) is None

    ds._path = "/tmp"
    assert _paths_from_walker(ds) is None


def test_paths_from_walker_relative():
    ds = MagicMock()
    ds._path = "/base"
    ds._walker = ["file1.wav", "file2.wav"]

    paths = _paths_from_walker(ds)
    assert len(paths) == 2
    assert paths[0] == Path("/base/file1.wav")
    assert paths[1] == Path("/base/file2.wav")


def test_paths_from_walker_absolute():
    ds = MagicMock()
    ds._path = "/base"
    ds._walker = ["/abs/file1.wav"]

    paths = _paths_from_walker(ds)
    assert paths[0] == Path("/abs/file1.wav")


def test_paths_from_walker_extensions():
    ds = MagicMock()
    ds._path = "/base"
    ds._walker = ["file1", "file2"]

    with patch("pathlib.Path.exists", autospec=True) as mock_exists:

        def side_effect_impl(self):
            s = str(self)
            if s.endswith("file1.wav"):
                return True
            return bool(s.endswith("file2.flac"))

        mock_exists.side_effect = side_effect_impl

        paths = _paths_from_walker(ds)
        assert len(paths) == 2
        assert str(paths[0]).endswith("file1.wav")
        assert str(paths[1]).endswith("file2.flac")


def test_paths_from_walker_exists():
    ds = MagicMock()
    ds._path = "/base"
    ds._walker = ["file1.wav"]

    with patch("pathlib.Path.exists", autospec=True) as mock_exists:
        mock_exists.return_value = True

        paths = _paths_from_walker(ds)

        assert len(paths) == 1
        assert str(paths[0]) == "/base/file1.wav"


def test_labels_speechcommands():
    paths = [Path("/data/yes/001.wav"), Path("/data/no/002.wav")]
    labels = _labels_from_paths(paths, dataset_class="SPEECHCOMMANDS")
    assert labels == ["yes", "no"]


def test_labels_yesno():
    paths = [Path("/data/1_0_1.wav"), Path("/data/0_1_0.wav")]
    labels = _labels_from_paths(paths, dataset_class="YESNO")
    assert labels == ["yes", "no"]


def test_labels_fallback():
    paths = [Path("/data/cat/001.jpg"), Path("/data/dog/002.jpg")]
    labels = _labels_from_paths(paths, dataset_class="OTHER")
    assert labels == ["cat", "dog"]


def test_extract_xy_walker():
    ds = MagicMock()
    ds._path = "/base"
    ds._walker = ["cat/1.wav", "dog/2.wav"]

    X, y = _extract_xy(ds, dataset_class="OTHER")

    assert len(X) == 2
    assert str(X[0]) == "/base/cat/1.wav"
    assert y[0] == "cat"
    assert y[1] == "dog"


def test_extract_xy_fallback(tmp_path):
    ds = [(np.array([0.1, 0.2]), 16000, "cat"), (np.array([0.3, 0.4]), 16000, "dog")]

    with patch("modssc.data_loader.providers.torchaudio._paths_from_walker", return_value=None):
        X, y = _extract_xy(ds, dataset_class="OTHER")

        assert len(X) == 2
        assert np.allclose(X[0], np.array([0.1, 0.2]))
        assert y[0] == "cat"


def test_extract_xy_fallback_invalid_sample():
    ds = [(np.array([0.1]), 16000)]

    with (
        patch("modssc.data_loader.providers.torchaudio._paths_from_walker", return_value=None),
        pytest.raises(ValueError, match="Expected torchaudio sample"),
    ):
        _extract_xy(ds, dataset_class="OTHER")


def test_resolve():
    provider = TorchaudioProvider()
    uri = ParsedURI(provider="torchaudio", reference="SPEECHCOMMANDS")
    identity = provider.resolve(uri, options={"task": "kws"})

    assert identity.canonical_uri == "torchaudio:SPEECHCOMMANDS"
    assert identity.dataset_id == "SPEECHCOMMANDS"
    assert identity.task == "kws"


def test_resolve_whitespace_option():
    provider = TorchaudioProvider()
    uri = ParsedURI(provider="torchaudio", reference="SPEECHCOMMANDS")

    identity = provider.resolve(uri, options={"dataset_class": " "})

    assert identity.dataset_id == "SPEECHCOMMANDS"


def test_load_canonical(tmp_path):
    provider = TorchaudioProvider()
    identity = DatasetIdentity(
        provider="torchaudio",
        canonical_uri="torchaudio:TEST",
        dataset_id="TEST",
        version=None,
        modality="audio",
        task="classification",
        resolved_kwargs={"dataset_class": "TEST"},
    )

    with patch("modssc.data_loader.providers.torchaudio.optional_import") as mock_import:
        mock_module = MagicMock()
        mock_import.return_value = mock_module

        mock_ds_cls = MagicMock()
        mock_module.TEST = mock_ds_cls

        with patch("modssc.data_loader.providers.torchaudio._official_or_single") as mock_oos:
            mock_train = MagicMock()
            mock_test = MagicMock()
            mock_oos.return_value = (mock_train, mock_test)

            with patch("modssc.data_loader.providers.torchaudio._extract_xy") as mock_ex:
                mock_ex.side_effect = [
                    (np.array(["t1"]), np.array([0])),
                    (np.array(["t2"]), np.array([1])),
                ]

                ds = provider.load_canonical(identity, raw_dir=tmp_path)

                assert ds.train.X[0] == "t1"
                assert ds.test.X[0] == "t2"
                assert ds.meta["provider"] == "torchaudio"


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


def test_load_canonical_single_split(tmp_path):
    provider = TorchaudioProvider()
    identity = DatasetIdentity(
        provider="torchaudio",
        canonical_uri="torchaudio:TEST",
        dataset_id="TEST",
        version=None,
        modality="audio",
        task="classification",
        resolved_kwargs={"dataset_class": "TEST"},
    )

    with patch("modssc.data_loader.providers.torchaudio.optional_import") as mock_import:
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        mock_module.TEST = MagicMock()

        with patch("modssc.data_loader.providers.torchaudio._official_or_single") as mock_oos:
            mock_train = MagicMock()
            mock_oos.return_value = (mock_train, None)

            with patch("modssc.data_loader.providers.torchaudio._extract_xy") as mock_ex:
                mock_ex.return_value = (np.array(["t1"]), np.array([0]))

                ds = provider.load_canonical(identity, raw_dir=tmp_path)

                assert ds.train.X[0] == "t1"
                assert ds.test is None
