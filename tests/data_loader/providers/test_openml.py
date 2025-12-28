from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modssc.data_loader.providers.openml import (
    OpenMLProvider,
    _apply_class_filter,
    _limit_samples,
    _normalize_filter,
    _parse_openml_ref,
    _to_numpy,
)
from modssc.data_loader.types import DatasetIdentity
from modssc.data_loader.uri import ParsedURI


def test_parse_openml_ref_digit():
    assert _parse_openml_ref("42") == {"data_id": 42}
    assert _parse_openml_ref(" 42 ") == {"data_id": 42}


def test_parse_openml_ref_kv_pairs():
    ref = "name=adult, version=2, foo=bar"
    parsed = _parse_openml_ref(ref)
    assert parsed["name"] == "adult"
    assert parsed["version"] == 2
    assert parsed["foo"] == "bar"


def test_parse_openml_ref_kv_with_data_id():
    ref = "data_id=100, version=active"
    parsed = _parse_openml_ref(ref)
    assert parsed["data_id"] == 100
    assert parsed["version"] == "active"


def test_parse_openml_ref_malformed():
    ref = "name=adult, malformed_part, version=1"
    parsed = _parse_openml_ref(ref)
    assert parsed["name"] == "adult"
    assert parsed["version"] == 1
    assert "malformed_part" not in parsed


def test_normalize_filter_variants():
    assert _normalize_filter(None) is None
    assert set(_normalize_filter({1, 2})) == {1, 2}
    assert _normalize_filter("x") == ["x"]


def test_apply_class_filter_and_limit_samples():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])
    X_f, y_f = _apply_class_filter(X, y, class_filter=[0])
    assert X_f.tolist() == [[1], [3]]
    assert y_f.tolist() == [0, 0]

    X_empty, y_empty = _limit_samples(X, y, max_samples=0, seed=None)
    assert X_empty.size == 0
    assert y_empty.size == 0

    X_take_no_seed, y_take_no_seed = _limit_samples(X, y, max_samples=2, seed=None)
    assert X_take_no_seed.shape == (2, 1)
    assert y_take_no_seed.shape == (2,)

    X_take, y_take = _limit_samples(X, y, max_samples=2, seed=123)
    assert X_take.shape == (2, 1)
    assert y_take.shape == (2,)


def test_resolve_by_id():
    provider = OpenMLProvider()
    uri = ParsedURI(provider="openml", reference="42")
    identity = provider.resolve(uri, options={})

    assert identity.canonical_uri == "openml:42"
    assert identity.dataset_id == "42"
    assert identity.resolved_kwargs["data_id"] == 42


def test_resolve_by_name():
    provider = OpenMLProvider()
    uri = ParsedURI(provider="openml", reference="name=iris")
    identity = provider.resolve(uri, options={})

    assert identity.canonical_uri == "openml:name=iris"
    assert identity.dataset_id == "iris"
    assert identity.resolved_kwargs["name"] == "iris"


def test_resolve_fallback_name():
    provider = OpenMLProvider()

    uri = ParsedURI(provider="openml", reference="my_dataset")
    identity = provider.resolve(uri, options={})

    assert identity.canonical_uri == "openml:name=my_dataset"
    assert identity.dataset_id == "my_dataset"


def test_resolve_options_override():
    provider = OpenMLProvider()
    uri = ParsedURI(provider="openml", reference="name=iris")

    identity = provider.resolve(uri, options={"version": 5, "as_frame": True})

    assert identity.resolved_kwargs["version"] == 5
    assert identity.resolved_kwargs["as_frame"] is True


@pytest.fixture
def mock_sklearn():
    with patch("modssc.data_loader.providers.openml.optional_import") as mock_import:
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        yield mock_module


def test_load_canonical_by_id(mock_sklearn, tmp_path):
    provider = OpenMLProvider()
    identity = DatasetIdentity(
        provider="openml",
        canonical_uri="openml:42",
        dataset_id="42",
        version=None,
        modality="tabular",
        task="classification",
        resolved_kwargs={"data_id": 42, "as_frame": True},
    )

    mock_sklearn.fetch_openml.return_value = (np.array([[1, 2]]), np.array([0, 1]))

    ds = provider.load_canonical(identity, raw_dir=tmp_path)

    mock_sklearn.fetch_openml.assert_called_once()
    call_kwargs = mock_sklearn.fetch_openml.call_args[1]
    assert call_kwargs["data_id"] == 42
    assert call_kwargs["as_frame"] is True
    assert call_kwargs["return_X_y"] is True
    assert call_kwargs["data_home"] == str(tmp_path)

    assert ds.train.X.shape == (1, 2)
    assert ds.train.y.shape == (2,)


def test_load_canonical_by_name_and_version(mock_sklearn, tmp_path):
    provider = OpenMLProvider()
    identity = DatasetIdentity(
        provider="openml",
        canonical_uri="openml:name=iris",
        dataset_id="iris",
        version="1",
        modality="tabular",
        task="classification",
        resolved_kwargs={"name": "iris", "version": 1},
    )

    mock_sklearn.fetch_openml.return_value = (np.array([[1]]), np.array([0]))

    provider.load_canonical(identity, raw_dir=tmp_path)

    call_kwargs = mock_sklearn.fetch_openml.call_args[1]
    assert call_kwargs["name"] == "iris"
    assert call_kwargs["version"] == 1
    assert "data_id" not in call_kwargs


def test_to_numpy_ndarray():
    arr = np.array([1, 2])
    assert _to_numpy(arr) is arr


def test_to_numpy_pandas_like():
    mock_df = MagicMock()
    mock_df.to_numpy.return_value = np.array([3, 4])
    res = _to_numpy(mock_df)
    assert np.array_equal(res, np.array([3, 4]))


def test_to_numpy_torch_like():
    mock_tensor = MagicMock()
    mock_tensor.numpy.return_value = np.array([5, 6])

    del mock_tensor.to_numpy
    res = _to_numpy(mock_tensor)
    assert np.array_equal(res, np.array([5, 6]))


def test_to_numpy_list():
    lst = [7, 8]
    res = _to_numpy(lst)
    assert isinstance(res, np.ndarray)
    assert np.array_equal(res, np.array([7, 8]))
