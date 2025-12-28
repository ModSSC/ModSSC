from unittest.mock import MagicMock, patch

from modssc.data_loader.providers.tfds import TFDSProvider
from modssc.data_loader.types import DatasetIdentity
from modssc.data_loader.uri import ParsedURI


def test_resolve_no_version():
    provider = TFDSProvider()
    parsed = ParsedURI(provider="tfds", reference="mnist")
    identity = provider.resolve(parsed, options={})

    assert identity.dataset_id == "mnist"
    assert identity.version is None
    assert identity.canonical_uri == "tfds:mnist"
    assert identity.resolved_kwargs["name"] == "mnist"
    assert identity.resolved_kwargs["version"] is None


def test_load_canonical_test_split_fails(tmp_path):
    provider = TFDSProvider()
    identity = DatasetIdentity(
        provider="tfds",
        canonical_uri="tfds:mnist",
        dataset_id="mnist",
        version=None,
        modality="image",
        task="classification",
        required_extra="tfds",
        resolved_kwargs={"name": "mnist", "version": None, "as_supervised": True},
    )

    mock_tfds = MagicMock()

    mock_train_ds = MagicMock()

    mock_train_ds.as_numpy.return_value = [(1, 0), (2, 1)]

    mock_train_ds.__iter__.return_value = [(1, 0), (2, 1)]

    def side_effect(name, split, **kwargs):
        if split == "train":
            return mock_train_ds
        if split == "test":
            raise ValueError("No test split")
        return None

    mock_tfds.load.side_effect = side_effect

    with patch("modssc.data_loader.providers.tfds.optional_import", return_value=mock_tfds):
        loaded = provider.load_canonical(identity, raw_dir=tmp_path)

        assert loaded.train is not None
        assert len(loaded.train.X) == 2
        assert loaded.test is None


def test_resolve_with_version():
    provider = TFDSProvider()
    parsed = ParsedURI(provider="tfds", reference="mnist/3.0.0")
    identity = provider.resolve(parsed, options={})

    assert identity.dataset_id == "mnist"
    assert identity.version == "3.0.0"
    assert identity.canonical_uri == "tfds:mnist/3.0.0"
    assert identity.resolved_kwargs["name"] == "mnist"
    assert identity.resolved_kwargs["version"] == "3.0.0"


def test_load_canonical_success(tmp_path):
    provider = TFDSProvider()
    identity = DatasetIdentity(
        provider="tfds",
        canonical_uri="tfds:mnist",
        dataset_id="mnist",
        version=None,
        modality="image",
        task="classification",
        required_extra="tfds",
        resolved_kwargs={"name": "mnist", "version": None, "as_supervised": True},
    )

    mock_tfds = MagicMock()

    mock_train_ds = MagicMock()
    mock_train_ds.as_numpy.return_value = [(1, 0), (2, 1)]
    mock_train_ds.__iter__.return_value = [(1, 0), (2, 1)]

    mock_test_ds = MagicMock()
    mock_test_ds.as_numpy.return_value = [(3, 0)]
    mock_test_ds.__iter__.return_value = [(3, 0)]

    def side_effect(name, split, **kwargs):
        if split == "train":
            return mock_train_ds
        if split == "test":
            return mock_test_ds
        return None

    mock_tfds.load.side_effect = side_effect

    with patch("modssc.data_loader.providers.tfds.optional_import", return_value=mock_tfds):
        loaded = provider.load_canonical(identity, raw_dir=tmp_path)

        assert loaded.train is not None
        assert len(loaded.train.X) == 2
        assert loaded.test is not None
        assert len(loaded.test.X) == 1
