from unittest.mock import MagicMock, patch

import modssc.data_loader.api as api
from modssc.data_loader.types import DatasetIdentity, LoadedDataset


def test_download_dataset_injects_meta_if_none(tmp_path):
    with patch("modssc.data_loader.api.create_provider") as mock_create_provider:
        mock_provider = MagicMock()

        mock_provider.resolve.return_value = DatasetIdentity(
            dataset_id="toy",
            provider="toy",
            version="1.0",
            modality="tabular",
            task="classification",
            canonical_uri="toy://toy",
        )

        mock_ds = LoadedDataset(train=MagicMock(), test=MagicMock(), meta=None)
        mock_provider.load_canonical.return_value = mock_ds
        mock_create_provider.return_value = mock_provider

        with (
            patch("modssc.data_loader.api.build_manifest"),
            patch("modssc.data_loader.api.write_manifest"),
            patch("modssc.data_loader.api.cache.index_upsert"),
            patch("modssc.data_loader.api.FileStorage"),
        ):
            ds = api.download_dataset("toy", cache_dir=tmp_path, force=True)
            assert ds.meta is not None
            assert "dataset_fingerprint" in ds.meta


def test_load_processed_injects_meta_if_none(tmp_path):
    with patch("modssc.data_loader.api.FileStorage") as mock_storage_cls:
        mock_storage = MagicMock()

        mock_ds = LoadedDataset(train=MagicMock(), test=MagicMock(), meta=None)
        mock_storage.load.return_value = mock_ds
        mock_storage_cls.return_value = mock_storage

        layout = MagicMock()
        ds = api._load_processed(layout, fingerprint="test_fp")
        assert ds.meta is not None
        assert ds.meta["dataset_fingerprint"] == "test_fp"
