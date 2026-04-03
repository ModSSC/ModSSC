from unittest.mock import MagicMock, patch

import numpy as np

import modssc.data_loader.api as api
from modssc.data_loader.types import DatasetIdentity, LoadedDataset, Split


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


def test_load_processed_rebases_cached_torchaudio_paths(tmp_path):
    with (
        patch("modssc.data_loader.api.FileStorage") as mock_storage_cls,
        patch("modssc.data_loader.api.cache.read_cached_manifest") as mock_read_manifest,
    ):
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage

        cached_path = (
            "/Users/melvin/Desktop/ModSSC Project/ModSSC/modssc_cache/datasets/raw/"
            "torchaudio/YESNO/noversion/source/waves_yesno/0_0_0_0_1_1_1_1.wav"
        )
        mock_storage.load.return_value = LoadedDataset(
            train=Split(
                X=np.asarray([cached_path], dtype=object),
                y=np.asarray(["yes"], dtype=object),
            ),
            test=None,
            meta={"provider": "torchaudio", "representation": "paths_or_waveforms"},
        )
        mock_read_manifest.return_value = MagicMock(
            identity={"provider": "torchaudio", "dataset_id": "YESNO", "version": None}
        )

        layout = api._layout(tmp_path)
        ds = api._load_processed(layout, fingerprint="test_fp")

        expected = str(
            layout.raw_dir("torchaudio", "YESNO", None)
            / "source"
            / "waves_yesno"
            / "0_0_0_0_1_1_1_1.wav"
        )
        assert ds.train.X.tolist() == [expected]
        assert ds.meta["dataset_fingerprint"] == "test_fp"
