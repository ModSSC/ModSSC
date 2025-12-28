from __future__ import annotations

from pathlib import Path

import pytest

from modssc.data_loader import api
from modssc.data_loader.cache import CacheLayout, ensure_layout, index_list
from modssc.data_loader.errors import (
    DatasetNotCachedError,
    OptionalDependencyError,
    UnknownDatasetError,
)


def test_available_datasets_and_dataset_info_key() -> None:
    keys = api.available_datasets()
    assert "toy" in keys

    spec = api.dataset_info("toy")
    assert spec.key == "toy"
    assert spec.provider == "toy"


def test_dataset_info_uri_and_unknown() -> None:
    spec = api.dataset_info("toy:default")
    assert spec.provider == "toy"
    assert spec.uri == "toy:default"

    with pytest.raises(UnknownDatasetError):
        api.dataset_info("not_a_dataset")

    with pytest.raises(UnknownDatasetError):
        api.load_dataset("not_a_dataset", download=False)


def test_load_dataset_requires_cache_or_download(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotCachedError):
        api.load_dataset("toy", cache_dir=tmp_path, download=False)

    ds = api.load_dataset("toy", cache_dir=tmp_path, download=True, force=True)
    assert ds.train.X is not None

    ds2 = api.load_dataset("toy", cache_dir=tmp_path, download=False)
    assert ds2.train.X.shape == ds.train.X.shape


def test_download_all_filters_and_skip_cached(tmp_path: Path) -> None:
    api.download_dataset("toy", cache_dir=tmp_path, force=True)

    report = api.download_all_datasets(
        cache_dir=tmp_path,
        include={"toy"},
        skip_cached=True,
        ignore_missing_extras=True,
    )
    assert report.skipped_already_cached == ["toy"]

    report2 = api.download_all_datasets(
        cache_dir=tmp_path,
        include={"toy"},
        exclude={"toy"},
        ignore_missing_extras=True,
    )
    assert report2.downloaded == []


def test_download_all_missing_extras_grouping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_download_dataset = api.download_dataset

    def fake_download_dataset(dataset_id: str, *args, **kwargs):
        if dataset_id == "cifar10":
            raise OptionalDependencyError(extra="vision", purpose="vision")
        return real_download_dataset("toy", cache_dir=tmp_path, force=True)

    monkeypatch.setattr(api, "download_dataset", fake_download_dataset)

    report = api.download_all_datasets(
        cache_dir=tmp_path,
        include={"toy", "cifar10"},
        ignore_missing_extras=True,
    )
    assert "cifar10" in report.skipped_missing_extras
    assert report.missing_extras.get("vision") == ["cifar10"]


def test_cache_records_view(tmp_path: Path) -> None:
    api.download_dataset("toy", cache_dir=tmp_path, force=True)
    layout = CacheLayout(root=tmp_path)
    ensure_layout(layout)
    rows = index_list(layout)
    assert rows
    assert any(r["canonical_uri"] == "toy:default" for r in rows)
