from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from modssc.data_loader import api
from modssc.data_loader.errors import DatasetNotCachedError, OptionalDependencyError


def test_download_and_load_offline(tmp_path) -> None:
    ds = api.download_dataset("toy", cache_dir=tmp_path, force=True)
    assert ds.train.X.shape[0] > 0
    ds2 = api.load_dataset("toy", cache_dir=tmp_path, download=False)
    assert ds2.train.X.shape == ds.train.X.shape


def test_offline_raises_if_not_cached(tmp_path) -> None:
    with pytest.raises(DatasetNotCachedError):
        api.load_dataset("toy", cache_dir=tmp_path, download=False)


def test_download_all_missing_extras_grouped(monkeypatch, tmp_path) -> None:
    def boom(*args, **kwargs):
        raise OptionalDependencyError(extra="hf", purpose="unit test")

    monkeypatch.setattr(api, "download_dataset", boom)

    report = api.download_all_datasets(
        cache_dir=tmp_path, include=["toy"], ignore_missing_extras=True
    )
    assert report.skipped_missing_extras == ["toy"]
    assert report.missing_extras["hf"] == ["toy"]

    with pytest.raises(OptionalDependencyError):
        api.download_all_datasets(cache_dir=tmp_path, include=["toy"], ignore_missing_extras=False)


def test_uri_download_with_stub_provider(monkeypatch, tmp_path) -> None:
    skl = types.ModuleType("sklearn")
    datasets = types.ModuleType("sklearn.datasets")

    def fetch_openml(**kwargs):
        return np.array([[1.0]]), np.array([0])

    datasets.fetch_openml = fetch_openml
    skl.datasets = datasets

    monkeypatch.setitem(sys.modules, "sklearn", skl)
    monkeypatch.setitem(sys.modules, "sklearn.datasets", datasets)

    ds = api.download_dataset("openml:61", cache_dir=tmp_path, force=True)
    assert ds.train.X.shape == (1, 1)
