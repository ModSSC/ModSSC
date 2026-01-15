from __future__ import annotations

import contextlib
import types

import numpy as np
import pytest

import modssc.data_loader.api as api
from modssc.data_loader.errors import DatasetNotCachedError, OptionalDependencyError
from modssc.data_loader.types import DatasetIdentity


def test_cache_dir_uses_env_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MODSSC_CACHE_DIR", str(tmp_path))
    assert api.cache_dir() == tmp_path.expanduser().resolve()


def test_load_dataset_triggers_download_path(tmp_path) -> None:
    ds = api.load_dataset("toy", cache_dir=tmp_path, download=True, force=False)
    assert ds.train is not None
    assert ds.train.X.shape[0] > 0


def test_load_dataset_offline_raises(tmp_path) -> None:
    with pytest.raises(DatasetNotCachedError):
        api.load_dataset("toy", cache_dir=tmp_path, download=False, force=False)


def test_download_dataset_cached_short_circuit(tmp_path) -> None:
    api.download_dataset("toy", cache_dir=tmp_path, force=True)

    ds2 = api.download_dataset("toy", cache_dir=tmp_path, force=False)
    assert ds2.train is not None


def test_download_all_skip_cached_and_filters(monkeypatch, tmp_path) -> None:
    api.download_dataset("toy", cache_dir=tmp_path, force=True)

    report = api.download_all_datasets(
        include=["toy"],
        cache_dir=tmp_path,
        force=False,
        skip_cached=True,
        ignore_missing_extras=True,
    )
    assert list(report.downloaded) == []
    assert list(report.skipped_already_cached) == ["toy"]

    specs = {
        "a": types.SimpleNamespace(modality="tabular"),
        "b": types.SimpleNamespace(modality="vision"),
        "c": types.SimpleNamespace(modality="tabular"),
    }
    monkeypatch.setattr(api, "DATASET_CATALOG", specs)

    selected = api._select_catalog_keys(include=None, exclude=["b"], modalities=["tabular"])
    assert selected == ["a", "c"]


def test_download_all_collects_optional_dependency_errors(monkeypatch, tmp_path) -> None:
    def boom(*args, **kwargs):
        raise OptionalDependencyError(extra="text", purpose="unit test")

    monkeypatch.setattr(api, "download_dataset", boom)

    report = api.download_all_datasets(
        include=["toy"],
        cache_dir=tmp_path,
        ignore_missing_extras=True,
    )
    assert list(report.skipped_missing_extras) == ["toy"]

    with pytest.raises(OptionalDependencyError):
        api.download_all_datasets(
            include=["toy"],
            cache_dir=tmp_path,
            ignore_missing_extras=False,
        )


def test_download_all_collects_failures(monkeypatch, tmp_path) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(api, "download_dataset", boom)

    report = api.download_all_datasets(
        include=["toy"],
        cache_dir=tmp_path,
        ignore_missing_extras=True,
    )
    assert "toy" in report.failed


def test_load_dataset_returns_cached_directly(tmp_path) -> None:
    api.download_dataset("toy", cache_dir=tmp_path, force=True)

    ds = api.load_dataset("toy", cache_dir=tmp_path, force=False, download=False)
    assert ds.train is not None


def test_download_all_populates_downloaded(tmp_path) -> None:
    report = api.download_all_datasets(
        include=["toy"],
        cache_dir=tmp_path,
        force=True,
        skip_cached=False,
    )
    assert list(report.downloaded) == ["toy"]
    assert list(report.skipped_already_cached) == []


def test_download_all_skip_cached_but_not_present(tmp_path) -> None:
    report = api.download_all_datasets(
        include=["toy"],
        cache_dir=tmp_path,
        skip_cached=True,
    )
    assert list(report.downloaded) == ["toy"]
    assert list(report.skipped_already_cached) == []


def test_provider_names() -> None:
    names = api.provider_names()
    assert "toy" in names
    assert "hf" in names


def test_provider_extra_mapping() -> None:
    assert api._provider_extra("toy") is None
    assert api._provider_extra("torchaudio") == "audio"
    assert api._provider_extra("unknown") is None


def test_split_stats_returns_none_for_missing_y() -> None:
    split = types.SimpleNamespace(y=None)
    assert api._split_stats(split) == (None, None)


def test_split_stats_handles_bad_array() -> None:
    class BadArray:
        def __array__(self, dtype=None):
            raise RuntimeError("no array")

    split = types.SimpleNamespace(y=BadArray())
    assert api._split_stats(split) == (None, None)


def test_split_stats_handles_scalar() -> None:
    split = types.SimpleNamespace(y=5)
    assert api._split_stats(split) == (None, None)


def test_split_stats_empty_integer_labels() -> None:
    split = types.SimpleNamespace(y=np.array([], dtype=np.int64))
    n_samples, n_classes = api._split_stats(split)
    assert n_samples == 0
    assert n_classes == 0


def test_split_stats_out_of_range_integer_labels() -> None:
    split = types.SimpleNamespace(y=np.array([-1, 2_000_001], dtype=np.int64))
    n_samples, n_classes = api._split_stats(split)
    assert n_samples == 2
    assert n_classes == 2


def test_split_stats_handles_unique_failure() -> None:
    class Unsortable:
        def __lt__(self, other):
            raise TypeError("no ordering")

    y = [Unsortable(), Unsortable()]
    split = types.SimpleNamespace(y=y)
    n_samples, n_classes = api._split_stats(split)
    assert n_samples == 2
    assert n_classes is None


def test_split_stats_large_sample_skips_unique() -> None:
    split = types.SimpleNamespace(y=np.zeros((1_000_001,), dtype=np.int8))
    n_samples, n_classes = api._split_stats(split)
    assert n_samples == 1_000_001
    assert n_classes is None


def test_download_and_store_cleanup_on_partial_state(monkeypatch, tmp_path) -> None:
    layout = api._layout(tmp_path)

    identity = DatasetIdentity(
        canonical_uri="toy://test",
        provider="toy",
        dataset_id="test",
        version="1",
        modality="tabular",
        task="classification",
        resolved_kwargs={
            "n_samples": 10,
            "n_features": 2,
            "n_classes": 2,
            "seed": 42,
            "test": False,
            "test_size": 0.2,
        },
    )
    fp = identity.fingerprint(schema_version=api.SCHEMA_VERSION)

    processed_dir = layout.processed_dir(fp)
    processed_dir.mkdir(parents=True)
    (processed_dir / "garbage").touch()

    monkeypatch.setattr(api, "_resolve_identity", lambda req: identity)

    api.download_dataset("toy://test", cache_dir=tmp_path, force=True)

    assert processed_dir.exists()
    assert not (processed_dir / "garbage").exists()


def _dummy_identity() -> DatasetIdentity:
    return DatasetIdentity(
        canonical_uri="toy://cached",
        provider="toy",
        dataset_id="cached",
        version="1",
        modality="tabular",
        task="classification",
        resolved_kwargs={},
    )


def test_download_and_store_short_circuits_before_lock(monkeypatch, tmp_path) -> None:
    layout = api._layout(tmp_path)
    identity = _dummy_identity()
    sentinel = object()

    calls = {"is_cached": 0, "cache_lock": 0}

    def fake_is_cached(*_args, **_kwargs):
        calls["is_cached"] += 1
        return True

    def fake_cache_lock(*_args, **_kwargs):
        calls["cache_lock"] += 1
        return contextlib.nullcontext()

    monkeypatch.setattr(api.cache, "is_cached", fake_is_cached)
    monkeypatch.setattr(api.cache, "cache_lock", fake_cache_lock)
    monkeypatch.setattr(api, "_load_processed", lambda *_args, **_kwargs: sentinel)

    assert api._download_and_store(layout, identity, force=False) is sentinel
    assert calls["is_cached"] == 1
    assert calls["cache_lock"] == 0


def test_download_and_store_rechecks_cache_after_lock(monkeypatch, tmp_path) -> None:
    layout = api._layout(tmp_path)
    identity = _dummy_identity()
    sentinel = object()

    calls = {"is_cached": 0, "cache_lock": 0, "load_processed": 0}

    def fake_is_cached(*_args, **_kwargs):
        calls["is_cached"] += 1
        return calls["is_cached"] >= 2

    def fake_cache_lock(*_args, **_kwargs):
        calls["cache_lock"] += 1
        return contextlib.nullcontext()

    def fake_load_processed(*_args, **_kwargs):
        calls["load_processed"] += 1
        return sentinel

    monkeypatch.setattr(api.cache, "is_cached", fake_is_cached)
    monkeypatch.setattr(api.cache, "cache_lock", fake_cache_lock)
    monkeypatch.setattr(api, "_load_processed", fake_load_processed)

    assert api._download_and_store(layout, identity, force=False) is sentinel
    assert calls["is_cached"] >= 2
    assert calls["cache_lock"] == 1
    assert calls["load_processed"] == 1


def test_dataset_info_uri():
    uri = "hf:foo/bar"
    info = api.dataset_info(uri)
    assert info.provider == "hf"
    assert info.key == uri
    assert info.required_extra is not None


def test_load_dataset_invalid_id_raises_unknown(tmp_path):
    from modssc.data_loader.errors import UnknownDatasetError

    with pytest.raises(UnknownDatasetError):
        api.load_dataset("invalid_id_obviously_not_a_uri", cache_dir=tmp_path)


def test_dataset_info_raises_unknown() -> None:
    from modssc.data_loader.errors import UnknownDatasetError

    with pytest.raises(UnknownDatasetError):
        api.dataset_info("invalid_dataset_id_and_not_a_uri")
