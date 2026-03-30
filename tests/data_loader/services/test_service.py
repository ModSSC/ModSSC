from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np

import modssc.data_loader.api as api
import modssc.data_loader.services.service as service
from modssc.data_loader.types import DatasetIdentity, LoadedDataset, Split


def test_api_module_aliases_internal_service() -> None:
    assert api is service
    assert api.load_dataset is service.load_dataset
    assert api.download_dataset is service.download_dataset
    assert api.dataset_info is service.dataset_info


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


def _dummy_dataset() -> LoadedDataset:
    return LoadedDataset(
        train=Split(X=np.zeros((2, 2), dtype=np.float32), y=np.array([0, 1], dtype=np.int64)),
        test=None,
        meta={},
    )


def test_download_dataset_redownloads_when_cached_load_returns_none(monkeypatch, tmp_path) -> None:
    identity = _dummy_identity()
    downloaded = _dummy_dataset()
    calls = {"download": 0}

    monkeypatch.setattr(service, "_resolve_identity", lambda _req: identity)
    monkeypatch.setattr(service.cache, "is_cached", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(service, "_load_processed_or_purge", lambda *_args, **_kwargs: None)

    def fake_download_and_store(layout, resolved_identity, *, force):
        assert layout.root == tmp_path.expanduser().resolve()
        assert resolved_identity is identity
        assert force is False
        calls["download"] += 1
        return downloaded

    monkeypatch.setattr(service, "_download_and_store", fake_download_and_store)

    result = service.download_dataset("toy://cached", cache_dir=tmp_path, force=False)

    assert result is downloaded
    assert calls["download"] == 1


def test_download_and_store_continues_when_cached_load_returns_none(monkeypatch, tmp_path) -> None:
    layout = service._layout(tmp_path)
    identity = _dummy_identity()
    downloaded = _dummy_dataset()
    calls = {"is_cached": 0, "load_processed_or_purge": 0, "load_canonical": 0, "save": 0}

    def fake_is_cached(*_args, **_kwargs):
        calls["is_cached"] += 1
        return True

    def fake_load_processed_or_purge(*_args, **_kwargs):
        calls["load_processed_or_purge"] += 1
        return None

    def fake_load_canonical(resolved_identity, *, raw_dir):
        assert resolved_identity is identity
        assert raw_dir.exists()
        calls["load_canonical"] += 1
        return downloaded

    class _Storage:
        def save(self, processed_dir, dataset):
            assert dataset is downloaded
            calls["save"] += 1
            processed_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(service.cache, "is_cached", fake_is_cached)
    monkeypatch.setattr(
        service.cache, "cache_lock", lambda *_args, **_kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(service, "_load_processed_or_purge", fake_load_processed_or_purge)
    monkeypatch.setattr(
        service,
        "create_provider",
        lambda _provider: SimpleNamespace(load_canonical=fake_load_canonical),
    )
    monkeypatch.setattr(service, "FileStorage", lambda: _Storage())
    monkeypatch.setattr(service, "build_manifest", lambda **_kwargs: {"manifest": True})
    monkeypatch.setattr(service, "write_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service.cache, "index_upsert", lambda *_args, **_kwargs: None)

    result = service._download_and_store(layout, identity, force=False)

    assert result is downloaded
    assert result.meta["dataset_fingerprint"] == identity.fingerprint(
        schema_version=service.SCHEMA_VERSION
    )
    assert calls == {
        "is_cached": 2,
        "load_processed_or_purge": 2,
        "load_canonical": 1,
        "save": 1,
    }
