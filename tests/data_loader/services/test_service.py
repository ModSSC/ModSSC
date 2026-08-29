from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

import modssc.data_loader.api as api
import modssc.data_loader.services.service as service
from modssc.data_loader.errors import DatasetNotCachedError, ManifestError
from modssc.data_loader.types import DatasetIdentity, LoadedDataset, Split


def test_api_module_aliases_internal_service() -> None:
    assert api is service
    assert api.load_dataset is service.load_dataset
    assert api.download_dataset is service.download_dataset
    assert api.dataset_info is service.dataset_info


def test_catalog_listing_and_curated_dataset_info() -> None:
    available = service.available_datasets()

    assert available == sorted(service.DATASET_CATALOG)
    assert available
    assert service.dataset_info(available[0]) is service.DATASET_CATALOG[available[0]]


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


def test_attach_resolved_identity_restores_modality_and_task_without_mutation() -> None:
    dataset = _dummy_dataset()
    identity = _dummy_identity()

    result = service._attach_resolved_identity(dataset, identity)

    assert result.meta["modality"] == "tabular"
    assert result.meta["task"] == "classification"
    assert dataset.meta == {}


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

    assert result.train is downloaded.train
    assert result.meta == {"modality": "tabular", "task": "classification"}
    assert downloaded.meta == {}
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
            (processed_dir / "layout.json").write_text("{}\n", encoding="utf-8")

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

    assert result.train is downloaded.train
    assert result.meta["dataset_fingerprint"] == identity.fingerprint(
        schema_version=service.SCHEMA_VERSION
    )
    assert len(result.meta["dataset_content_sha256"]) == 64
    assert calls == {
        "is_cached": 2,
        "load_processed_or_purge": 2,
        "load_canonical": 1,
        "save": 1,
    }


def test_load_processed_does_not_mutate_cache_when_content_manifest_is_missing(
    monkeypatch, tmp_path
) -> None:
    layout = service._layout(tmp_path)
    fingerprint = "cached-fingerprint"
    processed_dir = layout.processed_dir(fingerprint)
    processed_dir.mkdir(parents=True)
    (processed_dir / "layout.json").write_text("{}\n", encoding="utf-8")

    dataset = _dummy_dataset()

    class _Storage:
        def load(self, path):
            assert path == processed_dir
            return dataset

    monkeypatch.setattr(service, "FileStorage", lambda: _Storage())
    monkeypatch.setattr(
        service.cache,
        "read_cached_manifest",
        lambda *_args: SimpleNamespace(identity=_dummy_identity().as_dict()),
    )
    monkeypatch.setattr(
        service,
        "build_content_manifest",
        lambda *_args, **_kwargs: pytest.fail("ordinary cache reads must not build a sidecar"),
    )
    monkeypatch.setattr(
        service.cache,
        "atomic_write_text",
        lambda *_args: pytest.fail("ordinary cache reads must be read-only"),
    )
    monkeypatch.setattr(
        service,
        "_attach_content_evidence",
        lambda *_args, **_kwargs: pytest.fail("missing evidence must not be attached"),
    )

    result = service._load_processed(layout, fingerprint)

    assert result is dataset
    assert result.meta["dataset_fingerprint"] == fingerprint


def test_verify_dataset_content_rejects_uncached_dataset(monkeypatch, tmp_path) -> None:
    identity = _dummy_identity()
    monkeypatch.setattr(service, "_resolve_identity", lambda _request: identity)
    monkeypatch.setattr(service.cache, "is_cached", lambda *_args: False)

    with pytest.raises(DatasetNotCachedError, match="missing"):
        service.verify_dataset_content("missing", cache_dir=tmp_path)


def test_offline_load_propagates_corruption_without_purging(monkeypatch, tmp_path) -> None:
    identity = _dummy_identity()
    monkeypatch.setattr(service, "_resolve_identity", lambda _request: identity)
    monkeypatch.setattr(service.cache, "is_cached", lambda *_args: True)
    monkeypatch.setattr(
        service,
        "_load_processed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ManifestError("corrupt cache")),
    )
    monkeypatch.setattr(
        service,
        "_load_processed_or_purge",
        lambda *_args, **_kwargs: pytest.fail("offline load must not purge"),
    )

    with pytest.raises(ManifestError, match="corrupt cache"):
        service.load_dataset("cached", cache_dir=tmp_path, download=False)

    assert list(tmp_path.iterdir()) == []


def test_verify_dataset_content_rejects_missing_sidecar_without_backfill(
    monkeypatch, tmp_path
) -> None:
    identity = _dummy_identity()
    monkeypatch.setattr(service, "_resolve_identity", lambda _request: identity)
    monkeypatch.setattr(service.cache, "is_cached", lambda *_args: True)
    monkeypatch.setattr(
        service,
        "_load_processed_or_purge",
        lambda *_args, **_kwargs: pytest.fail("verification must not load or purge"),
    )
    monkeypatch.setattr(
        service.cache,
        "atomic_write_text",
        lambda *_args: pytest.fail("verification must not backfill"),
    )

    with pytest.raises(ManifestError, match="Invalid dataset content manifest"):
        service.verify_dataset_content("cached", cache_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_verify_dataset_content_uses_existing_manifest_without_backfill(
    monkeypatch, tmp_path
) -> None:
    identity = _dummy_identity()
    fingerprint = identity.fingerprint(schema_version=service.SCHEMA_VERSION)
    layout = service._layout(tmp_path)
    manifest_path = layout.content_manifest_path(fingerprint)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    expected = {"content_sha256": "b" * 64}

    monkeypatch.setattr(service, "_resolve_identity", lambda _request: identity)
    monkeypatch.setattr(service.cache, "is_cached", lambda *_args: True)
    monkeypatch.setattr(
        service,
        "_load_processed_or_purge",
        lambda *_args, **_kwargs: pytest.fail("existing manifest must not be backfilled"),
    )
    monkeypatch.setattr(
        service,
        "verify_content_manifest",
        lambda actual_layout, actual_fingerprint, *, identity, rehash: (
            expected
            if (
                actual_layout.root == layout.root
                and actual_fingerprint == fingerprint
                and identity == _dummy_identity().as_dict()
                and rehash is True
            )
            else pytest.fail("unexpected verification inputs")
        ),
    )

    assert service.verify_dataset_content("cached", cache_dir=tmp_path) is expected
