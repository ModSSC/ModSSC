from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

import modssc.data_loader.api as api
import modssc.data_loader.services.service as service
from modssc.data_loader.errors import DatasetNotCachedError
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


def test_verify_dataset_content_rejects_cache_purged_during_backfill(monkeypatch, tmp_path) -> None:
    identity = _dummy_identity()
    load_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(service, "_resolve_identity", lambda _request: identity)
    monkeypatch.setattr(service.cache, "is_cached", lambda *_args: True)

    def fake_load(layout, fingerprint, *, dataset_id):
        assert layout.root == tmp_path.expanduser().resolve()
        assert fingerprint == identity.fingerprint(schema_version=service.SCHEMA_VERSION)
        load_calls.append((fingerprint, dataset_id))
        return None

    monkeypatch.setattr(service, "_load_processed_or_purge", fake_load)

    with pytest.raises(DatasetNotCachedError, match="cached"):
        service.verify_dataset_content("cached", cache_dir=tmp_path)

    assert load_calls == [(identity.fingerprint(schema_version=service.SCHEMA_VERSION), "cached")]


def test_verify_dataset_content_verifies_after_successful_backfill(monkeypatch, tmp_path) -> None:
    identity = _dummy_identity()
    dataset = _dummy_dataset()
    expected = {"content_sha256": "a" * 64}
    fingerprint = identity.fingerprint(schema_version=service.SCHEMA_VERSION)
    content_manifest = {"schema_version": 1}
    writes: list[tuple[object, str]] = []

    monkeypatch.setattr(service, "_resolve_identity", lambda _request: identity)
    monkeypatch.setattr(service.cache, "is_cached", lambda *_args: True)
    monkeypatch.setattr(
        service,
        "_load_processed_or_purge",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        service,
        "build_content_manifest",
        lambda actual_layout, actual_fingerprint, actual_dataset, *, identity: (
            content_manifest
            if (
                actual_layout.root == tmp_path.expanduser().resolve()
                and actual_fingerprint == fingerprint
                and actual_dataset is dataset
                and identity == _dummy_identity().as_dict()
            )
            else pytest.fail("unexpected content-manifest inputs")
        ),
    )
    monkeypatch.setattr(
        service,
        "content_manifest_json",
        lambda value: (
            "serialized\n" if value is content_manifest else pytest.fail("wrong manifest")
        ),
    )
    monkeypatch.setattr(
        service.cache,
        "atomic_write_text",
        lambda path, text: writes.append((path, text)),
    )

    def fake_verify(layout, actual_fingerprint, *, identity, rehash):
        assert layout.root == tmp_path.expanduser().resolve()
        assert actual_fingerprint == fingerprint
        assert identity == _dummy_identity().as_dict()
        assert rehash is False
        return expected

    monkeypatch.setattr(service, "verify_content_manifest", fake_verify)

    result = service.verify_dataset_content("cached", cache_dir=tmp_path, rehash=False)

    assert result is expected
    assert writes == [
        (
            service._layout(tmp_path).content_manifest_path(fingerprint),
            "serialized\n",
        )
    ]
