import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modssc.data_loader import cache
from modssc.data_loader.cache import (
    CacheLayout,
    atomic_write_text,
    cache_lock,
    default_cache_dir,
    dir_size_bytes,
    index_list,
    is_cached,
    read_cached_manifest,
    rebuild_index,
)
from modssc.data_loader.errors import ManifestError
from modssc.data_loader.manifest import Manifest, write_manifest


@pytest.fixture
def layout(tmp_path):
    return CacheLayout(root=tmp_path / "cache")


def test_is_cached(layout):
    fp = "test_fp"

    assert not is_cached(layout, fp)

    layout.processed_dir(fp).mkdir(parents=True)
    assert not is_cached(layout, fp)

    layout.manifest_path(fp).parent.mkdir(parents=True)
    layout.manifest_path(fp).touch()
    assert is_cached(layout, fp)


def test_rebuild_index_invalid_manifest(layout):
    layout.manifests_root.mkdir(parents=True)
    bad_manifest = layout.manifests_root / "bad.json"
    bad_manifest.write_text("not json")

    rebuild_index(layout)

    assert len(index_list(layout)) == 0


def test_rebuild_index_missing_processed(layout):
    layout.manifests_root.mkdir(parents=True)

    fp1 = "missing_proc_1"
    (layout.manifests_root / f"{fp1}.json").write_text("{}")

    fp2 = "missing_proc_2"
    (layout.manifests_root / f"{fp2}.json").write_text("{}")

    with patch("modssc.data_loader.cache.read_manifest") as mock_read:
        mock_read.return_value = MagicMock(spec=Manifest)
        rebuild_index(layout)

    assert len(index_list(layout)) == 0


def test_dir_size_bytes_oserror(tmp_path):
    mock_p1 = MagicMock(spec=Path)
    mock_p1.is_file.return_value = True
    mock_p1.stat.side_effect = OSError("Simulated error")

    mock_p2 = MagicMock(spec=Path)
    mock_p2.is_file.return_value = True
    mock_p2.stat.return_value.st_size = 100

    with patch("pathlib.Path.rglob", return_value=[mock_p1, mock_p2]):
        size = dir_size_bytes(tmp_path)
        assert size == 100


def test_dir_size_bytes_skips_non_files(tmp_path):
    mock_dir = MagicMock(spec=Path)
    mock_dir.is_file.return_value = False

    with patch("pathlib.Path.rglob", return_value=[mock_dir]):
        assert dir_size_bytes(tmp_path) == 0


def test_cache_lock_cleanup(layout):
    fp = "lock_test"

    with pytest.raises(RuntimeError), cache_lock(layout, fp):
        assert layout.lock_path(fp).exists()
        raise RuntimeError("Boom")

    assert not layout.lock_path(fp).exists()


def test_default_cache_dir_dev_repo():
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("modssc.data_loader.cache.Path.cwd") as mock_cwd,
    ):
        mock_path = MagicMock(spec=Path)
        mock_cwd.return_value = mock_path

        mock_parent = MagicMock(spec=Path)
        mock_path.parents = [mock_parent]

        mock_pyproject = MagicMock()
        mock_pyproject.exists.return_value = True

        mock_path.__truediv__.return_value = mock_pyproject

        mock_cache_dir = MagicMock()
        mock_path.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = (
            mock_cache_dir
        )

        mock_root = MagicMock(spec=Path)
        mock_cwd.return_value = mock_root
        mock_root.parents = []

        mock_pyproj = MagicMock()
        mock_pyproj.exists.return_value = True

        mock_root.__truediv__.side_effect = (
            lambda x: mock_pyproj if x == "pyproject.toml" else MagicMock()
        )

        res = default_cache_dir()

        assert res is not None


def test_default_cache_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom_cache"
    monkeypatch.setenv("MODSSC_CACHE_DIR", str(custom))
    assert default_cache_dir() == custom.resolve()


def test_atomic_write_text_and_dir_size(tmp_path: Path) -> None:
    p = tmp_path / "file.txt"
    atomic_write_text(p, "hello")
    assert p.read_text(encoding="utf-8") == "hello"
    assert dir_size_bytes(tmp_path) >= 5


def test_cache_lock_creates_and_removes_lock(tmp_path: Path) -> None:
    layout = CacheLayout(root=tmp_path)
    fp = "abc123"
    lock_path = layout.lock_path(fp)
    assert not lock_path.exists()
    with cache_lock(layout, fp):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_default_cache_dir_no_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("modssc.data_loader.cache.user_cache_dir") as mock_user_cache,
    ):
        mock_user_cache.return_value = "/tmp/mock_cache"
        path = default_cache_dir()
        assert path == Path("/tmp/mock_cache/datasets")


def test_read_cached_manifest_missing(tmp_path):
    layout = CacheLayout(root=tmp_path)

    layout.manifests_root.mkdir(parents=True)

    with pytest.raises(ManifestError, match="Missing manifest"):
        read_cached_manifest(layout, "non_existent_fingerprint")


def test_read_cached_manifest_success(tmp_path):
    layout = CacheLayout(root=tmp_path)
    fp = "fp_read"
    manifest = Manifest(
        schema_version=1,
        fingerprint=fp,
        created_at="now",
        identity={
            "canonical_uri": "toy:default",
            "provider": "toy",
            "dataset_id": "toy",
            "version": None,
            "modality": "tabular",
            "task": "classification",
        },
        dataset={},
        meta={},
        environment={},
    )
    write_manifest(layout.manifest_path(fp), manifest)
    out = read_cached_manifest(layout, fp)
    assert out.fingerprint == fp


def test_rebuild_index_with_processed(layout):
    fp = "fp_ok"
    manifest = Manifest(
        schema_version=1,
        fingerprint=fp,
        created_at="now",
        identity={
            "canonical_uri": "toy:default",
            "provider": "toy",
            "dataset_id": "toy",
            "version": None,
            "modality": "tabular",
            "task": "classification",
        },
        dataset={},
        meta={},
        environment={},
    )
    write_manifest(layout.manifest_path(fp), manifest)
    layout.processed_dir(fp).mkdir(parents=True)

    rebuild_index(layout)
    rows = index_list(layout)
    assert any(r["fingerprint"] == fp for r in rows)


def test_cache_lock_open_failure(layout):
    fp = "lock_fail"
    with (
        patch("modssc.data_loader.cache.os.open", side_effect=OSError("boom")),
        pytest.raises(OSError),
        cache_lock(layout, fp),
    ):
        pass
    assert not layout.lock_path(fp).exists()


def _fake_manifest(fp: str, uri: str, created_at: str) -> Manifest:
    return Manifest(
        schema_version=1,
        fingerprint=fp,
        created_at=created_at,
        identity={
            "canonical_uri": uri,
            "provider": "toy",
            "dataset_id": "toy",
            "version": "1",
            "modality": "tabular",
        },
        dataset={},
        meta={},
        environment={"python": "3.x"},
    )


def test_index_upsert_list_purge_gc(tmp_path: Path) -> None:
    layout = cache.CacheLayout(root=tmp_path)
    cache.ensure_layout(layout)

    fp1 = "a" * 64
    fp2 = "b" * 64
    uri = "toy:default"

    (layout.processed_dir(fp1)).mkdir(parents=True, exist_ok=True)
    (layout.processed_dir(fp2)).mkdir(parents=True, exist_ok=True)
    layout.manifest_path(fp1).write_text(
        _fake_manifest(fp1, uri, "2025-01-01T00:00:00+00:00").to_json()
    )
    layout.manifest_path(fp2).write_text(
        _fake_manifest(fp2, uri, "2026-01-01T00:00:00+00:00").to_json()
    )

    cache.index_upsert(
        layout, fingerprint=fp1, manifest=_fake_manifest(fp1, uri, "2025-01-01T00:00:00+00:00")
    )
    cache.index_upsert(
        layout, fingerprint=fp2, manifest=_fake_manifest(fp2, uri, "2026-01-01T00:00:00+00:00")
    )

    rows = cache.index_list(layout)
    assert len(rows) == 2

    removed = cache.gc_keep_latest(layout)
    assert fp1 in removed
    rows2 = cache.index_list(layout)
    assert len(rows2) == 1

    cache.purge_fingerprint(layout, fp2)
    rows3 = cache.index_list(layout)
    assert len(rows3) == 0
