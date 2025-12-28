from __future__ import annotations

from pathlib import Path

from modssc.data_loader import cache
from modssc.data_loader.manifest import Manifest


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
