from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir

from modssc.data_loader.errors import ManifestError
from modssc.data_loader.manifest import Manifest, read_manifest

CACHE_ENV = "MODSSC_CACHE_DIR"


def default_cache_dir() -> Path:
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser().resolve()

    # Heuristic: if running in a dev repo (pyproject.toml exists in parents),
    # default to a local "cache" folder at the repo root.
    current = Path.cwd()
    # Check current and parents for pyproject.toml
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent / "cache" / "datasets"

    return Path(user_cache_dir("modssc")) / "datasets"


@dataclass(frozen=True)
class CacheLayout:
    root: Path

    @property
    def raw_root(self) -> Path:
        return self.root / "raw"

    @property
    def processed_root(self) -> Path:
        return self.root / "processed"

    @property
    def manifests_root(self) -> Path:
        return self.root / "manifests"

    @property
    def locks_root(self) -> Path:
        return self.root / "locks"

    @property
    def index_path(self) -> Path:
        return self.root / "index.sqlite"

    def processed_dir(self, fingerprint: str) -> Path:
        return self.processed_root / fingerprint

    def manifest_path(self, fingerprint: str) -> Path:
        return self.manifests_root / f"{fingerprint}.json"

    def lock_path(self, fingerprint: str) -> Path:
        return self.locks_root / f"{fingerprint}.lock"

    def raw_dir(self, provider: str, dataset_id: str, version: str | None) -> Path:
        # Avoid overly deep trees, keep stable per dataset identity.
        v = version or "noversion"
        safe_id = dataset_id.replace("/", "_")
        return self.raw_root / provider / safe_id / v


def ensure_layout(layout: CacheLayout) -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.raw_root.mkdir(parents=True, exist_ok=True)
    layout.processed_root.mkdir(parents=True, exist_ok=True)
    layout.manifests_root.mkdir(parents=True, exist_ok=True)
    layout.locks_root.mkdir(parents=True, exist_ok=True)
    _ensure_index(layout.index_path)


@contextmanager
def cache_lock(layout: CacheLayout, fingerprint: str) -> Iterator[None]:
    """Simple file lock using O_EXCL creation."""
    ensure_layout(layout)
    lock_path = layout.lock_path(fingerprint)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    fd: int | None = None
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        yield
    finally:
        if fd is not None:
            os.close(fd)
        # best effort cleanup
        with contextlib.suppress(Exception):
            lock_path.unlink(missing_ok=True)


def is_cached(layout: CacheLayout, fingerprint: str) -> bool:
    has_dir = layout.processed_dir(fingerprint).is_dir()
    has_manifest = layout.manifest_path(fingerprint).is_file()
    return has_dir and has_manifest


def read_cached_manifest(layout: CacheLayout, fingerprint: str) -> Manifest:
    path = layout.manifest_path(fingerprint)
    if not path.is_file():
        raise ManifestError(f"Missing manifest for fingerprint: {fingerprint}")
    return read_manifest(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            with contextlib.suppress(OSError):
                total += p.stat().st_size
    return total


# ----------------------------
# Index (sqlite) helpers
# ----------------------------


def _ensure_index(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS variants (
                fingerprint TEXT PRIMARY KEY,
                canonical_uri TEXT NOT NULL,
                provider TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                version TEXT,
                modality TEXT,
                created_at TEXT,
                processed_dir TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                size_bytes INTEGER
            )
            """
        )
        con.commit()
    finally:
        con.close()


def index_upsert(layout: CacheLayout, *, fingerprint: str, manifest: Manifest) -> None:
    con = sqlite3.connect(layout.index_path)
    try:
        processed = str(layout.processed_dir(fingerprint))
        mpath = str(layout.manifest_path(fingerprint))
        size = dir_size_bytes(Path(processed))
        ident = manifest.identity
        con.execute(
            """
            INSERT INTO variants (
                fingerprint, canonical_uri, provider, dataset_id, version, modality, created_at,
                processed_dir, manifest_path, size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET
                canonical_uri=excluded.canonical_uri,
                provider=excluded.provider,
                dataset_id=excluded.dataset_id,
                version=excluded.version,
                modality=excluded.modality,
                created_at=excluded.created_at,
                processed_dir=excluded.processed_dir,
                manifest_path=excluded.manifest_path,
                size_bytes=excluded.size_bytes
            """,
            (
                fingerprint,
                str(ident.get("canonical_uri")),
                str(ident.get("provider")),
                str(ident.get("dataset_id")),
                ident.get("version"),
                ident.get("modality"),
                manifest.created_at,
                processed,
                mpath,
                int(size),
            ),
        )
        con.commit()
    finally:
        con.close()


def index_list(layout: CacheLayout) -> list[dict[str, Any]]:
    con = sqlite3.connect(layout.index_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute("SELECT * FROM variants ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def index_find_by_dataset(layout: CacheLayout, canonical_uri: str) -> list[dict[str, Any]]:
    con = sqlite3.connect(layout.index_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT * FROM variants WHERE canonical_uri = ? ORDER BY created_at DESC",
            (canonical_uri,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def index_delete(layout: CacheLayout, fingerprints: Iterable[str]) -> None:
    con = sqlite3.connect(layout.index_path)
    try:
        con.executemany(
            "DELETE FROM variants WHERE fingerprint = ?", [(fp,) for fp in fingerprints]
        )
        con.commit()
    finally:
        con.close()


def rebuild_index(layout: CacheLayout) -> None:
    _ensure_index(layout.index_path)
    con = sqlite3.connect(layout.index_path)
    try:
        con.execute("DELETE FROM variants")
        con.commit()
    finally:
        con.close()

    for mf in layout.manifests_root.glob("*.json"):
        try:
            manifest = read_manifest(mf)
        except Exception:
            continue
        fp = mf.stem
        processed = layout.processed_dir(fp)
        if not processed.is_dir():
            continue
        index_upsert(layout, fingerprint=fp, manifest=manifest)


# ----------------------------
# Cache maintenance
# ----------------------------


def purge_fingerprint(layout: CacheLayout, fingerprint: str) -> None:
    shutil.rmtree(layout.processed_dir(fingerprint), ignore_errors=True)
    with contextlib.suppress(Exception):
        layout.manifest_path(fingerprint).unlink(missing_ok=True)
    index_delete(layout, [fingerprint])


def purge_dataset(layout: CacheLayout, dataset_id: str) -> list[str]:
    """Purge all variants matching a canonical URI or a curated key stored in canonical_uri."""
    matches = index_find_by_dataset(layout, dataset_id)
    fps = [m["fingerprint"] for m in matches]
    for fp in fps:
        purge_fingerprint(layout, fp)
    return fps


def gc_keep_latest(layout: CacheLayout) -> list[str]:
    """Keep only the latest variant for each canonical_uri."""
    con = sqlite3.connect(layout.index_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT canonical_uri, fingerprint, created_at
            FROM variants
            ORDER BY canonical_uri, created_at DESC
            """
        ).fetchall()
    finally:
        con.close()

    latest: dict[str, str] = {}
    to_delete: list[str] = []
    for r in rows:
        uri = str(r["canonical_uri"])
        fp = str(r["fingerprint"])
        if uri not in latest:
            latest[uri] = fp
        else:
            to_delete.append(fp)

    for fp in to_delete:
        purge_fingerprint(layout, fp)

    return to_delete
