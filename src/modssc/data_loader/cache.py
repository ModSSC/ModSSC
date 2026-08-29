from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir

from modssc.data_loader.errors import ManifestError
from modssc.data_loader.manifest import Manifest, read_manifest
from modssc.runtime.paths import default_local_cache_subdir
from modssc.utils.io import atomic_write_text as _atomic_write_text

CACHE_ENV = "MODSSC_CACHE_DIR"
CACHE_ROOT_ENV = "MODSSC_CACHE_ROOT"


def default_cache_dir() -> Path:
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser().resolve()

    root_override = os.environ.get(CACHE_ROOT_ENV)
    if root_override:
        return Path(root_override).expanduser().resolve() / "datasets"

    local = default_local_cache_subdir("datasets")
    if local is not None:
        return local

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

    @property
    def index_lock_path(self) -> Path:
        return self.locks_root / "index.lock"

    def processed_dir(self, fingerprint: str) -> Path:
        return self.processed_root / fingerprint

    def manifest_path(self, fingerprint: str) -> Path:
        return self.manifests_root / f"{fingerprint}.json"

    def content_manifest_path(self, fingerprint: str) -> Path:
        return self.manifests_root / f"{fingerprint}.content.json"

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
    """Serialize writers for one dataset fingerprint across processes.

    The lock file is intentionally persistent.  Ownership is carried by the
    operating-system lock on its descriptor, so a killed process releases the
    lock without leaving a stale ``O_EXCL`` sentinel behind.  Keeping the path
    also avoids the unlink/recreate race where two writers could otherwise
    lock different inodes for the same fingerprint.
    """

    ensure_layout(layout)
    lock_path = layout.lock_path(fingerprint)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor = _open_lock_file(lock_path)
    locked = False
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        os.ftruncate(descriptor, 0)
        os.write(descriptor, str(os.getpid()).encode("utf-8"))
        yield
    finally:
        if locked:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def index_lock(layout: CacheLayout) -> Iterator[None]:
    """Serialize mutations of the replaceable cache index."""

    layout.locks_root.mkdir(parents=True, exist_ok=True)
    descriptor = _open_lock_file(layout.index_lock_path)
    locked = False
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _open_lock_file(path: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(path), flags, 0o600)
    lock_stat = os.fstat(descriptor)
    if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
        os.close(descriptor)
        raise ManifestError(f"Cache lock must be a single-link regular file: {path}")
    return descriptor


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
    _atomic_write_text(path, text)


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
    with index_lock(layout):
        _ensure_index(layout.index_path)
        con = sqlite3.connect(layout.index_path)
        try:
            _index_upsert_connection(con, layout, fingerprint=fingerprint, manifest=manifest)
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
    with index_lock(layout):
        _ensure_index(layout.index_path)
        con = sqlite3.connect(layout.index_path)
        try:
            con.executemany(
                "DELETE FROM variants WHERE fingerprint = ?", [(fp,) for fp in fingerprints]
            )
            con.commit()
        finally:
            con.close()


def _index_upsert_connection(
    con: sqlite3.Connection,
    layout: CacheLayout,
    *,
    fingerprint: str,
    manifest: Manifest,
) -> None:
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


_MAIN_MANIFEST_NAME = re.compile(r"^(?P<fingerprint>[0-9a-f]{64})\.json$")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - Windows has no directory fsync
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rebuild_index_atomic(layout: CacheLayout, *, strict: bool = True) -> str:
    """Build a complete index beside the live one and atomically replace it.

    Only canonical main-manifest names are considered.  With ``strict=True`` an
    invalid published entry aborts the rebuild and leaves the previous index
    untouched.  The returned digest attests the new SQLite file bytes.
    """

    with index_lock(layout):
        return _rebuild_index_atomic_locked(layout, strict=strict)


def _rebuild_index_atomic_locked(layout: CacheLayout, *, strict: bool) -> str:
    """Rebuild the index while the caller holds ``index_lock(layout)``."""

    layout.root.mkdir(parents=True, exist_ok=True)
    layout.locks_root.mkdir(parents=True, exist_ok=True)
    if layout.index_path.is_symlink():
        raise ManifestError(f"Cache index must not be a symlink: {layout.index_path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".index.sqlite.", suffix=".tmp", dir=layout.root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _ensure_index(temporary)
        con = sqlite3.connect(temporary)
        try:
            con.execute("BEGIN IMMEDIATE")
            manifests = sorted(layout.manifests_root.glob("*.json"))
            for path in manifests:
                match = _MAIN_MANIFEST_NAME.fullmatch(path.name)
                if match is None:
                    continue
                fingerprint = match.group("fingerprint")
                try:
                    manifest = read_manifest(path)
                    if manifest.fingerprint != fingerprint:
                        raise ManifestError(f"Manifest fingerprint differs from filename: {path}")
                    if not layout.processed_dir(fingerprint).is_dir():
                        raise ManifestError(f"Published manifest has no processed cache: {path}")
                    _index_upsert_connection(
                        con,
                        layout,
                        fingerprint=fingerprint,
                        manifest=manifest,
                    )
                except Exception as exc:
                    if strict:
                        if isinstance(exc, ManifestError):
                            raise
                        raise ManifestError(f"Invalid published cache manifest: {path}") from exc
            con.commit()
            result = con.execute("PRAGMA integrity_check").fetchone()
            if result != ("ok",):
                raise ManifestError(f"Cache index integrity check failed: {result!r}")
        finally:
            con.close()

        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, layout.index_path)
        _fsync_directory(layout.root)
        digest = hashlib.sha256()
        with layout.index_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


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
    with contextlib.suppress(Exception):
        layout.content_manifest_path(fingerprint).unlink(missing_ok=True)
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
