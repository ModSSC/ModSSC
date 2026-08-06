"""Atomic publication of content-addressed, immutable campaign bundles."""

from __future__ import annotations

import fcntl
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from bench.utils.hashing import hash_any
from bench.utils.io import atomic_write_json

from .errors import CampaignError


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal_bundle(root: Path, *, kind: str) -> dict[str, object]:
    if (root / "BUNDLE.json").exists():
        raise CampaignError("E_CAMPAIGN_BUNDLE_EXISTS", "bundle is already sealed")
    symlinks = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise CampaignError(
            "E_CAMPAIGN_BUNDLE_INVALID",
            f"bundle must not contain symbolic links: {', '.join(sorted(symlinks))}",
        )
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    if not files:
        raise CampaignError("E_CAMPAIGN_BUNDLE_INVALID", "bundle contains no files")
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": kind,
        "files": files,
        "content_sha256": hash_any(files),
    }
    atomic_write_json(root / "BUNDLE.json", payload)
    return payload


@contextmanager
def immutable_bundle(destination: Path, *, kind: str) -> Iterator[Path]:
    final = destination.resolve(strict=False)
    final.parent.mkdir(parents=True, exist_ok=True)
    lock_path = final.parent / f".{final.name}.bundle.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    staging: Path | None = None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.path.lexists(final):
            raise CampaignError(
                "E_CAMPAIGN_BUNDLE_EXISTS", f"immutable bundle already exists: {final}"
            )
        staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent))
        yield staging
        seal_bundle(staging, kind=kind)
        os.rename(staging, final)
        staging = None
    except BlockingIOError as exc:
        raise CampaignError(
            "E_CAMPAIGN_BUNDLE_BUSY", f"bundle publication is already active: {final}"
        ) from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


__all__ = ["immutable_bundle", "seal_bundle"]
