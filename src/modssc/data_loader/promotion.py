from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from modssc.data_loader import content
from modssc.data_loader.cache import (
    CacheLayout,
    _rebuild_index_atomic_locked,
    index_lock,
)
from modssc.data_loader.errors import CachePromotionError, ManifestError
from modssc.data_loader.manifest import Manifest

_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_TRANSACTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MANIFEST_SCHEMA_VERSION = 1
_PROMOTION_SCHEMA_VERSION = 1
_BUFFER_SIZE = 1024 * 1024
_ATOMIC_NOREPLACE_UNSUPPORTED_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
    }
)
_INTENT_KEYS = frozenset(
    {
        "cache_root",
        "entries",
        "request",
        "schema_version",
        "staging_root",
        "transaction_id",
    }
)
_REQUEST_KEYS = frozenset({"content_manifest_sha256", "content_sha256", "fingerprint"})
_RECEIPT_KEYS = frozenset(
    {
        "cache_root",
        "completed_at",
        "index_sha256",
        "intent_sha256",
        "items",
        "schema_version",
        "staging_root",
        "transaction_id",
    }
)


@dataclass(frozen=True)
class CacheEntryExpectation:
    """An immutable identity expected in a staging cache."""

    fingerprint: str
    content_sha256: str | None = None
    content_manifest_sha256: str | None = None


@dataclass(frozen=True)
class CachePromotionItem:
    fingerprint: str
    dataset_id: str
    canonical_uri: str
    disposition: Literal["promoted", "reused"]
    manifest_sha256: str
    content_manifest_sha256: str
    content_sha256: str
    processed_file_count: int
    source_file_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_uri": self.canonical_uri,
            "content_manifest_sha256": self.content_manifest_sha256,
            "content_sha256": self.content_sha256,
            "dataset_id": self.dataset_id,
            "disposition": self.disposition,
            "fingerprint": self.fingerprint,
            "manifest_sha256": self.manifest_sha256,
            "processed_file_count": self.processed_file_count,
            "source_file_count": self.source_file_count,
        }


@dataclass(frozen=True)
class CachePromotionReport:
    transaction_id: str
    staging_root: Path
    destination_root: Path
    items: tuple[CachePromotionItem, ...]
    index_sha256: str
    receipt_path: Path
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "destination_root": str(self.destination_root),
            "index_sha256": self.index_sha256,
            "items": [item.to_dict() for item in self.items],
            "receipt_path": str(self.receipt_path),
            "receipt_sha256": self.receipt_sha256,
            "staging_root": str(self.staging_root),
            "transaction_id": self.transaction_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class _FileSnapshot:
    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    links: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


class _AtomicNoReplaceUnsupported(CachePromotionError):
    """The filesystem cannot provide an atomic exclusive rename."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_regular_file(
    path: Path,
    *,
    capture_bytes: bool = True,
) -> tuple[bytes, _FileSnapshot]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CachePromotionError(f"Cannot securely open cache file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CachePromotionError(f"Cache artifact is not a regular file: {path}")
        if before.st_nlink != 1:
            raise CachePromotionError(f"Cache artifact must have one hard link: {path}")
        try:
            lexical = os.lstat(path)
        except OSError as exc:
            raise CachePromotionError(f"Cache artifact disappeared while reading: {path}") from exc
        if _stat_identity(lexical) != _stat_identity(before):
            raise CachePromotionError(f"Cache artifact changed while opening: {path}")

        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if capture_bytes else None
        while True:
            chunk = os.read(descriptor, _BUFFER_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise CachePromotionError(f"Cache artifact changed while reading: {path}")
        try:
            lexical_after = os.lstat(path)
        except OSError as exc:
            raise CachePromotionError(f"Cache artifact disappeared while reading: {path}") from exc
        if _stat_identity(after) != _stat_identity(lexical_after):
            raise CachePromotionError(f"Cache artifact was replaced while reading: {path}")
        payload = b"".join(chunks) if chunks is not None else b""
        return payload, _FileSnapshot(
            device=int(after.st_dev),
            inode=int(after.st_ino),
            mode=int(after.st_mode),
            uid=int(after.st_uid),
            gid=int(after.st_gid),
            links=int(after.st_nlink),
            size=int(after.st_size),
            mtime_ns=int(after.st_mtime_ns),
            ctime_ns=int(after.st_ctime_ns),
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(descriptor)


def _canonical_root(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise CachePromotionError(f"{label} must already exist: {absolute}") from exc
    if resolved != absolute:
        raise CachePromotionError(f"{label} must not contain symlink components: {absolute}")
    try:
        root_stat = os.lstat(absolute)
    except OSError as exc:
        raise CachePromotionError(f"Cannot inspect {label}: {absolute}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise CachePromotionError(f"{label} is not a directory: {absolute}")
    return absolute


def _relative_path(value: str, *, purpose: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise CachePromotionError(f"Invalid {purpose} relative path: {value!r}")
    return path


def _relative_to(path: Path, root: Path, *, purpose: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise CachePromotionError(f"{purpose} escapes cache root: {path}") from exc


def _assert_no_symlink_chain(root: Path, relative: Path, *, allow_missing: bool) -> None:
    current = root
    for part in relative.parts:
        current /= part
        try:
            current_stat = os.lstat(current)
        except FileNotFoundError:
            if allow_missing:
                return
            raise CachePromotionError(f"Cache artifact is missing: {current}") from None
        except OSError as exc:
            raise CachePromotionError(f"Cannot inspect cache path: {current}") from exc
        if stat.S_ISLNK(current_stat.st_mode):
            raise CachePromotionError(f"Cache path contains a symlink: {current}")


def _scan_directory(path: Path, *, cache_root: Path) -> list[dict[str, Any]]:
    relative_root = _relative_path(
        _relative_to(path, cache_root, purpose="processed directory"),
        purpose="processed directory",
    )
    _assert_no_symlink_chain(cache_root, relative_root, allow_missing=False)
    try:
        directory_stat = os.lstat(path)
    except OSError as exc:
        raise CachePromotionError(f"Processed cache directory is missing: {path}") from exc
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise CachePromotionError(f"Processed cache is not a directory: {path}")
    if directory_stat.st_dev != os.lstat(cache_root).st_dev:
        raise CachePromotionError(f"Processed cache is on a nested filesystem: {path}")

    files: list[dict[str, Any]] = []
    for current, directory_names, file_names in os.walk(path, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            child = current_path / name
            child_stat = os.lstat(child)
            if not stat.S_ISDIR(child_stat.st_mode) or stat.S_ISLNK(child_stat.st_mode):
                raise CachePromotionError(f"Processed cache contains a non-directory: {child}")
        for name in file_names:
            child = current_path / name
            _, snapshot = _read_regular_file(child, capture_bytes=False)
            files.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "sha256": snapshot.sha256,
                    "size_bytes": snapshot.size,
                }
            )
    files.sort(key=lambda item: str(item["path"]))
    return files


def _expect_sha256(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise CachePromotionError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _expectation_payload(entries: Sequence[CacheEntryExpectation]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for entry in entries:
        if (
            not isinstance(entry.fingerprint, str)
            or _FINGERPRINT.fullmatch(entry.fingerprint) is None
        ):
            raise CachePromotionError(
                f"fingerprint must be 64 lowercase hexadecimal characters: {entry.fingerprint!r}"
            )
        if entry.fingerprint in fingerprints:
            raise CachePromotionError(f"Duplicate cache fingerprint: {entry.fingerprint}")
        fingerprints.add(entry.fingerprint)
        normalized.append(
            {
                "content_manifest_sha256": _expect_sha256(
                    entry.content_manifest_sha256,
                    field="content_manifest_sha256",
                ),
                "content_sha256": _expect_sha256(
                    entry.content_sha256,
                    field="content_sha256",
                ),
                "fingerprint": entry.fingerprint,
            }
        )
    if not normalized:
        raise CachePromotionError("At least one cache fingerprint is required")
    normalized.sort(key=lambda item: str(item["fingerprint"]))
    return normalized


def _parse_main_manifest(payload: bytes, *, fingerprint: str, path: Path) -> Manifest:
    try:
        manifest = Manifest.from_json(payload.decode("utf-8"))
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CachePromotionError(f"Invalid main dataset manifest: {path}") from exc
    if manifest.schema_version != _MANIFEST_SCHEMA_VERSION:
        raise CachePromotionError(f"Unsupported main dataset manifest schema: {path}")
    if manifest.fingerprint != fingerprint:
        raise CachePromotionError(f"Main manifest fingerprint differs: {path}")
    for field in ("canonical_uri", "provider", "dataset_id"):
        if not isinstance(manifest.identity.get(field), str) or not manifest.identity[field]:
            raise CachePromotionError(f"Main manifest has invalid identity field {field!r}: {path}")
    return manifest


def _file_plan(path: Path, *, root: Path, snapshot: _FileSnapshot) -> dict[str, Any]:
    if snapshot.device != os.lstat(root).st_dev:
        raise CachePromotionError(f"Cache artifact is on a nested filesystem: {path}")
    return {
        "path": _relative_to(path, root, purpose="cache artifact"),
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size,
    }


def _build_entry_plan(
    staging: CacheLayout,
    destination: CacheLayout,
    expectation: Mapping[str, Any],
) -> dict[str, Any]:
    fingerprint = str(expectation["fingerprint"])
    manifest_path = _artifact_path(
        staging.root,
        _relative_to(
            staging.manifest_path(fingerprint),
            staging.root,
            purpose="main manifest",
        ),
        purpose="main manifest",
    )
    manifest_bytes, manifest_snapshot = _read_regular_file(manifest_path)
    manifest = _parse_main_manifest(
        manifest_bytes,
        fingerprint=fingerprint,
        path=manifest_path,
    )
    content_path = _artifact_path(
        staging.root,
        _relative_to(
            staging.content_manifest_path(fingerprint),
            staging.root,
            purpose="content manifest",
        ),
        purpose="content manifest",
    )
    content_bytes, content_snapshot = _read_regular_file(content_path)
    try:
        content_manifest = json.loads(content_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CachePromotionError(f"Invalid dataset content manifest: {content_path}") from exc
    if not isinstance(content_manifest, Mapping):
        raise CachePromotionError(f"Dataset content manifest root is not a mapping: {content_path}")
    try:
        records = content._validate_manifest(content_manifest, fingerprint=fingerprint)
        evidence = content.verify_content_manifest(
            staging,
            fingerprint,
            identity=manifest.identity,
            rehash=True,
        )
    except ManifestError as exc:
        raise CachePromotionError(f"Dataset content preflight failed: {fingerprint}") from exc

    expected_content = expectation.get("content_sha256")
    if expected_content is not None and evidence["content_sha256"] != expected_content:
        raise CachePromotionError(f"Dataset content digest differs: {fingerprint}")
    expected_manifest = expectation.get("content_manifest_sha256")
    if expected_manifest is not None and content_snapshot.sha256 != expected_manifest:
        raise CachePromotionError(f"Dataset content-manifest digest differs: {fingerprint}")
    if evidence["content_manifest_sha256"] != content_snapshot.sha256:
        raise CachePromotionError(
            f"Dataset content manifest changed during preflight: {fingerprint}"
        )

    processed_records = {
        str(record["path"]): record for record in records if record["kind"] == "processed"
    }
    processed_dir = staging.processed_dir(fingerprint)
    processed_files = _scan_directory(processed_dir, cache_root=staging.root)
    if set(processed_records) != {str(item["path"]) for item in processed_files}:
        raise CachePromotionError(
            f"Processed cache inventory differs from content manifest: {fingerprint}"
        )
    for item in processed_files:
        record = processed_records[str(item["path"])]
        if item["size_bytes"] != record["size_bytes"]:
            raise CachePromotionError(f"Processed cache size differs: {fingerprint}")
        if item["sha256"] != record["storage_sha256"]:
            raise CachePromotionError(f"Processed cache storage digest differs: {fingerprint}")

    source_files: list[dict[str, Any]] = []
    staging_source_root = content._source_root(staging, manifest.identity)
    destination_source_root = content._source_root(destination, manifest.identity)
    for record in records:
        if record["kind"] != "source":
            continue
        if staging_source_root is None or destination_source_root is None:
            raise CachePromotionError(f"Source record has no native cache root: {fingerprint}")
        relative = _relative_path(str(record["path"]), purpose="source record")
        staging_source = staging_source_root / relative
        source_relative = _relative_path(
            _relative_to(staging_source, staging.root, purpose="source record"),
            purpose="source record",
        )
        _assert_no_symlink_chain(staging.root, source_relative, allow_missing=False)
        _, snapshot = _read_regular_file(staging_source, capture_bytes=False)
        if snapshot.device != os.lstat(staging.root).st_dev:
            raise CachePromotionError(f"Source record is on a nested filesystem: {staging_source}")
        if snapshot.size != record["size_bytes"] or snapshot.sha256 != record["sha256"]:
            raise CachePromotionError(
                f"Source record differs from content manifest: {staging_source}"
            )
        destination_source = destination_source_root / relative
        source_files.append(
            {
                "destination": _relative_to(
                    destination_source,
                    destination.root,
                    purpose="source destination",
                ),
                "path": source_relative.as_posix(),
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size,
            }
        )
    source_files.sort(key=lambda item: str(item["destination"]))

    return {
        "canonical_uri": str(manifest.identity["canonical_uri"]),
        "content_manifest": _file_plan(
            content_path,
            root=staging.root,
            snapshot=content_snapshot,
        ),
        "content_sha256": str(evidence["content_sha256"]),
        "dataset_id": str(manifest.identity["dataset_id"]),
        "fingerprint": fingerprint,
        "main_manifest": _file_plan(
            manifest_path,
            root=staging.root,
            snapshot=manifest_snapshot,
        ),
        "processed": {
            "destination": _relative_to(
                destination.processed_dir(fingerprint),
                destination.root,
                purpose="processed destination",
            ),
            "files": processed_files,
            "path": _relative_to(
                processed_dir,
                staging.root,
                purpose="processed cache",
            ),
        },
        "source_files": source_files,
    }


def _build_intent(
    *,
    staging: CacheLayout,
    destination: CacheLayout,
    transaction_id: str,
    request: list[dict[str, Any]],
) -> dict[str, Any]:
    entries = [_build_entry_plan(staging, destination, expectation) for expectation in request]
    for entry in entries:
        _reject_incomplete_published_destination(entry, destination=destination)
    source_destinations: dict[str, tuple[str, int]] = {}
    for entry in entries:
        for source in entry["source_files"]:
            destination_path = str(source["destination"])
            evidence = (str(source["sha256"]), int(source["size_bytes"]))
            previous = source_destinations.setdefault(destination_path, evidence)
            if previous != evidence:
                raise CachePromotionError(
                    f"Source records disagree about destination: {destination_path}"
                )
    return {
        "cache_root": str(destination.root),
        "entries": entries,
        "request": request,
        "schema_version": _PROMOTION_SCHEMA_VERSION,
        "staging_root": str(staging.root),
        "transaction_id": transaction_id,
    }


def _reject_incomplete_published_destination(
    entry: Mapping[str, Any],
    *,
    destination: CacheLayout,
) -> None:
    main = entry["main_manifest"]
    main_path = _artifact_path(
        destination.root,
        str(main["path"]),
        purpose="published main manifest",
    )
    if _file_state(main_path, root=destination.root, expected=main) == "absent":
        return
    processed = entry["processed"]
    processed_path = _artifact_path(
        destination.root,
        str(processed["destination"]),
        purpose="published processed cache",
    )
    if _directory_state(processed_path, root=destination.root, expected=processed) != "exact":
        raise CachePromotionError("Published destination cache is incomplete")
    content_manifest = entry["content_manifest"]
    content_path = _artifact_path(
        destination.root,
        str(content_manifest["path"]),
        purpose="published content manifest",
    )
    if _file_state(content_path, root=destination.root, expected=content_manifest) != "exact":
        raise CachePromotionError("Published destination cache is incomplete")
    for source in entry["source_files"]:
        source_path = _artifact_path(
            destination.root,
            str(source["destination"]),
            purpose="published source record",
        )
        if _file_state(source_path, root=destination.root, expected=source) != "exact":
            raise CachePromotionError("Published destination cache is incomplete")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - Windows has no directory fsync
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir():
        raise CachePromotionError(f"Promotion-record directory is missing: {path.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise
    except OSError as exc:
        raise CachePromotionError(f"Cannot create immutable promotion record: {path}") from exc
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_json_record(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    payload, _ = _read_regular_file(path)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CachePromotionError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CachePromotionError(f"{label} root is not a mapping: {path}")
    if payload != _canonical_json(value) + b"\n":
        raise CachePromotionError(f"{label} is not canonical JSON: {path}")
    return value, payload


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CachePromotionError(f"Cannot securely open cache lock: {path}") from exc
    lock_stat = os.fstat(descriptor)
    if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
        os.close(descriptor)
        raise CachePromotionError(f"Cache lock must be a single-link regular file: {path}")
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
        os.write(descriptor, str(os.getpid()).encode("ascii"))
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


def _artifact_path(root: Path, value: str, *, purpose: str) -> Path:
    relative = _relative_path(value, purpose=purpose)
    _assert_no_symlink_chain(root, relative, allow_missing=True)
    return root / relative


def _validate_intent_envelope(
    intent: Mapping[str, Any],
    *,
    staging: CacheLayout,
    destination: CacheLayout,
    transaction_id: str,
) -> None:
    if set(intent) != _INTENT_KEYS:
        raise CachePromotionError("Cache-promotion intent keys differ")
    schema_version = intent.get("schema_version")
    if type(schema_version) is not int or schema_version != _PROMOTION_SCHEMA_VERSION:
        raise CachePromotionError("Unsupported cache-promotion intent schema")
    if intent.get("transaction_id") != transaction_id:
        raise CachePromotionError("Cache-promotion intent transaction differs")
    if intent.get("staging_root") != str(staging.root):
        raise CachePromotionError("Cache-promotion intent staging root differs")
    if intent.get("cache_root") != str(destination.root):
        raise CachePromotionError("Cache-promotion intent destination root differs")


def _normalize_intent_request(raw_request: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_request, list) or not raw_request:
        raise CachePromotionError("Cache-promotion intent request is empty")
    expectations: list[CacheEntryExpectation] = []
    for raw in raw_request:
        if not isinstance(raw, Mapping) or set(raw) != _REQUEST_KEYS:
            raise CachePromotionError("Cache-promotion intent request item keys differ")
        expectations.append(
            CacheEntryExpectation(
                fingerprint=raw.get("fingerprint"),
                content_sha256=raw.get("content_sha256"),
                content_manifest_sha256=raw.get("content_manifest_sha256"),
            )
        )
    normalized = _expectation_payload(expectations)
    if normalized != raw_request:
        raise CachePromotionError("Cache-promotion intent request is not canonical")
    return normalized


def _all_staged_request(
    *,
    staging: CacheLayout,
    destination: CacheLayout,
    transaction_id: str,
    intent_path: Path,
    transaction_root: Path,
) -> list[dict[str, Any]]:
    if os.path.lexists(intent_path):
        transaction_relative = _relative_path(
            _relative_to(
                transaction_root,
                destination.root,
                purpose="transaction directory",
            ),
            purpose="transaction directory",
        )
        _assert_no_symlink_chain(
            destination.root,
            transaction_relative,
            allow_missing=False,
        )
        intent, _ = _read_json_record(intent_path, label="promotion intent")
        _validate_intent_envelope(
            intent,
            staging=staging,
            destination=destination,
            transaction_id=transaction_id,
        )
        return _normalize_intent_request(intent.get("request"))

    _assert_no_symlink_chain(
        staging.root,
        Path("manifests"),
        allow_missing=False,
    )
    fingerprints = sorted(
        match.group(0)[:-5]
        for path in staging.manifests_root.iterdir()
        if (match := re.fullmatch(r"[0-9a-f]{64}\.json", path.name)) is not None
    )
    return _expectation_payload(
        [CacheEntryExpectation(fingerprint=fingerprint) for fingerprint in fingerprints]
    )


def _file_state(
    path: Path,
    *,
    root: Path,
    expected: Mapping[str, Any],
) -> Literal["absent", "exact"]:
    relative = _relative_path(_relative_to(path, root, purpose="artifact"), purpose="artifact")
    _assert_no_symlink_chain(root, relative, allow_missing=True)
    if not os.path.lexists(path):
        return "absent"
    _, snapshot = _read_regular_file(path, capture_bytes=False)
    if snapshot.size != expected.get("size_bytes") or snapshot.sha256 != expected.get("sha256"):
        raise CachePromotionError(f"Existing cache artifact conflicts with intent: {path}")
    return "exact"


def _directory_state(
    path: Path,
    *,
    root: Path,
    expected: Mapping[str, Any],
) -> Literal["absent", "exact"]:
    state = _directory_progress(path, root=root, expected=expected)
    if state == "partial":
        raise CachePromotionError(f"Existing processed cache conflicts with intent: {path}")
    return state


def _directory_progress(
    path: Path,
    *,
    root: Path,
    expected: Mapping[str, Any],
) -> Literal["absent", "partial", "exact"]:
    relative = _relative_path(_relative_to(path, root, purpose="directory"), purpose="directory")
    _assert_no_symlink_chain(root, relative, allow_missing=True)
    if not os.path.lexists(path):
        return "absent"
    current = _scan_directory(path, cache_root=root)
    expected_files = expected.get("files")
    if not isinstance(expected_files, list):
        raise CachePromotionError("Processed cache intent files are invalid")
    expected_by_path = {str(item.get("path")): item for item in expected_files}
    if len(expected_by_path) != len(expected_files):
        raise CachePromotionError("Processed cache intent contains duplicate files")
    for item in current:
        if expected_by_path.get(str(item["path"])) != item:
            raise CachePromotionError(f"Existing processed cache conflicts with intent: {path}")

    expected_directories: set[str] = set()
    for item in expected_files:
        relative_file = _relative_path(str(item.get("path")), purpose="processed file")
        for parent in relative_file.parents:
            if parent != Path("."):
                expected_directories.add(parent.as_posix())
    for current_root, directory_names, _ in os.walk(path, followlinks=False):
        current_path = Path(current_root)
        for name in directory_names:
            child = current_path / name
            if child.relative_to(path).as_posix() not in expected_directories:
                raise CachePromotionError(f"Existing processed cache conflicts with intent: {path}")
    if current != expected_files:
        return "partial"
    if any(not (path / directory).is_dir() for directory in expected_directories):
        raise CachePromotionError(f"Existing processed cache conflicts with intent: {path}")
    return "exact"


def _mkdir_safe(root: Path, parent: Path) -> None:
    relative = _relative_path(
        _relative_to(parent, root, purpose="destination parent"),
        purpose="destination parent",
    )
    _assert_no_symlink_chain(root, relative, allow_missing=True)
    parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_chain(root, relative, allow_missing=False)
    if os.lstat(parent).st_dev != os.lstat(root).st_dev:
        raise CachePromotionError(f"Destination parent is on a nested filesystem: {parent}")


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise _AtomicNoReplaceUnsupported("renameat2(RENAME_NOREPLACE) is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
    elif sys.platform == "darwin":  # pragma: no cover - exercised on macOS CI
        library = ctypes.CDLL(None, use_errno=True)
        renamex_np = getattr(library, "renamex_np", None)
        if renamex_np is None:
            raise _AtomicNoReplaceUnsupported("renamex_np(RENAME_EXCL) is unavailable")
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(source), os.fsencode(destination), 0x00000004)
    elif os.name == "nt":  # pragma: no cover - Windows rename already rejects destinations
        try:
            os.rename(source, destination)
        except FileExistsError:
            result = -1
            ctypes.set_errno(errno.EEXIST)
        else:
            result = 0
    else:  # pragma: no cover - fail closed on an unknown rename contract
        raise CachePromotionError("No atomic no-clobber rename primitive is available")
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise CachePromotionError(f"Cache destination appeared concurrently: {destination}")
        if error in _ATOMIC_NOREPLACE_UNSUPPORTED_ERRNOS:
            raise _AtomicNoReplaceUnsupported(
                f"Atomic no-clobber rename is unsupported for {source} -> {destination}: "
                f"{os.strerror(error)}"
            )
        raise CachePromotionError(
            f"Atomic cache promotion failed: {source} -> {destination}: {os.strerror(error)}"
        )


def _publish_temp_path(
    publish_root: Path,
    destination: Path,
    *,
    destination_root: Path,
    expected: Mapping[str, Any],
) -> Path:
    relative = _relative_to(destination, destination_root, purpose="published artifact")
    identity = _canonical_json(
        {
            "path": relative,
            "sha256": expected.get("sha256"),
            "size_bytes": expected.get("size_bytes"),
        }
    )
    return publish_root / f"{_sha256_bytes(identity)}.partial"


def _tree_claim_path(
    publish_root: Path,
    destination: Path,
    *,
    destination_root: Path,
) -> Path:
    relative = _relative_to(destination, destination_root, purpose="published directory")
    return publish_root / f"{_sha256_bytes(relative.encode('utf-8'))}.tree-claim.json"


def _tree_claim_payload(
    plan: Mapping[str, Any],
    destination: Path,
    *,
    destination_root: Path,
) -> dict[str, Any]:
    return {
        "destination": _relative_to(
            destination,
            destination_root,
            purpose="published directory",
        ),
        "plan_sha256": _sha256_bytes(_canonical_json(plan)),
        "schema_version": 1,
    }


def _tree_claim_state(
    claim_path: Path,
    *,
    expected: Mapping[str, Any],
) -> Literal["absent", "exact"]:
    if not os.path.lexists(claim_path):
        return "absent"
    claim, _ = _read_json_record(claim_path, label="processed-tree publication claim")
    if claim != expected:
        raise CachePromotionError(f"Processed-tree publication claim differs: {claim_path}")
    return "exact"


def _write_tree_claim(
    claim_path: Path,
    *,
    expected: Mapping[str, Any],
) -> bool:
    payload = _canonical_json(expected) + b"\n"
    try:
        _write_exclusive(claim_path, payload)
    except FileExistsError:
        if _tree_claim_state(claim_path, expected=expected) != "exact":
            raise CachePromotionError(
                f"Processed-tree publication claim failed verification: {claim_path}"
            ) from None
        return False
    return True


def _reconcile_publish_temp(
    temp: Path,
    destination: Path,
    *,
    destination_root: Path,
    expected: Mapping[str, Any],
) -> None:
    temp_relative = _relative_path(
        _relative_to(temp, destination_root, purpose="publish temporary file"),
        purpose="publish temporary file",
    )
    _assert_no_symlink_chain(destination_root, temp_relative, allow_missing=True)
    if not os.path.lexists(temp):
        return
    temp_stat = os.lstat(temp)
    if not stat.S_ISREG(temp_stat.st_mode) or stat.S_ISLNK(temp_stat.st_mode):
        raise CachePromotionError(f"Publish temporary artifact is unsafe: {temp}")
    if temp_stat.st_dev != os.lstat(destination_root).st_dev:
        raise CachePromotionError(f"Publish temporary artifact is on a nested filesystem: {temp}")

    if temp_stat.st_nlink == 2 and os.path.lexists(destination):
        destination_relative = _relative_path(
            _relative_to(destination, destination_root, purpose="published artifact"),
            purpose="published artifact",
        )
        _assert_no_symlink_chain(
            destination_root,
            destination_relative,
            allow_missing=False,
        )
        destination_stat = os.lstat(destination)
        if (
            not stat.S_ISREG(destination_stat.st_mode)
            or destination_stat.st_dev != temp_stat.st_dev
            or destination_stat.st_ino != temp_stat.st_ino
        ):
            raise CachePromotionError(
                f"Publish temporary artifact has an unexpected hard link: {temp}"
            )
        os.unlink(temp)
        _fsync_directory(temp.parent)
        return
    if temp_stat.st_nlink != 1:
        raise CachePromotionError(f"Publish temporary artifact has an unexpected hard link: {temp}")
    if os.path.lexists(destination):
        if _file_state(destination, root=destination_root, expected=expected) != "exact":
            raise CachePromotionError(f"Published artifact failed verification: {destination}")
        os.unlink(temp)
        _fsync_directory(temp.parent)


def _copy_regular_file_exclusive(
    source: Path,
    temp: Path,
    *,
    expected: Mapping[str, Any],
) -> None:
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        destination_flags |= os.O_NOFOLLOW
    try:
        source_descriptor = os.open(source, source_flags)
    except OSError as exc:
        raise CachePromotionError(f"Cannot securely open promotion source: {source}") from exc
    destination_descriptor: int | None = None
    created = False
    try:
        source_before = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_before.st_mode) or source_before.st_nlink != 1:
            raise CachePromotionError(f"Promotion source is unsafe: {source}")
        if _stat_identity(os.lstat(source)) != _stat_identity(source_before):
            raise CachePromotionError(f"Promotion source changed while opening: {source}")
        try:
            destination_descriptor = os.open(temp, destination_flags, 0o600)
        except OSError as exc:
            raise CachePromotionError(f"Cannot create publish temporary file: {temp}") from exc
        created = True
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(source_descriptor, _BUFFER_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise CachePromotionError(f"Cannot write publish temporary file: {temp}")
                view = view[written:]
        os.fsync(destination_descriptor)
        source_after = os.fstat(source_descriptor)
        if _stat_identity(source_before) != _stat_identity(source_after):
            raise CachePromotionError(f"Promotion source changed while copying: {source}")
        if _stat_identity(source_after) != _stat_identity(os.lstat(source)):
            raise CachePromotionError(f"Promotion source was replaced while copying: {source}")
        if size != expected.get("size_bytes") or digest.hexdigest() != expected.get("sha256"):
            raise CachePromotionError(f"Promotion source differs from intent: {source}")
    except BaseException:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
            destination_descriptor = None
        if created and os.path.lexists(temp):
            temp_stat = os.lstat(temp)
            if stat.S_ISREG(temp_stat.st_mode) and temp_stat.st_nlink == 1:
                os.unlink(temp)
        raise
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(source_descriptor)
    _fsync_directory(temp.parent)


def _publish_file_fallback(
    plan: Mapping[str, Any],
    *,
    source: Path,
    destination: Path,
    staging_root: Path,
    destination_root: Path,
    publish_root: Path,
) -> bool:
    _mkdir_safe(destination_root, publish_root)
    _mkdir_safe(destination_root, destination.parent)
    temp = _publish_temp_path(
        publish_root,
        destination,
        destination_root=destination_root,
        expected=plan,
    )
    _reconcile_publish_temp(
        temp,
        destination,
        destination_root=destination_root,
        expected=plan,
    )
    if _file_state(destination, root=destination_root, expected=plan) == "exact":
        return False
    if _file_state(source, root=staging_root, expected=plan) != "exact":
        raise CachePromotionError(f"Promotion source and destination are both absent: {source}")

    if os.path.lexists(temp):
        try:
            _file_state(temp, root=destination_root, expected=plan)
        except CachePromotionError:
            temp_stat = os.lstat(temp)
            if not stat.S_ISREG(temp_stat.st_mode) or temp_stat.st_nlink != 1:
                raise
            os.unlink(temp)
            _fsync_directory(temp.parent)
    if not os.path.lexists(temp):
        _copy_regular_file_exclusive(source, temp, expected=plan)
        if _file_state(temp, root=destination_root, expected=plan) != "exact":
            raise CachePromotionError(f"Publish temporary file failed verification: {temp}")

    published = True
    try:
        os.link(temp, destination, follow_symlinks=False)
    except FileExistsError:
        if _file_state(destination, root=destination_root, expected=plan) != "exact":
            raise CachePromotionError(
                f"Cache destination appeared concurrently: {destination}"
            ) from None
        published = False
    except OSError as exc:
        raise CachePromotionError(
            f"Exclusive cache publication failed: {temp} -> {destination}"
        ) from exc
    _fsync_directory(destination.parent)
    _reconcile_publish_temp(
        temp,
        destination,
        destination_root=destination_root,
        expected=plan,
    )
    if _file_state(destination, root=destination_root, expected=plan) != "exact":
        raise CachePromotionError(f"Published cache file failed verification: {destination}")
    return published


def _promote_file(
    plan: Mapping[str, Any],
    *,
    staging_root: Path,
    destination_root: Path,
    publish_root: Path,
) -> bool:
    source = _artifact_path(staging_root, str(plan["path"]), purpose="source artifact")
    destination_value = str(plan.get("destination", plan["path"]))
    destination = _artifact_path(
        destination_root,
        destination_value,
        purpose="destination artifact",
    )
    temp = _publish_temp_path(
        publish_root,
        destination,
        destination_root=destination_root,
        expected=plan,
    )
    _reconcile_publish_temp(
        temp,
        destination,
        destination_root=destination_root,
        expected=plan,
    )
    destination_state = _file_state(destination, root=destination_root, expected=plan)
    source_state = _file_state(source, root=staging_root, expected=plan)
    if destination_state == "exact":
        return False
    if source_state != "exact":
        raise CachePromotionError(f"Promotion source and destination are both absent: {source}")
    _mkdir_safe(destination_root, destination.parent)
    try:
        _atomic_rename_noreplace(source, destination)
    except _AtomicNoReplaceUnsupported:
        return _publish_file_fallback(
            plan,
            source=source,
            destination=destination,
            staging_root=staging_root,
            destination_root=destination_root,
            publish_root=publish_root,
        )
    _fsync_directory(source.parent)
    _fsync_directory(destination.parent)
    if _file_state(destination, root=destination_root, expected=plan) != "exact":
        raise CachePromotionError(f"Promoted cache file failed verification: {destination}")
    return True


def _promote_directory(
    plan: Mapping[str, Any],
    *,
    staging_root: Path,
    destination_root: Path,
    publish_root: Path,
) -> bool:
    source = _artifact_path(staging_root, str(plan["path"]), purpose="processed source")
    destination = _artifact_path(
        destination_root,
        str(plan["destination"]),
        purpose="processed destination",
    )
    expected_files = plan.get("files")
    if not isinstance(expected_files, list):
        raise CachePromotionError("Processed cache intent files are invalid")
    claim_path = _tree_claim_path(
        publish_root,
        destination,
        destination_root=destination_root,
    )
    claim_payload = _tree_claim_payload(
        plan,
        destination,
        destination_root=destination_root,
    )
    for expected_file in expected_files:
        relative = _relative_path(
            str(expected_file.get("path")),
            purpose="processed file",
        )
        destination_file = destination / relative
        temp = _publish_temp_path(
            publish_root,
            destination_file,
            destination_root=destination_root,
            expected=expected_file,
        )
        _reconcile_publish_temp(
            temp,
            destination_file,
            destination_root=destination_root,
            expected=expected_file,
        )
    destination_state = _directory_progress(
        destination,
        root=destination_root,
        expected=plan,
    )
    source_state = _directory_state(source, root=staging_root, expected=plan)
    if destination_state == "exact":
        return False
    if source_state != "exact":
        raise CachePromotionError(f"Processed source and destination are both absent: {source}")
    _mkdir_safe(destination_root, destination.parent)
    directory_created = False
    if destination_state == "absent":
        try:
            _atomic_rename_noreplace(source, destination)
        except _AtomicNoReplaceUnsupported:
            _mkdir_safe(destination_root, publish_root)
            claim_created = _write_tree_claim(claim_path, expected=claim_payload)
            destination_state = _directory_progress(
                destination,
                root=destination_root,
                expected=plan,
            )
            if destination_state != "absent" and claim_created:
                raise CachePromotionError(
                    f"Processed destination appeared before it was claimed: {destination}"
                ) from None
        else:
            _fsync_directory(source.parent)
            _fsync_directory(destination.parent)
            if _directory_state(destination, root=destination_root, expected=plan) != "exact":
                raise CachePromotionError(
                    f"Promoted processed cache failed verification: {destination}"
                )
            return True
    elif _tree_claim_state(claim_path, expected=claim_payload) != "exact":
        raise CachePromotionError(
            f"Partial processed destination has no exact transaction claim: {destination}"
        )

    _mkdir_safe(destination_root, publish_root)
    if destination_state == "absent":
        try:
            os.mkdir(destination, 0o700)
        except FileExistsError:
            raise CachePromotionError(
                f"Processed destination appeared concurrently: {destination}"
            ) from None
        except OSError as exc:
            raise CachePromotionError(
                f"Cannot create claimed processed destination: {destination}"
            ) from exc
        _fsync_directory(destination.parent)
        directory_created = True
    published = directory_created
    for expected_file in expected_files:
        relative = _relative_path(
            str(expected_file["path"]),
            purpose="processed file",
        )
        source_file = source / relative
        destination_file = destination / relative
        file_plan = {
            "destination": _relative_to(
                destination_file,
                destination_root,
                purpose="processed destination file",
            ),
            "path": _relative_to(
                source_file,
                staging_root,
                purpose="processed source file",
            ),
            "sha256": expected_file["sha256"],
            "size_bytes": expected_file["size_bytes"],
        }
        if _publish_file_fallback(
            file_plan,
            source=source_file,
            destination=destination_file,
            staging_root=staging_root,
            destination_root=destination_root,
            publish_root=publish_root,
        ):
            published = True
    _fsync_directory(source.parent)
    _fsync_directory(destination.parent)
    if _directory_state(destination, root=destination_root, expected=plan) != "exact":
        raise CachePromotionError(f"Promoted processed cache failed verification: {destination}")
    return published


def _promotion_checkpoint(_phase: str) -> None:
    """Fault-injection seam used by recovery tests."""


def _validate_intent(
    intent: Mapping[str, Any],
    *,
    staging: CacheLayout,
    destination: CacheLayout,
    transaction_id: str,
    request: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _validate_intent_envelope(
        intent,
        staging=staging,
        destination=destination,
        transaction_id=transaction_id,
    )
    if intent.get("request") != request:
        raise CachePromotionError("Transaction id was already used for another promotion request")
    entries = intent.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CachePromotionError("Cache-promotion intent has no entries")
    if [entry.get("fingerprint") for entry in entries] != [item["fingerprint"] for item in request]:
        raise CachePromotionError("Cache-promotion intent entry order differs")
    return entries


def _verify_final_entry(
    entry: Mapping[str, Any],
    *,
    destination: CacheLayout,
) -> dict[str, str]:
    fingerprint = str(entry["fingerprint"])
    if (
        _directory_state(
            _artifact_path(
                destination.root,
                str(entry["processed"]["destination"]),
                purpose="processed destination",
            ),
            root=destination.root,
            expected=entry["processed"],
        )
        != "exact"
    ):
        raise CachePromotionError(f"Promoted processed cache is absent: {fingerprint}")
    for source in entry["source_files"]:
        target = _artifact_path(
            destination.root,
            str(source["destination"]),
            purpose="source destination",
        )
        if _file_state(target, root=destination.root, expected=source) != "exact":
            raise CachePromotionError(f"Promoted source record is absent: {target}")
    for key in ("content_manifest", "main_manifest"):
        artifact = entry[key]
        target = _artifact_path(
            destination.root,
            str(artifact["path"]),
            purpose=f"{key} destination",
        )
        if _file_state(target, root=destination.root, expected=artifact) != "exact":
            raise CachePromotionError(f"Promoted manifest is absent: {target}")
    main_bytes, _ = _read_regular_file(destination.manifest_path(fingerprint))
    manifest = _parse_main_manifest(
        main_bytes,
        fingerprint=fingerprint,
        path=destination.manifest_path(fingerprint),
    )
    try:
        evidence = content.verify_content_manifest(
            destination,
            fingerprint,
            identity=manifest.identity,
            rehash=True,
        )
    except ManifestError as exc:
        raise CachePromotionError(f"Live cache verification failed: {fingerprint}") from exc
    if evidence["content_sha256"] != entry["content_sha256"]:
        raise CachePromotionError(f"Live content digest differs: {fingerprint}")
    return evidence


def _report_items(
    entries: Sequence[Mapping[str, Any]],
    *,
    moved_paths: set[str],
) -> tuple[CachePromotionItem, ...]:
    items: list[CachePromotionItem] = []
    for entry in entries:
        paths = {
            str(entry["processed"]["destination"]),
            str(entry["content_manifest"]["path"]),
            str(entry["main_manifest"]["path"]),
            *(str(source["destination"]) for source in entry["source_files"]),
        }
        items.append(
            CachePromotionItem(
                fingerprint=str(entry["fingerprint"]),
                dataset_id=str(entry["dataset_id"]),
                canonical_uri=str(entry["canonical_uri"]),
                disposition="promoted" if paths & moved_paths else "reused",
                manifest_sha256=str(entry["main_manifest"]["sha256"]),
                content_manifest_sha256=str(entry["content_manifest"]["sha256"]),
                content_sha256=str(entry["content_sha256"]),
                processed_file_count=len(entry["processed"]["files"]),
                source_file_count=len(entry["source_files"]),
            )
        )
    return tuple(items)


def _receipt_payload(
    *,
    transaction_id: str,
    staging: CacheLayout,
    destination: CacheLayout,
    intent_sha256: str,
    items: Sequence[CachePromotionItem],
    index_sha256: str,
) -> dict[str, Any]:
    return {
        "cache_root": str(destination.root),
        "completed_at": datetime.now(UTC).isoformat(),
        "index_sha256": index_sha256,
        "intent_sha256": intent_sha256,
        "items": [item.to_dict() for item in items],
        "schema_version": _PROMOTION_SCHEMA_VERSION,
        "staging_root": str(staging.root),
        "transaction_id": transaction_id,
    }


def _report_from_receipt(
    receipt: Mapping[str, Any],
    receipt_bytes: bytes,
    *,
    receipt_path: Path,
    staging: CacheLayout,
    destination: CacheLayout,
    entries: Sequence[Mapping[str, Any]],
    intent_sha256: str,
) -> CachePromotionReport:
    if set(receipt) != _RECEIPT_KEYS:
        raise CachePromotionError("Cache-promotion receipt keys differ")
    schema_version = receipt.get("schema_version")
    if type(schema_version) is not int or schema_version != _PROMOTION_SCHEMA_VERSION:
        raise CachePromotionError("Unsupported cache-promotion receipt schema")
    if receipt.get("transaction_id") != receipt_path.parent.name:
        raise CachePromotionError("Cache-promotion receipt transaction differs")
    if receipt.get("intent_sha256") != intent_sha256:
        raise CachePromotionError("Cache-promotion receipt intent digest differs")
    if receipt.get("staging_root") != str(staging.root):
        raise CachePromotionError("Cache-promotion receipt staging root differs")
    if receipt.get("cache_root") != str(destination.root):
        raise CachePromotionError("Cache-promotion receipt destination root differs")
    completed_at = receipt.get("completed_at")
    if not isinstance(completed_at, str):
        raise CachePromotionError("Cache-promotion receipt completion time is invalid")
    try:
        completed = datetime.fromisoformat(completed_at)
    except ValueError as exc:
        raise CachePromotionError("Cache-promotion receipt completion time is invalid") from exc
    if (
        completed.tzinfo is None
        or completed.utcoffset() != timedelta(0)
        or completed_at != completed.isoformat()
    ):
        raise CachePromotionError("Cache-promotion receipt completion time is not canonical UTC")
    index_sha256 = receipt.get("index_sha256")
    if not isinstance(index_sha256, str) or _FINGERPRINT.fullmatch(index_sha256) is None:
        raise CachePromotionError("Cache-promotion receipt index digest is invalid")
    _, index_snapshot = _read_regular_file(destination.index_path, capture_bytes=False)
    if index_snapshot.sha256 != index_sha256:
        raise CachePromotionError("Cache-promotion receipt index digest differs from live index")
    expected_fingerprints = [str(entry["fingerprint"]) for entry in entries]
    receipt_items = receipt.get("items")
    if not isinstance(receipt_items, list) or len(receipt_items) != len(entries):
        raise CachePromotionError("Cache-promotion receipt entries differ")
    if any(not isinstance(item, Mapping) for item in receipt_items):
        raise CachePromotionError("Cache-promotion receipt item is invalid")
    if [item.get("fingerprint") for item in receipt_items] != expected_fingerprints:
        raise CachePromotionError("Cache-promotion receipt entries differ")
    for raw_item, entry in zip(receipt_items, entries, strict=True):
        disposition = raw_item.get("disposition")
        if disposition not in {"promoted", "reused"}:
            raise CachePromotionError("Cache-promotion receipt disposition is invalid")
        expected_item = _report_items([entry], moved_paths=set())[0].to_dict()
        expected_item["disposition"] = disposition
        if dict(raw_item) != expected_item:
            raise CachePromotionError("Cache-promotion receipt evidence differs")
    for entry in entries:
        _verify_final_entry(entry, destination=destination)
    items = _report_items(entries, moved_paths=set())
    return CachePromotionReport(
        transaction_id=receipt_path.parent.name,
        staging_root=staging.root,
        destination_root=destination.root,
        items=items,
        index_sha256=index_sha256,
        receipt_path=receipt_path,
        receipt_sha256=_sha256_bytes(receipt_bytes),
    )


def promote_cache_entries(
    *,
    staging_dir: Path,
    cache_dir: Path,
    entries: Sequence[CacheEntryExpectation] | None,
    transaction_id: str,
) -> CachePromotionReport:
    """Promote verified cache entries without overwriting scientific artifacts.

    The operation is recoverable by rerunning the exact same request.  Source
    records are moved first, then processed directories and content manifests;
    main manifests are published last.  The derived SQLite index is replaced
    atomically and an immutable receipt is written only after live rehashing.
    ``entries=None`` selects every canonical staging manifest for a new
    transaction, or recovers the exact request from an existing intent.
    """

    if _TRANSACTION_ID.fullmatch(transaction_id) is None or transaction_id in {".", ".."}:
        raise CachePromotionError(f"Invalid cache-promotion transaction id: {transaction_id!r}")
    staging_root = _canonical_root(Path(staging_dir), label="staging cache root")
    destination_root = _canonical_root(Path(cache_dir), label="destination cache root")
    if staging_root == destination_root:
        raise CachePromotionError("Staging and destination cache roots must differ")
    if os.lstat(staging_root).st_dev != os.lstat(destination_root).st_dev:
        raise CachePromotionError("Staging and destination cache roots must share a filesystem")

    staging = CacheLayout(staging_root)
    destination = CacheLayout(destination_root)
    transaction_root = destination.root / ".transactions" / transaction_id
    publish_root = transaction_root / "publish"
    intent_path = transaction_root / "intent.json"
    receipt_path = transaction_root / "receipt.json"
    request = (
        _all_staged_request(
            staging=staging,
            destination=destination,
            transaction_id=transaction_id,
            intent_path=intent_path,
            transaction_root=transaction_root,
        )
        if entries is None
        else _expectation_payload(entries)
    )

    lock_paths = sorted(
        [
            *(staging.lock_path(str(item["fingerprint"])) for item in request),
            *(destination.lock_path(str(item["fingerprint"])) for item in request),
        ],
        key=lambda path: str(path),
    )
    _mkdir_safe(staging.root, staging.locks_root)
    _mkdir_safe(destination.root, destination.locks_root)
    with ExitStack() as stack:
        for path in lock_paths:
            stack.enter_context(_file_lock(path))

        if os.path.lexists(intent_path):
            transaction_relative = _relative_path(
                _relative_to(transaction_root, destination.root, purpose="transaction directory"),
                purpose="transaction directory",
            )
            _assert_no_symlink_chain(
                destination.root,
                transaction_relative,
                allow_missing=False,
            )
            intent, intent_bytes = _read_json_record(intent_path, label="promotion intent")
        else:
            intent = _build_intent(
                staging=staging,
                destination=destination,
                transaction_id=transaction_id,
                request=request,
            )
            intent_bytes = _canonical_json(intent) + b"\n"
            _mkdir_safe(destination.root, transaction_root)
            try:
                _write_exclusive(intent_path, intent_bytes)
            except FileExistsError:
                intent, intent_bytes = _read_json_record(intent_path, label="promotion intent")
        planned_entries = _validate_intent(
            intent,
            staging=staging,
            destination=destination,
            transaction_id=transaction_id,
            request=request,
        )
        intent_sha256 = _sha256_bytes(intent_bytes)

        if os.path.lexists(receipt_path):
            receipt, receipt_bytes = _read_json_record(receipt_path, label="promotion receipt")
            with index_lock(destination):
                return _report_from_receipt(
                    receipt,
                    receipt_bytes,
                    receipt_path=receipt_path,
                    staging=staging,
                    destination=destination,
                    entries=planned_entries,
                    intent_sha256=intent_sha256,
                )

        moved_paths: set[str] = set()
        sources: dict[str, Mapping[str, Any]] = {}
        for entry in planned_entries:
            for source in entry["source_files"]:
                destination_value = str(source["destination"])
                previous = sources.setdefault(destination_value, source)
                if previous != source:
                    raise CachePromotionError(
                        f"Promotion intent contains conflicting sources: {destination_value}"
                    )
        for destination_value, source in sorted(sources.items()):
            if _promote_file(
                source,
                staging_root=staging.root,
                destination_root=destination.root,
                publish_root=publish_root,
            ):
                moved_paths.add(destination_value)
        _promotion_checkpoint("source-records")

        for entry in planned_entries:
            if _promote_directory(
                entry["processed"],
                staging_root=staging.root,
                destination_root=destination.root,
                publish_root=publish_root,
            ):
                moved_paths.add(str(entry["processed"]["destination"]))
        _promotion_checkpoint("processed")

        for entry in planned_entries:
            if _promote_file(
                entry["content_manifest"],
                staging_root=staging.root,
                destination_root=destination.root,
                publish_root=publish_root,
            ):
                moved_paths.add(str(entry["content_manifest"]["path"]))
        _promotion_checkpoint("content-manifests")

        for entry in planned_entries:
            if _promote_file(
                entry["main_manifest"],
                staging_root=staging.root,
                destination_root=destination.root,
                publish_root=publish_root,
            ):
                moved_paths.add(str(entry["main_manifest"]["path"]))
        _promotion_checkpoint("main-manifests")

        for entry in planned_entries:
            _verify_final_entry(entry, destination=destination)
        _promotion_checkpoint("live-rehash")

        with index_lock(destination):
            index_sha256 = _rebuild_index_atomic_locked(destination, strict=True)
            _promotion_checkpoint("index")
            items = _report_items(planned_entries, moved_paths=moved_paths)
            receipt = _receipt_payload(
                transaction_id=transaction_id,
                staging=staging,
                destination=destination,
                intent_sha256=intent_sha256,
                items=items,
                index_sha256=index_sha256,
            )
            receipt_bytes = _canonical_json(receipt) + b"\n"
            try:
                _write_exclusive(receipt_path, receipt_bytes)
            except FileExistsError:
                existing, existing_bytes = _read_json_record(
                    receipt_path,
                    label="promotion receipt",
                )
                return _report_from_receipt(
                    existing,
                    existing_bytes,
                    receipt_path=receipt_path,
                    staging=staging,
                    destination=destination,
                    entries=planned_entries,
                    intent_sha256=intent_sha256,
                )
            _promotion_checkpoint("receipt")
            return CachePromotionReport(
                transaction_id=transaction_id,
                staging_root=staging.root,
                destination_root=destination.root,
                items=items,
                index_sha256=index_sha256,
                receipt_path=receipt_path,
                receipt_sha256=_sha256_bytes(receipt_bytes),
            )


__all__ = [
    "CacheEntryExpectation",
    "CachePromotionItem",
    "CachePromotionReport",
    "promote_cache_entries",
]
