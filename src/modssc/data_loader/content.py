from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_loader.errors import ManifestError
from modssc.data_loader.types import LoadedDataset

CONTENT_MANIFEST_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _content_sha256(payload: Mapping[str, Any]) -> str:
    files = payload.get("files", [])
    semantic = {
        "schema_version": payload.get("schema_version"),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "files": [
            {
                "kind": record["kind"],
                "path": record["path"],
                "sha256": record["sha256"],
            }
            for record in files
        ],
    }
    return hashlib.sha256(_canonical_json(semantic)).hexdigest()


def _path_strings(dataset: LoadedDataset) -> Iterable[str]:
    for split in (dataset.train, dataset.test):
        if split is None:
            continue
        try:
            values = np.asarray(split.X)
        except Exception:
            continue
        if values.dtype.kind not in {"O", "S", "U"}:
            continue
        for value in values.flat:
            if isinstance(value, str):
                yield value


def _source_root(layout: Any, identity: Mapping[str, Any]) -> Path | None:
    if identity.get("provider") != "torchaudio":
        return None
    dataset_id = identity.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ManifestError("Torchaudio content manifest has no dataset_id")
    version = identity.get("version")
    return layout.raw_dir("torchaudio", dataset_id, version) / "source"


def _resolve_source_path(value: str, *, source_root: Path) -> tuple[str, Path]:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = source_root / candidate
    try:
        candidate.resolve(strict=True).relative_to(source_root.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        pass
    else:
        relative = candidate.resolve(strict=True).relative_to(source_root.resolve(strict=True))
        return relative.as_posix(), candidate
    if "source" in candidate.parts:
        source_index = max(index for index, part in enumerate(candidate.parts) if part == "source")
        candidate = source_root / Path(*candidate.parts[source_index + 1 :])
    try:
        relative = candidate.resolve(strict=True).relative_to(source_root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ManifestError(
            f"Referenced dataset source file is missing or outside the cache: {value}"
        ) from exc
    if not candidate.is_file():
        raise ManifestError(f"Referenced dataset source is not a file: {value}")
    return relative.as_posix(), candidate


def _jsonl_values(path: Path) -> list[Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _processed_record(
    path: Path,
    *,
    relative_path: str,
    source_root: Path | None,
) -> dict[str, Any]:
    digest_mode = "bytes"
    semantic_digest = _sha256_file(path)
    if path.name.endswith(".jsonl.gz"):
        values = _jsonl_values(path)
        digest_mode = "json-values"
        if source_root is not None and path.name in {"train_X.jsonl.gz", "test_X.jsonl.gz"}:
            values = [
                _resolve_source_path(str(value), source_root=source_root)[0] for value in values
            ]
            digest_mode = "audio-paths"
        semantic_digest = hashlib.sha256(_canonical_json(values)).hexdigest()
    return {
        "kind": "processed",
        "path": relative_path,
        "size_bytes": int(path.stat().st_size),
        "sha256": semantic_digest,
        "storage_sha256": _sha256_file(path),
        "digest_mode": digest_mode,
    }


def _file_record(*, kind: str, relative_path: str, path: Path) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": relative_path,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def build_content_manifest(
    layout: Any,
    fingerprint: str,
    dataset: LoadedDataset,
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash the processed cache and any source files referenced by it.

    Absolute cache paths are deliberately excluded from the digest so a cache can
    be moved between machines without changing its scientific identity.
    """

    processed_dir = layout.processed_dir(fingerprint)
    if not processed_dir.is_dir():
        raise ManifestError(f"Missing processed dataset cache: {processed_dir}")
    source_root = _source_root(layout, identity)
    records = [
        _processed_record(
            path,
            relative_path=path.relative_to(processed_dir).as_posix(),
            source_root=source_root,
        )
        for path in sorted(processed_dir.rglob("*"))
        if path.is_file()
    ]
    if source_root is not None:
        sources: dict[str, Path] = {}
        for value in _path_strings(dataset):
            relative, path = _resolve_source_path(value, source_root=source_root)
            sources[relative] = path
        records.extend(
            _file_record(kind="source", relative_path=relative, path=path)
            for relative, path in sorted(sources.items())
        )

    records.sort(key=lambda record: (str(record["kind"]), str(record["path"])))
    payload = {
        "schema_version": CONTENT_MANIFEST_SCHEMA_VERSION,
        "dataset_fingerprint": fingerprint,
        "files": records,
    }
    return {**payload, "content_sha256": _content_sha256(payload)}


def content_manifest_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n"


def read_content_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Invalid dataset content manifest: {path}") from exc
    if not isinstance(raw, dict):
        raise ManifestError(f"Dataset content manifest root must be a mapping: {path}")
    return raw


def _validate_manifest(manifest: Mapping[str, Any], *, fingerprint: str) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != CONTENT_MANIFEST_SCHEMA_VERSION:
        raise ManifestError("Unsupported dataset content manifest schema")
    if manifest.get("dataset_fingerprint") != fingerprint:
        raise ManifestError("Dataset content manifest fingerprint differs")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ManifestError("Dataset content manifest file table is empty")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for record in files:
        if not isinstance(record, Mapping):
            raise ManifestError("Invalid dataset content manifest file record")
        kind = record.get("kind")
        relative = record.get("path")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        storage_digest = record.get("storage_sha256")
        digest_mode = record.get("digest_mode")
        if kind not in {"processed", "source"}:
            raise ManifestError("Invalid dataset content manifest file kind")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ManifestError("Invalid dataset content manifest relative path")
        if ".." in Path(relative).parts:
            raise ManifestError("Dataset content manifest path escapes its cache root")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ManifestError("Invalid dataset content manifest file size")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ManifestError("Invalid dataset content manifest file digest")
        if kind == "processed":
            if not isinstance(storage_digest, str) or len(storage_digest) != 64:
                raise ManifestError("Invalid processed storage digest")
            if digest_mode not in {"bytes", "json-values", "audio-paths"}:
                raise ManifestError("Invalid processed dataset digest mode")
        identity = (kind, relative)
        if identity in identities:
            raise ManifestError("Duplicate dataset content manifest file record")
        identities.add(identity)
        normalized.append(dict(record))
    normalized.sort(key=lambda record: (str(record["kind"]), str(record["path"])))
    payload = {
        "schema_version": CONTENT_MANIFEST_SCHEMA_VERSION,
        "dataset_fingerprint": fingerprint,
        "files": normalized,
    }
    if manifest.get("content_sha256") != _content_sha256(payload):
        raise ManifestError("Dataset content manifest semantic digest differs")
    return normalized


def _record_path(
    record: Mapping[str, Any], *, layout: Any, fingerprint: str, identity: Mapping[str, Any]
) -> Path:
    relative = Path(str(record["path"]))
    if record["kind"] == "processed":
        root = layout.processed_dir(fingerprint)
    else:
        root = _source_root(layout, identity)
        if root is None:
            raise ManifestError("Source record is invalid for a non-path dataset")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ManifestError("Dataset content manifest path escapes its cache root") from exc
    return path


def verify_content_manifest(
    layout: Any,
    fingerprint: str,
    *,
    identity: Mapping[str, Any],
    rehash: bool,
) -> dict[str, str]:
    """Validate a content manifest and return a cheap cache-state attestation.

    ``rehash=True`` is intended for preflight. Individual tasks can use the
    size/mtime state digest and compare it with that preflight attestation,
    avoiding a complete rehash for every seed while still detecting ordinary
    post-preflight cache changes.
    """

    manifest_path = layout.content_manifest_path(fingerprint)
    manifest = read_content_manifest(manifest_path)
    records = _validate_manifest(manifest, fingerprint=fingerprint)
    state_records: list[dict[str, Any]] = []
    for record in records:
        path = _record_path(record, layout=layout, fingerprint=fingerprint, identity=identity)
        try:
            stat = path.stat()
        except OSError as exc:
            raise ManifestError(f"Dataset content file is missing: {path}") from exc
        if not path.is_file() or stat.st_size != record["size_bytes"]:
            raise ManifestError(f"Dataset content file size differs: {path}")
        if rehash:
            if record["kind"] == "processed":
                current = _processed_record(
                    path,
                    relative_path=str(record["path"]),
                    source_root=_source_root(layout, identity),
                )
                if current["sha256"] != record["sha256"]:
                    raise ManifestError(f"Dataset content file digest differs: {path}")
                if current["storage_sha256"] != record["storage_sha256"]:
                    raise ManifestError(f"Dataset storage file digest differs: {path}")
            elif _sha256_file(path) != record["sha256"]:
                raise ManifestError(f"Dataset content file digest differs: {path}")
        state_records.append(
            {
                "kind": record["kind"],
                "path": record["path"],
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return {
        "content_sha256": str(manifest["content_sha256"]),
        "content_manifest_sha256": _sha256_file(manifest_path),
        "cache_state_sha256": hashlib.sha256(_canonical_json(state_records)).hexdigest(),
        "cache_fingerprint": fingerprint,
    }


__all__ = [
    "CONTENT_MANIFEST_SCHEMA_VERSION",
    "build_content_manifest",
    "content_manifest_json",
    "read_content_manifest",
    "verify_content_manifest",
]
