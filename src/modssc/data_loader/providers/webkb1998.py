from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from modssc.data_loader.errors import DataLoaderError
from modssc.data_loader.providers.base import BaseProvider
from modssc.data_loader.types import DatasetIdentity, LoadedDataset, Split
from modssc.data_loader.uri import ParsedURI

SOURCE_URL = (
    "https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-51/www/"
    "co-training/data/course-cotrain-data.tar.gz"
)
ARCHIVE_SHA256 = "1f0d9d7f55c90754e7581272d67613e7c358323e44e2b84c438bed063653e7db"
ARCHIVE_FILENAME = "course-cotrain-data.tar.gz"
DATASET_REFERENCE = "course"
DATASET_VERSION = "1998-course-v1"
EXPECTED_PAIRS = 1051
EXPECTED_CLASS_COUNTS = {"course": 230, "non-course": 821}
LABEL_TO_ID = {"non-course": 0, "course": 1}
VIEW_NAMES = ("fulltext", "inlinks")
ARCHIVE_ROOT = "course-cotrain-data"

_MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024
_MAX_TAR_MEMBERS = 4096
_MAX_MEMBER_BYTES = 1024 * 1024
_MAX_TOTAL_MEMBER_BYTES = 16 * 1024 * 1024
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_DOWNLOAD_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_archive(path: Path, *, expected_sha256: str) -> None:
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise DataLoaderError(
            f"WebKB 1998 archive SHA-256 mismatch: expected {expected_sha256}, observed {observed}."
        )


def _download_archive(url: str, destination: Path, *, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file():
            raise DataLoaderError(f"WebKB 1998 archive path is not a file: {destination}")
        _verify_archive(destination, expected_sha256=expected_sha256)
        return destination

    request = urllib.request.Request(url, headers={"User-Agent": "ModSSC/1 webkb1998"})
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{ARCHIVE_FILENAME}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            hasher = hashlib.sha256()
            downloaded = 0
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > _MAX_DOWNLOAD_BYTES:
                        raise DataLoaderError(
                            "WebKB 1998 archive exceeds the maximum allowed download size."
                        )
                    output.write(chunk)
                    hasher.update(chunk)

        observed = hasher.hexdigest()
        if observed != expected_sha256:
            raise DataLoaderError(
                "WebKB 1998 downloaded archive SHA-256 mismatch: "
                f"expected {expected_sha256}, observed {observed}."
            )
        os.replace(temp_path, destination)
        return destination
    except DataLoaderError:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError) as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise DataLoaderError(f"Unable to download the WebKB 1998 archive from {url}.") from exc


def _member_parts(member: tarfile.TarInfo) -> tuple[str, ...]:
    raw_name = member.name.rstrip("/")
    parts = tuple(raw_name.split("/"))
    if (
        not raw_name
        or raw_name.startswith("/")
        or "\\" in raw_name
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise DataLoaderError(f"Unsafe path in WebKB 1998 archive: {member.name!r}")
    return parts


def _validate_member(member: tarfile.TarInfo) -> tuple[str, ...] | None:
    parts = _member_parts(member)
    allowed_directories = {
        (ARCHIVE_ROOT,),
        *((ARCHIVE_ROOT, view) for view in VIEW_NAMES),
        *((ARCHIVE_ROOT, view, class_name) for view in VIEW_NAMES for class_name in LABEL_TO_ID),
    }
    if member.isdir():
        if parts not in allowed_directories:
            raise DataLoaderError(f"Unexpected directory in WebKB 1998 archive: {member.name!r}")
        return None
    if not member.isfile():
        raise DataLoaderError(
            f"Unsupported non-regular member in WebKB 1998 archive: {member.name!r}"
        )
    if (
        len(parts) != 4
        or parts[0] != ARCHIVE_ROOT
        or parts[1] not in VIEW_NAMES
        or parts[2] not in LABEL_TO_ID
        or not parts[3]
    ):
        raise DataLoaderError(f"Unexpected file in WebKB 1998 archive: {member.name!r}")
    if member.size < 0 or member.size > _MAX_MEMBER_BYTES:
        raise DataLoaderError(f"Invalid member size in WebKB 1998 archive: {member.name!r}")
    return parts


def _canonical_digest(records: list[dict[str, str]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def _alignment_error(
    fulltext_keys: set[tuple[str, str]], inlink_keys: set[tuple[str, str]]
) -> DataLoaderError:
    missing_inlinks = sorted(fulltext_keys - inlink_keys)[:3]
    missing_fulltext = sorted(inlink_keys - fulltext_keys)[:3]
    return DataLoaderError(
        "WebKB 1998 views are not exactly aligned: "
        f"missing_inlinks={missing_inlinks}, missing_fulltext={missing_fulltext}."
    )


def _read_webkb_archive(
    archive_path: Path,
    *,
    expected_pairs: int,
    expected_class_counts: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    by_view: dict[str, dict[tuple[str, str], tuple[str, str]]] = {view: {} for view in VIEW_NAMES}
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > _MAX_TAR_MEMBERS:
                raise DataLoaderError("WebKB 1998 archive contains too many members.")

            regular_members: list[tuple[tarfile.TarInfo, tuple[str, ...]]] = []
            total_size = 0
            for member in members:
                parts = _validate_member(member)
                if parts is None:
                    continue
                total_size += int(member.size)
                if total_size > _MAX_TOTAL_MEMBER_BYTES:
                    raise DataLoaderError(
                        "WebKB 1998 archive exceeds the maximum uncompressed size."
                    )
                regular_members.append((member, parts))

            for member, parts in regular_members:
                view, class_name, sample_id = parts[1], parts[2], parts[3]
                key = (class_name, sample_id)
                if key in by_view[view]:
                    raise DataLoaderError(
                        f"Duplicate WebKB 1998 member for {view}/{class_name}/{sample_id}."
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise DataLoaderError(f"Unable to read WebKB 1998 member: {member.name!r}")
                payload = source.read(member.size + 1)
                if len(payload) != member.size:
                    raise DataLoaderError(f"Truncated WebKB 1998 member: {member.name!r}")
                by_view[view][key] = (
                    payload.decode("latin-1"),
                    hashlib.sha256(payload).hexdigest(),
                )
    except (tarfile.TarError, OSError) as exc:
        raise DataLoaderError(f"Invalid WebKB 1998 tar archive: {archive_path}") from exc

    fulltext_keys = set(by_view["fulltext"])
    inlink_keys = set(by_view["inlinks"])
    if fulltext_keys != inlink_keys:
        raise _alignment_error(fulltext_keys, inlink_keys)
    if len(fulltext_keys) != int(expected_pairs):
        raise DataLoaderError(
            "Unexpected number of WebKB 1998 pairs: "
            f"expected {expected_pairs}, observed {len(fulltext_keys)}."
        )

    observed_counts = {
        class_name: sum(key[0] == class_name for key in fulltext_keys) for class_name in LABEL_TO_ID
    }
    normalized_expected_counts = {
        class_name: int(expected_class_counts[class_name]) for class_name in LABEL_TO_ID
    }
    if observed_counts != normalized_expected_counts:
        raise DataLoaderError(
            "Unexpected WebKB 1998 class counts: "
            f"expected {normalized_expected_counts}, observed {observed_counts}."
        )

    ordered_keys = sorted(fulltext_keys, key=lambda key: (LABEL_TO_ID[key[0]], key[1]))
    X = np.empty((len(ordered_keys), 2), dtype=object)
    y = np.empty((len(ordered_keys),), dtype=np.int64)
    fulltext_records: list[dict[str, str]] = []
    inlink_records: list[dict[str, str]] = []
    pair_records: list[dict[str, str]] = []
    sample_ids: list[str] = []

    for row, (class_name, sample_id) in enumerate(ordered_keys):
        fulltext, fulltext_sha256 = by_view["fulltext"][(class_name, sample_id)]
        inlinks, inlinks_sha256 = by_view["inlinks"][(class_name, sample_id)]
        X[row, 0] = fulltext
        X[row, 1] = inlinks
        y[row] = LABEL_TO_ID[class_name]
        row_id = f"{class_name}/{sample_id}"
        sample_ids.append(row_id)
        fulltext_records.append({"id": row_id, "sha256": fulltext_sha256})
        inlink_records.append({"id": row_id, "sha256": inlinks_sha256})
        pair_records.append(
            {
                "id": row_id,
                "fulltext_sha256": fulltext_sha256,
                "inlinks_sha256": inlinks_sha256,
            }
        )

    hashes = {
        "fulltext_sha256": _canonical_digest(fulltext_records),
        "inlinks_sha256": _canonical_digest(inlink_records),
        "pair_manifest_sha256": _canonical_digest(pair_records),
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
    }
    parsed_meta: dict[str, Any] = {
        "n_samples": len(ordered_keys),
        "class_counts": observed_counts,
        "label_mapping": dict(LABEL_TO_ID),
        "view_names": list(VIEW_NAMES),
        "row_order": "label_id_then_source_id",
        "sample_ids": sample_ids,
        "text_encoding": "latin-1-byte-preserving",
        **hashes,
    }
    return X, y, parsed_meta


class WebKB1998Provider(BaseProvider):
    """Strict stdlib loader for the WebKB subset used by Blum and Mitchell (1998)."""

    name = "webkb1998"
    required_extra = None

    def resolve(self, parsed: ParsedURI, *, options: Mapping[str, Any]) -> DatasetIdentity:
        reference = parsed.reference.strip().lower()
        if reference != DATASET_REFERENCE:
            raise DataLoaderError(
                f"Unknown WebKB 1998 dataset reference: {parsed.reference!r}. "
                f"Expected {DATASET_REFERENCE!r}."
            )
        if options:
            raise DataLoaderError(
                "WebKB 1998 does not accept source overrides; its URL and SHA-256 are pinned."
            )
        return DatasetIdentity(
            provider=self.name,
            canonical_uri=f"{self.name}:{DATASET_REFERENCE}",
            dataset_id=DATASET_REFERENCE,
            version=DATASET_VERSION,
            modality="text",
            task="classification",
            required_extra=None,
            resolved_kwargs={
                "source_url": SOURCE_URL,
                "archive_sha256": ARCHIVE_SHA256,
                "archive_filename": ARCHIVE_FILENAME,
                "expected_pairs": EXPECTED_PAIRS,
                "expected_class_counts": dict(EXPECTED_CLASS_COUNTS),
                "text_encoding": "latin-1-byte-preserving",
            },
        )

    def load_canonical(self, identity: DatasetIdentity, *, raw_dir: Path) -> LoadedDataset:
        cfg = dict(identity.resolved_kwargs)
        canonical = (
            identity.provider == self.name
            and identity.dataset_id == DATASET_REFERENCE
            and cfg.get("source_url") == SOURCE_URL
            and cfg.get("archive_sha256") == ARCHIVE_SHA256
            and cfg.get("archive_filename") == ARCHIVE_FILENAME
        )
        if not canonical:
            raise DataLoaderError("WebKB 1998 identity does not match the pinned source artifact.")

        archive_path = _download_archive(
            SOURCE_URL,
            raw_dir / ARCHIVE_FILENAME,
            expected_sha256=ARCHIVE_SHA256,
        )
        X, y, parsed_meta = _read_webkb_archive(
            archive_path,
            expected_pairs=EXPECTED_PAIRS,
            expected_class_counts=EXPECTED_CLASS_COUNTS,
        )
        meta = {
            "provider": self.name,
            "dataset_id": DATASET_REFERENCE,
            "version": DATASET_VERSION,
            "source_url": SOURCE_URL,
            "archive_filename": ARCHIVE_FILENAME,
            "archive_sha256": ARCHIVE_SHA256,
            "archive_size_bytes": int(archive_path.stat().st_size),
            "official_split": False,
            "license": None,
            **parsed_meta,
        }
        return LoadedDataset(train=Split(X=X, y=y), test=None, meta=meta)
