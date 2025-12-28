from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Split:
    """A canonical dataset split.

    X and y are backend-agnostic containers (often numpy arrays).
    edges and masks are used for graph datasets.
    """

    X: Any
    y: Any
    edges: Any | None = None
    masks: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LoadedDataset:
    """Canonical dataset container.

    If the provider supplies official splits, test may be present.
    If not, test is None.

    This module does not create custom splits.
    """

    train: Split
    test: Split | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    """Curated dataset spec (catalog entry).

    The fingerprint used for caching intentionally ignores documentation-only fields.
    """

    key: str
    provider: str
    uri: str
    modality: str
    task: str
    description: str

    required_extra: str | None = None
    source_kwargs: Mapping[str, Any] = field(default_factory=dict)

    homepage: str | None = None
    license: str | None = None
    citation: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "provider": self.provider,
            "uri": self.uri,
            "modality": self.modality,
            "task": self.task,
            "description": self.description,
            "required_extra": self.required_extra,
            "source_kwargs": dict(self.source_kwargs),
            "homepage": self.homepage,
            "license": self.license,
            "citation": self.citation,
        }

    def fingerprint_payload(self, *, schema_version: int) -> dict[str, Any]:
        """Payload used to compute the cache fingerprint.

        Only fields that can change dataset bytes are included.
        """
        return {
            "schema_version": int(schema_version),
            "provider": self.provider,
            "uri": self.uri,
            "source_kwargs": dict(self.source_kwargs),
        }

    def fingerprint(self, *, schema_version: int) -> str:
        payload = self.fingerprint_payload(schema_version=schema_version)
        try:
            blob = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        except TypeError as e:
            raise ValueError("DatasetSpec.source_kwargs must be JSON serializable.") from e
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class DatasetRequest:
    """A dataset request.

    - id can be a curated key or a provider URI
    - options can override or extend catalog source_kwargs
    """

    id: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetIdentity:
    """Resolved dataset identity (provider level)."""

    provider: str
    canonical_uri: str
    dataset_id: str
    version: str | None
    modality: str
    task: str
    required_extra: str | None = None
    resolved_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "canonical_uri": self.canonical_uri,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "modality": self.modality,
            "task": self.task,
            "required_extra": self.required_extra,
            "resolved_kwargs": dict(self.resolved_kwargs),
        }

    def fingerprint_payload(self, *, schema_version: int) -> dict[str, Any]:
        return {
            "schema_version": int(schema_version),
            "provider": self.provider,
            "canonical_uri": self.canonical_uri,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "modality": self.modality,
            "task": self.task,
            "resolved_kwargs": dict(self.resolved_kwargs),
        }

    def fingerprint(self, *, schema_version: int) -> str:
        payload = self.fingerprint_payload(schema_version=schema_version)
        try:
            blob = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        except TypeError as e:
            raise ValueError("DatasetIdentity.resolved_kwargs must be JSON serializable.") from e
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class DownloadReport:
    """Report returned by download_all_datasets."""

    downloaded: Sequence[str] = ()
    skipped_already_cached: Sequence[str] = ()
    skipped_missing_extras: Sequence[str] = ()
    missing_extras: Mapping[str, Sequence[str]] = field(default_factory=dict)
    failed: Mapping[str, str] = field(default_factory=dict)

    def has_failures(self) -> bool:
        return bool(self.failed)

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(f"Downloaded: {len(self.downloaded)}")
        lines.append(f"Skipped (already cached): {len(self.skipped_already_cached)}")
        lines.append(f"Skipped (missing extras): {len(self.skipped_missing_extras)}")
        lines.append(f"Failed: {len(self.failed)}")
        return "\n".join(lines)
