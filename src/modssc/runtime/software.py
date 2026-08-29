"""Selective software provenance for portable checkpoint identities.

The runtime records only distributions required by the materialized pipeline.
This avoids coupling a NumPy-only checkpoint to every optional package installed
on a host while still invalidating a checkpoint when a selected dependency
changes.  Callers may combine component-declared extras with explicit
distribution names for custom factories and plugins.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from types import MappingProxyType
from typing import Any

_MANIFEST_SCHEMA_VERSION = 1
_SOFTWARE_IDENTITY_SCHEMA_VERSION = 2
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_REQUIREMENT_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

# These values describe executable code, not the host on which that code runs.
# Accelerator and filesystem information deliberately remain outside resume
# compatibility so an otherwise identical checkpoint can move between hosts.
_CODE_PROVENANCE_FIELDS = (
    "python",
    "python_implementation",
    "modssc",
    "distribution_sha256",
    "git_sha",
    "git_dirty",
    "git_diff_sha256",
)

# Legacy callers created identity payloads before manifests existed.  Keeping a
# narrow compatibility adapter prevents old API clients from crashing; new
# collection always emits an explicit manifest and does not use this global set.
_LEGACY_DISTRIBUTION_FIELDS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "scikit_learn",
    "torch": "torch",
    "torch-geometric": "torch_geometric",
}


class SoftwareProvenanceError(ValueError):
    """Raised when required software provenance cannot be represented safely."""


def normalize_distribution_name(value: str) -> str:
    """Return the stable PEP 503 spelling used as a manifest key."""

    if not isinstance(value, str) or not value.strip():
        raise SoftwareProvenanceError("distribution names must be non-empty strings")
    name = value.strip()
    if _NAME_PATTERN.fullmatch(name) is None:
        raise SoftwareProvenanceError(f"invalid distribution name: {value!r}")
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_distribution_name(requirement: str) -> str:
    """Extract and normalize a distribution name from one requirement string."""

    if not isinstance(requirement, str):
        raise SoftwareProvenanceError("requirements must be strings")
    match = _REQUIREMENT_NAME_PATTERN.match(requirement)
    if match is None:
        raise SoftwareProvenanceError(f"invalid requirement: {requirement!r}")
    return normalize_distribution_name(match.group(1))


def resolve_required_distributions(
    *,
    extras: Iterable[str] = (),
    optional_dependencies: Mapping[str, Iterable[str]] | None = None,
    explicit: Iterable[str] = (),
    base: Iterable[str] = ("numpy", "scipy"),
) -> tuple[str, ...]:
    """Resolve selected extras and explicit declarations into one stable set.

    ``optional_dependencies`` is intentionally injected by the caller.  A wheel
    adapter can read installed metadata while a checkout adapter can read its
    current ``pyproject.toml``; the native selection semantics stay identical.
    Only the named extras are expanded, so unrelated installed extras never
    become checkpoint identity-bearing.
    """

    groups = optional_dependencies or {}
    normalized_groups = {str(name): tuple(values) for name, values in groups.items()}
    selected: set[str] = {
        normalize_distribution_name(distribution) for distribution in (*base, *explicit)
    }
    for extra in extras:
        if not isinstance(extra, str) or not extra.strip():
            raise SoftwareProvenanceError("extra names must be non-empty strings")
        name = extra.strip()
        if name not in normalized_groups:
            raise SoftwareProvenanceError(f"unknown optional dependency extra: {name!r}")
        selected.update(
            requirement_distribution_name(requirement) for requirement in normalized_groups[name]
        )
    return tuple(sorted(selected))


@dataclass(frozen=True)
class SoftwareManifest:
    """Exact version manifest for the distributions selected by one pipeline."""

    required_distributions: tuple[str, ...]
    versions: Mapping[str, str | None]

    def __post_init__(self) -> None:
        normalized = tuple(
            sorted(normalize_distribution_name(name) for name in self.required_distributions)
        )
        if len(set(normalized)) != len(normalized):
            raise SoftwareProvenanceError("required_distributions must not contain duplicates")

        normalized_versions: dict[str, str | None] = {}
        for raw_name, version in self.versions.items():
            name = normalize_distribution_name(raw_name)
            if name in normalized_versions:
                raise SoftwareProvenanceError(f"duplicate version entry for {name!r}")
            if version is not None and (not isinstance(version, str) or not version.strip()):
                raise SoftwareProvenanceError(
                    f"version for distribution {name!r} must be a non-empty string or null"
                )
            normalized_versions[name] = version.strip() if isinstance(version, str) else None
        if set(normalized_versions) != set(normalized):
            raise SoftwareProvenanceError(
                "manifest version keys must exactly match required_distributions"
            )

        object.__setattr__(self, "required_distributions", normalized)
        object.__setattr__(
            self,
            "versions",
            MappingProxyType(dict(sorted(normalized_versions.items()))),
        )

    @property
    def missing_versions(self) -> tuple[str, ...]:
        return tuple(name for name in self.required_distributions if self.versions[name] is None)

    def require_complete(self) -> None:
        """Fail closed when a checkpoint identity would contain unknown versions."""

        missing = self.missing_versions
        if missing:
            raise SoftwareProvenanceError(
                f"version metadata is unavailable for required distributions: {list(missing)!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "required_distributions": list(self.required_distributions),
            "versions": dict(self.versions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SoftwareManifest:
        if not isinstance(value, Mapping):
            raise SoftwareProvenanceError("software manifest must be a mapping")
        expected = {"schema_version", "required_distributions", "versions"}
        if set(value) != expected:
            raise SoftwareProvenanceError("software manifest fields differ from the runtime schema")
        if value.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
            raise SoftwareProvenanceError("unsupported software manifest schema_version")
        required = value.get("required_distributions")
        versions = value.get("versions")
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise SoftwareProvenanceError(
                "software manifest required_distributions must be a list of strings"
            )
        if not isinstance(versions, Mapping):
            raise SoftwareProvenanceError("software manifest versions must be a mapping")
        return cls(required_distributions=tuple(required), versions=dict(versions))


def _installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_software_manifest(
    required_distributions: Iterable[str],
    *,
    version_getter: Callable[[str], str | None] | None = None,
    require_complete: bool = False,
) -> SoftwareManifest:
    """Collect exact versions for a pre-resolved set of distributions."""

    names = tuple(sorted({normalize_distribution_name(name) for name in required_distributions}))
    get_version = version_getter or _installed_version
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = get_version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    manifest = SoftwareManifest(required_distributions=names, versions=versions)
    if require_complete:
        manifest.require_complete()
    return manifest


def attach_software_manifest(
    runtime_versions: Mapping[str, Any],
    *,
    required_distributions: Iterable[str],
    version_getter: Callable[[str], str | None] | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Return runtime reporting data with its selective native manifest."""

    if not isinstance(runtime_versions, Mapping):
        raise TypeError("runtime_versions must be a mapping")
    manifest = collect_software_manifest(
        required_distributions,
        version_getter=version_getter,
        require_complete=require_complete,
    )
    out = dict(runtime_versions)
    out["software_manifest"] = manifest.to_dict()
    return out


def _legacy_manifest(runtime_versions: Mapping[str, Any]) -> SoftwareManifest:
    versions = {
        distribution: runtime_versions.get(field)
        for distribution, field in _LEGACY_DISTRIBUTION_FIELDS.items()
        if field in runtime_versions
    }
    return SoftwareManifest(
        required_distributions=tuple(versions),
        versions=versions,
    )


def software_identity_payload(runtime_versions: Mapping[str, Any]) -> dict[str, Any]:
    """Select host-independent code and declared dependency provenance."""

    if not isinstance(runtime_versions, Mapping):
        raise TypeError("runtime_versions must be a mapping")
    manifest_value = runtime_versions.get("software_manifest")
    manifest = (
        _legacy_manifest(runtime_versions)
        if manifest_value is None
        else SoftwareManifest.from_dict(manifest_value)
    )
    return {
        **{field: runtime_versions.get(field) for field in _CODE_PROVENANCE_FIELDS},
        "software_manifest": manifest.to_dict(),
    }


def software_sha256(runtime_versions: Mapping[str, Any]) -> str:
    """Hash the native software payload used for checkpoint compatibility."""

    payload = {
        "schema_version": _SOFTWARE_IDENTITY_SCHEMA_VERSION,
        "software": software_identity_payload(runtime_versions),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SoftwareManifest",
    "SoftwareProvenanceError",
    "attach_software_manifest",
    "collect_software_manifest",
    "normalize_distribution_name",
    "requirement_distribution_name",
    "resolve_required_distributions",
    "software_identity_payload",
    "software_sha256",
]
