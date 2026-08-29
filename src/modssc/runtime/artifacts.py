"""Portable contracts and runtime attestations for external artifacts.

The declarative contract contains only a path relative to a caller-provided
root, an artifact kind, and a SHA-256 digest. Runtime paths and filesystem
metadata are kept in the attestation produced by :func:`verify_artifact`.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat as stat_module
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

ArtifactKind = Literal["file", "tree"]
ArtifactPathKind = Literal["file", "directory", "symlink"]

_CONTRACT_SCHEMA_VERSION = 1
_ATTESTATION_SCHEMA_VERSION = 1
_TREE_DIGEST_SCHEMA_VERSION = 1
_ARTIFACT_KINDS = frozenset({"file", "tree"})
_PATH_KINDS = frozenset({"file", "directory", "symlink"})


class ArtifactContractError(ValueError):
    """Raised when a declarative artifact contract is malformed."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact is missing, changed, or has the wrong digest."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ArtifactContractError(f"{field} must be a non-empty portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ArtifactContractError(f"{field} must not be absolute or escape its root")
    return path.as_posix()


def _integer(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactContractError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ArtifactContractError(f"{field} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class ArtifactContract:
    """Portable expected identity for one file or directory tree."""

    path: str
    sha256: str
    kind: ArtifactKind = "file"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, field="artifact path"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, field="artifact sha256"))
        if not isinstance(self.kind, str) or self.kind not in _ARTIFACT_KINDS:
            raise ArtifactContractError("artifact kind must be 'file' or 'tree'")
        object.__setattr__(self, "kind", cast(ArtifactKind, self.kind))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _CONTRACT_SCHEMA_VERSION,
            "path": self.path,
            "kind": self.kind,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactContract:
        if not isinstance(value, Mapping):
            raise ArtifactContractError("artifact contract must be a mapping")
        expected_fields = {"schema_version", "path", "kind", "sha256"}
        if set(value) != expected_fields:
            raise ArtifactContractError("artifact contract fields differ from the runtime schema")
        if value.get("schema_version") != _CONTRACT_SCHEMA_VERSION:
            raise ArtifactContractError("unsupported artifact contract schema_version")
        return cls(
            path=value["path"],
            kind=value["kind"],
            sha256=value["sha256"],
        )


@dataclass(frozen=True)
class ArtifactPathState:
    """Revalidable state for one path inside a verified artifact."""

    path: str
    kind: ArtifactPathKind
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    mode: int
    content_sha256: str | None = None
    link_target: str | None = None
    link_mtime_ns: int | None = None
    link_ctime_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, field="artifact state path"))
        if not isinstance(self.kind, str) or self.kind not in _PATH_KINDS:
            raise ArtifactContractError("artifact state kind is invalid")
        object.__setattr__(self, "kind", cast(ArtifactPathKind, self.kind))
        for field, minimum in (
            ("size_bytes", 0),
            ("mtime_ns", None),
            ("ctime_ns", None),
            ("mode", 0),
        ):
            object.__setattr__(
                self,
                field,
                _integer(getattr(self, field), field=f"artifact state {field}", minimum=minimum),
            )

        if self.kind == "directory":
            if self.content_sha256 is not None or self.link_target is not None:
                raise ArtifactContractError("directory state must not contain file identity")
            if self.link_mtime_ns is not None or self.link_ctime_ns is not None:
                raise ArtifactContractError("directory state must not contain link metadata")
            return

        object.__setattr__(
            self,
            "content_sha256",
            _sha256(self.content_sha256, field="artifact state content_sha256"),
        )
        if self.kind == "file":
            if self.link_target is not None:
                raise ArtifactContractError("regular file state must not contain a link target")
            if self.link_mtime_ns is not None or self.link_ctime_ns is not None:
                raise ArtifactContractError("regular file state must not contain link metadata")
            return

        if (
            not isinstance(self.link_target, str)
            or not self.link_target
            or PurePosixPath(self.link_target).is_absolute()
        ):
            raise ArtifactContractError("symlink state requires a relative link target")
        for field in ("link_mtime_ns", "link_ctime_ns"):
            object.__setattr__(
                self,
                field,
                _integer(getattr(self, field), field=f"artifact state {field}"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "mode": self.mode,
            "content_sha256": self.content_sha256,
            "link_target": self.link_target,
            "link_mtime_ns": self.link_mtime_ns,
            "link_ctime_ns": self.link_ctime_ns,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactPathState:
        if not isinstance(value, Mapping):
            raise ArtifactContractError("artifact path state must be a mapping")
        expected_fields = {
            "path",
            "kind",
            "size_bytes",
            "mtime_ns",
            "ctime_ns",
            "mode",
            "content_sha256",
            "link_target",
            "link_mtime_ns",
            "link_ctime_ns",
        }
        if set(value) != expected_fields:
            raise ArtifactContractError("artifact path state fields differ from the schema")
        return cls(**{field: value[field] for field in expected_fields})


@dataclass(frozen=True)
class ArtifactAttestation:
    """Full-hash preflight result plus the observed filesystem state."""

    contract: ArtifactContract
    observed_sha256: str
    paths: tuple[ArtifactPathState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.contract, ArtifactContract):
            raise ArtifactContractError("attestation contract must be an ArtifactContract")
        object.__setattr__(
            self,
            "observed_sha256",
            _sha256(self.observed_sha256, field="attestation observed_sha256"),
        )
        if self.observed_sha256 != self.contract.sha256:
            raise ArtifactContractError("attestation digest differs from its contract")
        if not isinstance(self.paths, tuple) or not self.paths:
            raise ArtifactContractError("attestation paths must be a non-empty tuple")
        if any(not isinstance(item, ArtifactPathState) for item in self.paths):
            raise ArtifactContractError("attestation paths contain an invalid state")
        logical_paths = [item.path for item in self.paths]
        if logical_paths[0] != "." or len(logical_paths) != len(set(logical_paths)):
            raise ArtifactContractError("attestation paths must start at '.' and be unique")

    @property
    def state_sha256(self) -> str:
        payload = [item.to_dict() for item in self.paths]
        return hashlib.sha256(_canonical_json(payload)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _ATTESTATION_SCHEMA_VERSION,
            "contract": self.contract.to_dict(),
            "observed_sha256": self.observed_sha256,
            "state_sha256": self.state_sha256,
            "paths": [item.to_dict() for item in self.paths],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactAttestation:
        if not isinstance(value, Mapping):
            raise ArtifactContractError("artifact attestation must be a mapping")
        expected_fields = {
            "schema_version",
            "contract",
            "observed_sha256",
            "state_sha256",
            "paths",
        }
        if set(value) != expected_fields:
            raise ArtifactContractError("artifact attestation fields differ from the schema")
        if value.get("schema_version") != _ATTESTATION_SCHEMA_VERSION:
            raise ArtifactContractError("unsupported artifact attestation schema_version")
        raw_paths = value.get("paths")
        if not isinstance(raw_paths, list):
            raise ArtifactContractError("artifact attestation paths must be a list")
        attestation = cls(
            contract=ArtifactContract.from_dict(value["contract"]),
            observed_sha256=value["observed_sha256"],
            paths=tuple(ArtifactPathState.from_dict(item) for item in raw_paths),
        )
        declared_state = _sha256(value.get("state_sha256"), field="attestation state_sha256")
        if declared_state != attestation.state_sha256:
            raise ArtifactContractError("artifact attestation state digest differs")
        return attestation


def _root_path(root: str | Path) -> Path:
    try:
        resolved = Path(root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArtifactIntegrityError(f"artifact root is missing or invalid: {root}") from exc
    if not resolved.is_dir():
        raise ArtifactIntegrityError(f"artifact root is not a directory: {resolved}")
    return resolved


def _target_path(root: Path, contract: ArtifactContract) -> Path:
    relative = PurePosixPath(contract.path)
    lexical = root.joinpath(*relative.parts)
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            f"artifact is missing or escapes its declared root: {contract.path}"
        ) from exc
    if contract.kind == "file" and not lexical.is_file():
        raise ArtifactIntegrityError(f"declared file artifact is not a file: {contract.path}")
    if contract.kind == "tree" and (lexical.is_symlink() or not lexical.is_dir()):
        raise ArtifactIntegrityError(
            f"declared tree artifact is not a regular directory: {contract.path}"
        )
    return lexical


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(stat_module.S_IMODE(value.st_mode)),
    )


def _hash_file_stable(path: Path) -> tuple[str, os.stat_result]:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise ArtifactIntegrityError(f"cannot read artifact file: {path}") from exc
    if _stat_signature(before) != _stat_signature(after):
        raise ArtifactIntegrityError(f"artifact changed while being verified: {path}")
    return digest.hexdigest(), after


def _ensure_inside_root(path: Path, *, root: Path) -> None:
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"artifact path escapes its declared root: {path}") from exc


def _capture_path(path: Path, *, logical_path: str, root: Path) -> ArtifactPathState:
    try:
        link_stat = path.lstat()
    except OSError as exc:
        raise ArtifactIntegrityError(f"artifact path is missing: {path}") from exc

    if stat_module.S_ISLNK(link_stat.st_mode):
        link_target = os.readlink(path)
        if PurePosixPath(link_target).is_absolute():
            raise ArtifactIntegrityError(f"artifact contains an absolute symlink: {path}")
        _ensure_inside_root(path, root=root)
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ArtifactIntegrityError(f"artifact symlink does not target a regular file: {path}")
        content_sha256, target_stat = _hash_file_stable(path)
        try:
            final_link_stat = path.lstat()
        except OSError as exc:
            raise ArtifactIntegrityError(f"artifact symlink disappeared: {path}") from exc
        if os.readlink(path) != link_target or _stat_signature(link_stat) != _stat_signature(
            final_link_stat
        ):
            raise ArtifactIntegrityError(f"artifact symlink changed while being verified: {path}")
        return ArtifactPathState(
            path=logical_path,
            kind="symlink",
            size_bytes=int(target_stat.st_size),
            mtime_ns=int(target_stat.st_mtime_ns),
            ctime_ns=int(target_stat.st_ctime_ns),
            mode=int(stat_module.S_IMODE(target_stat.st_mode)),
            content_sha256=content_sha256,
            link_target=link_target,
            link_mtime_ns=int(final_link_stat.st_mtime_ns),
            link_ctime_ns=int(final_link_stat.st_ctime_ns),
        )

    _ensure_inside_root(path, root=root)
    if stat_module.S_ISREG(link_stat.st_mode):
        content_sha256, file_stat = _hash_file_stable(path)
        return ArtifactPathState(
            path=logical_path,
            kind="file",
            size_bytes=int(file_stat.st_size),
            mtime_ns=int(file_stat.st_mtime_ns),
            ctime_ns=int(file_stat.st_ctime_ns),
            mode=int(stat_module.S_IMODE(file_stat.st_mode)),
            content_sha256=content_sha256,
        )
    if stat_module.S_ISDIR(link_stat.st_mode):
        return ArtifactPathState(
            path=logical_path,
            kind="directory",
            size_bytes=int(link_stat.st_size),
            mtime_ns=int(link_stat.st_mtime_ns),
            ctime_ns=int(link_stat.st_ctime_ns),
            mode=int(stat_module.S_IMODE(link_stat.st_mode)),
        )
    raise ArtifactIntegrityError(f"artifact contains an unsupported filesystem entry: {path}")


def _tree_digest(paths: tuple[ArtifactPathState, ...]) -> str:
    entries: list[dict[str, Any]] = []
    for state in paths[1:]:
        entry: dict[str, Any] = {"path": state.path, "kind": state.kind}
        if state.kind != "directory":
            entry.update(
                {
                    "size_bytes": state.size_bytes,
                    "sha256": state.content_sha256,
                }
            )
        if state.kind == "symlink":
            entry["link_target"] = state.link_target
        entries.append(entry)
    payload = {
        "schema_version": _TREE_DIGEST_SCHEMA_VERSION,
        "kind": "tree",
        "entries": entries,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _scan_artifact(
    contract: ArtifactContract,
    *,
    root: str | Path,
) -> tuple[str, tuple[ArtifactPathState, ...]]:
    resolved_root = _root_path(root)
    target = _target_path(resolved_root, contract)
    if contract.kind == "file":
        state = _capture_path(target, logical_path=".", root=resolved_root)
        if state.kind == "directory" or state.content_sha256 is None:
            raise ArtifactIntegrityError(f"declared file artifact is invalid: {contract.path}")
        return state.content_sha256, (state,)

    states = [_capture_path(target, logical_path=".", root=resolved_root)]

    def walk(directory: Path, prefix: PurePosixPath) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise ArtifactIntegrityError(f"cannot enumerate artifact tree: {directory}") from exc
        for child in children:
            logical = (prefix / child.name).as_posix()
            state = _capture_path(child, logical_path=logical, root=resolved_root)
            states.append(state)
            if state.kind == "directory":
                walk(child, prefix / child.name)

    walk(target, PurePosixPath("."))
    ordered = tuple(states)
    return _tree_digest(ordered), ordered


def artifact_sha256(
    root: str | Path,
    *,
    path: str,
    kind: ArtifactKind,
) -> str:
    """Compute the digest used by an :class:`ArtifactContract`.

    File digests are the conventional SHA-256 of their bytes. Tree digests are
    the SHA-256 of a canonical, sorted manifest containing every relative path,
    entry kind, file size, file digest, and relative symlink target. Filesystem
    timestamps and absolute roots are deliberately excluded.
    """

    contract = ArtifactContract(path=path, kind=kind, sha256="0" * 64)
    digest, _states = _scan_artifact(contract, root=root)
    return digest


def verify_artifact(
    contract: ArtifactContract,
    *,
    root: str | Path,
) -> ArtifactAttestation:
    """Fully re-hash one declared artifact and create a runtime attestation."""

    if not isinstance(contract, ArtifactContract):
        raise TypeError("contract must be an ArtifactContract")
    observed_sha256, paths = _scan_artifact(contract, root=root)
    if observed_sha256 != contract.sha256:
        raise ArtifactIntegrityError(
            f"artifact SHA-256 differs for {contract.path!r}: "
            f"computed {observed_sha256}, expected {contract.sha256}"
        )
    return ArtifactAttestation(
        contract=contract,
        observed_sha256=observed_sha256,
        paths=paths,
    )


def revalidate_artifact(
    attestation: ArtifactAttestation,
    *,
    root: str | Path,
) -> ArtifactAttestation:
    """Re-hash and compare an artifact with its successful preflight state."""

    if not isinstance(attestation, ArtifactAttestation):
        raise TypeError("attestation must be an ArtifactAttestation")
    observed_sha256, paths = _scan_artifact(attestation.contract, root=root)
    if observed_sha256 != attestation.observed_sha256:
        raise ArtifactIntegrityError(
            f"artifact changed after preflight: {attestation.contract.path} (SHA-256 differs)"
        )
    if paths != attestation.paths:
        previous = {item.path: item for item in attestation.paths}
        current = {item.path: item for item in paths}
        changed_path = next(
            path
            for path in sorted(previous.keys() | current.keys())
            if previous.get(path) != current.get(path)
        )
        raise ArtifactIntegrityError(
            f"artifact changed after preflight: {attestation.contract.path} "
            f"(filesystem state differs at {changed_path})"
        )
    return ArtifactAttestation(
        contract=attestation.contract,
        observed_sha256=observed_sha256,
        paths=paths,
    )


__all__ = [
    "ArtifactAttestation",
    "ArtifactContract",
    "ArtifactContractError",
    "ArtifactIntegrityError",
    "ArtifactKind",
    "ArtifactPathState",
    "artifact_sha256",
    "revalidate_artifact",
    "verify_artifact",
]
