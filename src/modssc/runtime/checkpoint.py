from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .execution import ExecutionContext, ResumePolicy, RunIdentity, normalize_resume_policy

PayloadSerializer = Callable[[Any], bytes]
PayloadDeserializer = Callable[[bytes], Any]

_CHECKPOINT_SCHEMA_VERSION = 1
_POINTER_FILENAME = "CURRENT.json"
_PAYLOAD_FILENAME = "payload.bin"
_METADATA_FILENAME = "checkpoint.json"


class CheckpointError(RuntimeError):
    """Base class for native checkpoint failures."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when a required checkpoint does not exist."""


class CheckpointIdentityError(CheckpointError):
    """Raised when checkpoint state belongs to another run identity."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint state is malformed or content has changed."""


@dataclass(frozen=True)
class CheckpointRecord:
    step: int
    reason: str | None
    created_at: str
    payload_sha256: str
    payload_size_bytes: int
    generation_dir: Path


@dataclass(frozen=True)
class LoadedCheckpoint:
    payload: Any
    record: CheckpointRecord


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _step(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    return value


def _reason(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("checkpoint reason must be a non-empty string or None")
    return value


def _payload_bytes(payload: Any, serializer: PayloadSerializer | None) -> bytes:
    encoded = serializer(payload) if serializer is not None else payload
    if not isinstance(encoded, bytes | bytearray | memoryview):
        if serializer is None:
            raise TypeError("checkpoint payload must be bytes-like when no serializer is provided")
        raise TypeError("checkpoint serializer must return bytes-like data")
    return bytes(encoded)


def _write_durable(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_durable(temporary, _canonical_json(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CheckpointIntegrityError(f"{label} is missing or is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"{label} must be a JSON object")
    return value


class CheckpointStore:
    """Immutable, content-addressed checkpoints with an atomic current pointer."""

    def __init__(self, root: str | Path, identity: RunIdentity) -> None:
        if not isinstance(identity, RunIdentity):
            raise TypeError("identity must be a RunIdentity")
        self.root = Path(root).expanduser().resolve()
        self.identity = identity

    @classmethod
    def from_context(cls, context: ExecutionContext) -> CheckpointStore:
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")
        return cls(context.checkpoint_dir, context.identity)

    @property
    def generations_dir(self) -> Path:
        return self.root / "generations"

    @property
    def pointer_path(self) -> Path:
        return self.root / _POINTER_FILENAME

    @property
    def has_checkpoint(self) -> bool:
        return self.pointer_path.is_file() and not self.pointer_path.is_symlink()

    def save(
        self,
        payload: Any,
        *,
        step: int,
        reason: str | None = None,
        serializer: PayloadSerializer | None = None,
    ) -> CheckpointRecord:
        normalized_step = _step(step)
        normalized_reason = _reason(reason)
        encoded = _payload_bytes(payload, serializer)
        payload_sha256 = _sha256_bytes(encoded)
        generation_name = f"step-{normalized_step:012d}-{payload_sha256}"
        generation = self.generations_dir / generation_name

        self.generations_dir.mkdir(parents=True, exist_ok=True)
        if generation.exists():
            record = self._read_generation(generation)
        else:
            staging = self.generations_dir / f".staging-{uuid.uuid4().hex}"
            staging.mkdir()
            try:
                _write_durable(staging / _PAYLOAD_FILENAME, encoded)
                created_at = datetime.now(UTC).isoformat()
                metadata = {
                    "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                    "identity": self.identity.to_dict(),
                    "identity_sha256": self.identity.sha256,
                    "step": normalized_step,
                    "reason": normalized_reason,
                    "created_at": created_at,
                    "payload_sha256": payload_sha256,
                    "payload_size_bytes": len(encoded),
                }
                _write_durable(staging / _METADATA_FILENAME, _canonical_json(metadata))
                os.replace(staging, generation)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            record = self._read_generation(generation)

        pointer = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "identity_sha256": self.identity.sha256,
            "generation": generation_name,
            "step": record.step,
            "payload_sha256": record.payload_sha256,
        }
        _replace_json(self.pointer_path, pointer)
        return record

    def load(
        self,
        *,
        resume_policy: ResumePolicy = "auto",
        deserializer: PayloadDeserializer | None = None,
    ) -> LoadedCheckpoint | None:
        policy = normalize_resume_policy(resume_policy)
        if policy == "never":
            return None
        if not self.has_checkpoint:
            if policy == "required":
                raise CheckpointNotFoundError("a checkpoint is required but none exists")
            return None

        pointer = _read_json(self.pointer_path, label="checkpoint pointer")
        expected_pointer_fields = {
            "schema_version",
            "identity_sha256",
            "generation",
            "step",
            "payload_sha256",
        }
        if set(pointer) != expected_pointer_fields or pointer.get("schema_version") != 1:
            raise CheckpointIntegrityError("checkpoint pointer fields differ from the schema")
        if pointer.get("identity_sha256") != self.identity.sha256:
            raise CheckpointIdentityError("checkpoint pointer identity differs from the run")
        generation_name = pointer.get("generation")
        if (
            not isinstance(generation_name, str)
            or not generation_name
            or Path(generation_name).name != generation_name
        ):
            raise CheckpointIntegrityError("checkpoint pointer generation is invalid")
        if not _is_sha256(pointer.get("payload_sha256")):
            raise CheckpointIntegrityError("checkpoint pointer payload digest is invalid")
        try:
            pointer_step = _step(pointer.get("step"))
        except ValueError as exc:
            raise CheckpointIntegrityError("checkpoint pointer step is invalid") from exc

        generation = (self.generations_dir / generation_name).resolve()
        try:
            generation.relative_to(self.generations_dir.resolve())
        except ValueError as exc:
            raise CheckpointIntegrityError("checkpoint generation escapes its store") from exc
        record = self._read_generation(generation)
        if record.step != pointer_step or record.payload_sha256 != pointer["payload_sha256"]:
            raise CheckpointIntegrityError("checkpoint pointer and generation differ")

        encoded = (generation / _PAYLOAD_FILENAME).read_bytes()
        payload = deserializer(encoded) if deserializer is not None else encoded
        return LoadedCheckpoint(payload=payload, record=record)

    def load_from_context(
        self,
        context: ExecutionContext,
        *,
        deserializer: PayloadDeserializer | None = None,
    ) -> LoadedCheckpoint | None:
        if context.identity != self.identity or context.checkpoint_dir != self.root:
            raise CheckpointIdentityError("execution context differs from the checkpoint store")
        return self.load(resume_policy=context.resume_policy, deserializer=deserializer)

    def prune(self, *, keep_last: int = 1) -> tuple[Path, ...]:
        """Remove superseded generations while preserving the current pointer.

        This is intended for large training states where retaining every
        periodic generation would exhaust a shared filesystem. Integrity of
        the retained generation is verified before any older state is removed.
        """

        if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 1:
            raise ValueError("keep_last must be a positive integer")
        if not self.generations_dir.is_dir():
            return ()
        current: str | None = None
        if self.has_checkpoint:
            pointer = _read_json(self.pointer_path, label="checkpoint pointer")
            value = pointer.get("generation")
            if not isinstance(value, str) or Path(value).name != value:
                raise CheckpointIntegrityError("checkpoint pointer generation is invalid")
            current = value
        generations: list[tuple[CheckpointRecord, Path]] = []
        for candidate in self.generations_dir.iterdir():
            if candidate.name.startswith(".staging-"):
                continue
            generations.append((self._read_generation(candidate), candidate))
        generations.sort(
            key=lambda item: (item[0].step, item[0].created_at, item[1].name),
            reverse=True,
        )
        retained = {path.name for _, path in generations[:keep_last]}
        if current is not None:
            retained.add(current)
            current_path = self.generations_dir / current
            self._read_generation(current_path)
        removed: list[Path] = []
        for _, path in generations:
            if path.name in retained:
                continue
            shutil.rmtree(path)
            removed.append(path)
        return tuple(removed)

    def _read_generation(self, generation: Path) -> CheckpointRecord:
        if not generation.is_dir() or generation.is_symlink():
            raise CheckpointIntegrityError("checkpoint generation is missing or invalid")
        metadata = _read_json(
            generation / _METADATA_FILENAME,
            label="checkpoint metadata",
        )
        expected_metadata_fields = {
            "schema_version",
            "identity",
            "identity_sha256",
            "step",
            "reason",
            "created_at",
            "payload_sha256",
            "payload_size_bytes",
        }
        if set(metadata) != expected_metadata_fields or metadata.get("schema_version") != 1:
            raise CheckpointIntegrityError("checkpoint metadata fields differ from the schema")
        if metadata.get("identity_sha256") != self.identity.sha256:
            raise CheckpointIdentityError("checkpoint metadata identity differs from the run")
        try:
            metadata_identity = RunIdentity.from_dict(metadata.get("identity"))
        except (TypeError, ValueError) as exc:
            raise CheckpointIntegrityError("checkpoint metadata run identity is invalid") from exc
        if metadata_identity != self.identity:
            raise CheckpointIdentityError("checkpoint metadata belongs to another run")

        try:
            step = _step(metadata.get("step"))
            reason = _reason(metadata.get("reason"))
        except ValueError as exc:
            raise CheckpointIntegrityError("checkpoint metadata values are invalid") from exc
        created_at = metadata.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            raise CheckpointIntegrityError("checkpoint metadata created_at is invalid")
        payload_sha256 = metadata.get("payload_sha256")
        if not _is_sha256(payload_sha256):
            raise CheckpointIntegrityError("checkpoint metadata payload digest is invalid")
        payload_size = metadata.get("payload_size_bytes")
        if isinstance(payload_size, bool) or not isinstance(payload_size, int) or payload_size < 0:
            raise CheckpointIntegrityError("checkpoint metadata payload size is invalid")

        payload_path = generation / _PAYLOAD_FILENAME
        if not payload_path.is_file() or payload_path.is_symlink():
            raise CheckpointIntegrityError("checkpoint payload is missing or invalid")
        if payload_path.stat().st_size != payload_size:
            raise CheckpointIntegrityError("checkpoint payload size differs from metadata")
        if _sha256_file(payload_path) != payload_sha256:
            raise CheckpointIntegrityError("checkpoint payload digest differs from metadata")
        return CheckpointRecord(
            step=step,
            reason=reason,
            created_at=created_at,
            payload_sha256=payload_sha256,
            payload_size_bytes=payload_size,
            generation_dir=generation,
        )


__all__ = [
    "CheckpointError",
    "CheckpointIdentityError",
    "CheckpointIntegrityError",
    "CheckpointNotFoundError",
    "CheckpointRecord",
    "CheckpointStore",
    "LoadedCheckpoint",
    "PayloadDeserializer",
    "PayloadSerializer",
]
