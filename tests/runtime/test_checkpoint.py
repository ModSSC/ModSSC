from __future__ import annotations

import json
from pathlib import Path

import pytest

from modssc.runtime import checkpoint as checkpoint_module
from modssc.runtime.checkpoint import (
    CheckpointIdentityError,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
    CheckpointStore,
)
from modssc.runtime.execution import ExecutionContext, RunIdentity

CONFIG_SHA256 = "a" * 64
OTHER_CONFIG_SHA256 = "b" * 64


def _identity(seed: int = 7) -> RunIdentity:
    return RunIdentity(CONFIG_SHA256, seed)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_checkpoint_bytes_round_trip_is_content_addressed_and_atomic(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())
    assert not store.has_checkpoint

    record = store.save(b"checkpoint-data", step=12, reason="periodic")

    assert store.has_checkpoint
    assert record.step == 12
    assert record.reason == "periodic"
    assert record.payload_size_bytes == len(b"checkpoint-data")
    assert record.generation_dir.name == f"step-{12:012d}-{record.payload_sha256}"
    assert (record.generation_dir / "payload.bin").read_bytes() == b"checkpoint-data"
    assert not list(store.root.glob(".CURRENT.json.*.tmp"))
    assert not list(store.generations_dir.glob(".staging-*"))

    loaded = store.load(resume_policy="auto")
    assert loaded is not None
    assert loaded.payload == b"checkpoint-data"
    assert loaded.record == record

    repeated = store.save(memoryview(b"checkpoint-data"), step=12, reason="ignored-on-reuse")
    assert repeated == record
    assert len(list(store.generations_dir.iterdir())) == 1


def test_checkpoint_supports_serializer_callbacks(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())
    record = store.save(
        {"epoch": 3},
        step=3,
        serializer=lambda value: json.dumps(value, sort_keys=True).encode("utf-8"),
    )
    loaded = store.load(
        resume_policy="required",
        deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    assert loaded is not None
    assert loaded.payload == {"epoch": 3}
    assert loaded.record == record


def test_checkpoint_context_controls_resume_without_environment(tmp_path: Path) -> None:
    context = ExecutionContext(
        _identity(),
        tmp_path / "runs",
        resume_policy="auto",
        checkpoint_root=tmp_path / "checkpoints",
    )
    store = CheckpointStore.from_context(context)

    assert store.root == context.checkpoint_dir
    assert store.load_from_context(context) is None
    store.save(bytearray(b"state"), step=1)
    assert store.load_from_context(context).payload == b"state"  # type: ignore[union-attr]

    other_context = ExecutionContext(
        RunIdentity(OTHER_CONFIG_SHA256, 7),
        tmp_path / "runs",
        resume_policy="auto",
        checkpoint_root=tmp_path / "checkpoints",
    )
    with pytest.raises(CheckpointIdentityError, match="context"):
        store.load_from_context(other_context)

    with pytest.raises(TypeError, match="ExecutionContext"):
        CheckpointStore.from_context(object())  # type: ignore[arg-type]


def test_checkpoint_missing_and_never_policies(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())

    assert store.load(resume_policy="never") is None
    assert store.load(resume_policy="auto") is None
    with pytest.raises(CheckpointNotFoundError, match="required"):
        store.load(resume_policy="required")
    with pytest.raises(ValueError, match="resume_policy"):
        store.load(resume_policy="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"payload": b"x", "step": -1}, ValueError, "step"),
        ({"payload": b"x", "step": True}, ValueError, "step"),
        ({"payload": b"x", "step": 1, "reason": ""}, ValueError, "reason"),
        ({"payload": object(), "step": 1}, TypeError, "bytes-like"),
        (
            {"payload": object(), "step": 1, "serializer": lambda _: "not-bytes"},
            TypeError,
            "serializer",
        ),
    ],
)
def test_checkpoint_rejects_invalid_save_values(
    tmp_path: Path,
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())
    with pytest.raises(error, match=match):
        store.save(**kwargs)  # type: ignore[arg-type]


def test_checkpoint_store_rejects_invalid_identity(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="RunIdentity"):
        CheckpointStore(tmp_path, object())  # type: ignore[arg-type]


def test_checkpoint_rejects_another_identity_at_the_same_root(tmp_path: Path) -> None:
    root = tmp_path / "state"
    CheckpointStore(root, _identity()).save(b"state", step=1)
    other = CheckpointStore(root, RunIdentity(OTHER_CONFIG_SHA256, 7))

    with pytest.raises(CheckpointIdentityError, match="pointer identity"):
        other.load(resume_policy="required")
    with pytest.raises(CheckpointIdentityError, match="metadata identity"):
        other.save(b"state", step=1)


@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        (lambda value: value.update(extra=True), CheckpointIntegrityError, "fields"),
        (
            lambda value: value.update(identity_sha256="0" * 64),
            CheckpointIdentityError,
            "pointer identity",
        ),
        (
            lambda value: value.update(generation="../escape"),
            CheckpointIntegrityError,
            "generation",
        ),
        (
            lambda value: value.update(payload_sha256="bad"),
            CheckpointIntegrityError,
            "digest",
        ),
        (lambda value: value.update(step=-1), CheckpointIntegrityError, "step"),
        (lambda value: value.update(step=99), CheckpointIntegrityError, "generation differ"),
    ],
)
def test_checkpoint_validates_pointer(
    tmp_path: Path,
    mutation,
    error: type[Exception],
    match: str,
) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())
    store.save(b"state", step=1)
    pointer = _read_json(store.pointer_path)
    mutation(pointer)
    _write_json(store.pointer_path, pointer)

    with pytest.raises(error, match=match):
        store.load(resume_policy="required")


def test_checkpoint_rejects_generation_escape_through_symlink(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())
    record = store.save(b"state", step=1)
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = store.generations_dir / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    pointer = _read_json(store.pointer_path)
    pointer["generation"] = escape.name
    pointer["step"] = record.step
    pointer["payload_sha256"] = record.payload_sha256
    _write_json(store.pointer_path, pointer)

    with pytest.raises(CheckpointIntegrityError, match="escapes"):
        store.load(resume_policy="required")


@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        (lambda value: value.update(extra=True), CheckpointIntegrityError, "fields"),
        (
            lambda value: value.update(identity_sha256="0" * 64),
            CheckpointIdentityError,
            "metadata identity",
        ),
        (
            lambda value: value.update(identity={"schema_version": 99}),
            CheckpointIntegrityError,
            "run identity",
        ),
        (
            lambda value: value.update(identity=RunIdentity(OTHER_CONFIG_SHA256, 7).to_dict()),
            CheckpointIdentityError,
            "another run",
        ),
        (lambda value: value.update(step=-1), CheckpointIntegrityError, "values"),
        (lambda value: value.update(reason=""), CheckpointIntegrityError, "values"),
        (lambda value: value.update(created_at=""), CheckpointIntegrityError, "created_at"),
        (lambda value: value.update(payload_sha256="bad"), CheckpointIntegrityError, "digest"),
        (lambda value: value.update(payload_size_bytes=True), CheckpointIntegrityError, "size"),
    ],
)
def test_checkpoint_validates_metadata(
    tmp_path: Path,
    mutation,
    error: type[Exception],
    match: str,
) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())
    record = store.save(b"state", step=1)
    metadata_path = record.generation_dir / "checkpoint.json"
    metadata = _read_json(metadata_path)
    mutation(metadata)
    _write_json(metadata_path, metadata)

    with pytest.raises(error, match=match):
        store.load(resume_policy="required")


def test_checkpoint_detects_missing_and_tampered_payload(tmp_path: Path) -> None:
    missing_store = CheckpointStore(tmp_path / "missing", _identity())
    missing_record = missing_store.save(b"state", step=1)
    (missing_record.generation_dir / "payload.bin").unlink()
    with pytest.raises(CheckpointIntegrityError, match="payload is missing"):
        missing_store.load(resume_policy="required")

    size_store = CheckpointStore(tmp_path / "size", _identity())
    size_record = size_store.save(b"state", step=1)
    (size_record.generation_dir / "payload.bin").write_bytes(b"longer-state")
    with pytest.raises(CheckpointIntegrityError, match="size differs"):
        size_store.load(resume_policy="required")

    digest_store = CheckpointStore(tmp_path / "digest", _identity())
    digest_record = digest_store.save(b"state", step=1)
    (digest_record.generation_dir / "payload.bin").write_bytes(b"other")
    with pytest.raises(CheckpointIntegrityError, match="digest differs"):
        digest_store.load(resume_policy="required")


def test_checkpoint_detects_missing_metadata(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())
    record = store.save(b"state", step=1)
    (record.generation_dir / "checkpoint.json").unlink()

    with pytest.raises(CheckpointIntegrityError, match="metadata is missing"):
        store.load(resume_policy="required")


def test_checkpoint_cleans_staging_after_failed_metadata_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())
    real_write = checkpoint_module._write_durable

    def fail_metadata(path: Path, payload: bytes) -> None:
        if path.name == "checkpoint.json":
            raise OSError("injected write failure")
        real_write(path, payload)

    monkeypatch.setattr(checkpoint_module, "_write_durable", fail_metadata)
    with pytest.raises(OSError, match="injected"):
        store.save(b"state", step=1)

    assert not list(store.generations_dir.glob(".staging-*"))


def test_checkpoint_rejects_invalid_json_documents(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())
    store.save(b"state", step=1)

    store.pointer_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="invalid JSON"):
        store.load(resume_policy="required")

    store.pointer_path.write_text("[]", encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="JSON object"):
        store.load(resume_policy="required")


def test_checkpoint_rejects_invalid_generation_directory(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())
    with pytest.raises(CheckpointIntegrityError, match="generation"):
        store._read_generation(tmp_path / "absent")

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(CheckpointIntegrityError, match="generation"):
        store._read_generation(link)


def test_checkpoint_prune_keeps_current_and_requested_history(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())
    first = store.save(b"one", step=1)
    second = store.save(b"two", step=2)
    third = store.save(b"three", step=3)

    removed = store.prune(keep_last=2)

    assert removed == (first.generation_dir,)
    assert not first.generation_dir.exists()
    assert second.generation_dir.is_dir()
    assert third.generation_dir.is_dir()
    assert store.load(resume_policy="required").payload == b"three"  # type: ignore[union-attr]


@pytest.mark.parametrize("keep_last", [0, -1, True])
def test_checkpoint_prune_rejects_invalid_retention(tmp_path: Path, keep_last: object) -> None:
    store = CheckpointStore(tmp_path / "state", _identity())
    with pytest.raises(ValueError, match="keep_last"):
        store.prune(keep_last=keep_last)  # type: ignore[arg-type]
