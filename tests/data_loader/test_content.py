from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import modssc.data_loader.content as content
from modssc.data_loader.cache import CacheLayout
from modssc.data_loader.errors import ManifestError
from modssc.data_loader.storage import FileStorage
from modssc.data_loader.types import LoadedDataset, Split


def _identity(*, provider: str = "toy", dataset_id: str = "toy") -> dict[str, object]:
    return {
        "provider": provider,
        "canonical_uri": f"{provider}:{dataset_id}",
        "dataset_id": dataset_id,
        "version": None,
        "modality": "audio" if provider == "torchaudio" else "tabular",
        "task": "classification",
        "resolved_kwargs": {},
    }


def _write_manifest(layout: CacheLayout, fingerprint: str, manifest: dict[str, object]) -> None:
    path = layout.content_manifest_path(fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.content_manifest_json(manifest), encoding="utf-8")


def _valid_processed_record(*, path: str = "layout.json") -> dict[str, object]:
    return {
        "kind": "processed",
        "path": path,
        "size_bytes": 1,
        "sha256": "1" * 64,
        "storage_sha256": "2" * 64,
        "digest_mode": "bytes",
    }


def _manifest_for_records(
    records: list[object], *, fingerprint: str = "f" * 64
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": content.CONTENT_MANIFEST_SCHEMA_VERSION,
        "dataset_fingerprint": fingerprint,
        "files": records,
    }
    return {**payload, "content_sha256": content._content_sha256(payload)}


def _prepare_numeric_manifest(tmp_path: Path) -> tuple[CacheLayout, str, dict[str, object]]:
    layout = CacheLayout(tmp_path)
    fingerprint = "d" * 64
    dataset = LoadedDataset(
        train=Split(X=np.asarray([[1.0]], dtype=np.float32), y=np.asarray([0])),
        meta={"dataset_fingerprint": fingerprint},
    )
    FileStorage().save(layout.processed_dir(fingerprint), dataset)
    manifest = content.build_content_manifest(layout, fingerprint, dataset, identity=_identity())
    _write_manifest(layout, fingerprint, manifest)
    return layout, fingerprint, manifest


def test_path_discovery_never_casts_numeric_arrays_to_object(monkeypatch) -> None:
    array = np.ones((128, 128), dtype=np.uint8)
    dataset = LoadedDataset(train=Split(X=array, y=np.zeros(128, dtype=np.int64)))
    original = content.np.asarray

    def guarded_asarray(value, *args, **kwargs):
        assert "dtype" not in kwargs
        assert not args
        return original(value)

    monkeypatch.setattr(content.np, "asarray", guarded_asarray)

    assert list(content._path_strings(dataset)) == []


def test_path_discovery_skips_unconvertible_splits_and_non_strings() -> None:
    class BrokenArray:
        def __array__(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("cannot convert")

    broken = LoadedDataset(train=Split(X=BrokenArray(), y=np.asarray([0])))
    assert list(content._path_strings(broken)) == []

    mixed = LoadedDataset(
        train=Split(X=np.asarray(["train.wav", 7], dtype=object), y=np.asarray([0, 1])),
        test=Split(X=np.asarray(["test.wav"], dtype=object), y=np.asarray([0])),
    )
    assert list(content._path_strings(mixed)) == ["train.wav", "test.wav"]


def test_torchaudio_source_root_requires_dataset_id(tmp_path) -> None:
    layout = CacheLayout(tmp_path)

    with pytest.raises(ManifestError, match="no dataset_id"):
        content._source_root(layout, _identity(provider="torchaudio", dataset_id=""))


def test_source_path_resolution_covers_relative_missing_and_directory_paths(tmp_path) -> None:
    source_root = tmp_path / "cache" / "source"
    audio = source_root / "yes" / "sample.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")

    assert content._resolve_source_path("yes/sample.wav", source_root=source_root) == (
        "yes/sample.wav",
        audio,
    )

    missing = tmp_path / "elsewhere" / "missing.wav"
    with pytest.raises(ManifestError, match="missing or outside"):
        content._resolve_source_path(str(missing), source_root=source_root)

    directory = source_root / "folder"
    directory.mkdir()
    legacy_directory = tmp_path / "old-cache" / "source" / "folder"
    with pytest.raises(ManifestError, match="is not a file"):
        content._resolve_source_path(str(legacy_directory), source_root=source_root)


def test_build_content_manifest_requires_processed_cache(tmp_path) -> None:
    with pytest.raises(ManifestError, match="Missing processed dataset cache"):
        content.build_content_manifest(
            CacheLayout(tmp_path),
            "0" * 64,
            LoadedDataset(train=Split(X=np.asarray([1]), y=np.asarray([0]))),
            identity=_identity(),
        )


def test_content_manifest_detects_cached_array_tampering(tmp_path) -> None:
    layout = CacheLayout(tmp_path)
    fingerprint = "a" * 64
    dataset = LoadedDataset(
        train=Split(
            X=np.array([[1.0], [2.0]], dtype=np.float32),
            y=np.array([0, 1], dtype=np.int64),
        ),
        meta={"dataset_fingerprint": fingerprint},
    )
    FileStorage().save(layout.processed_dir(fingerprint), dataset)
    manifest = content.build_content_manifest(layout, fingerprint, dataset, identity=_identity())
    _write_manifest(layout, fingerprint, manifest)

    evidence = content.verify_content_manifest(
        layout, fingerprint, identity=_identity(), rehash=True
    )
    assert evidence["content_sha256"] == manifest["content_sha256"]

    np.save(
        layout.processed_dir(fingerprint) / "train_X.npy",
        np.array([[9.0], [2.0]], dtype=np.float32),
    )
    with pytest.raises(ManifestError, match="digest differs"):
        content.verify_content_manifest(layout, fingerprint, identity=_identity(), rehash=True)


def test_audio_content_covers_sources_and_is_cache_root_independent(tmp_path) -> None:
    fingerprint = "b" * 64
    first = CacheLayout(tmp_path / "first")
    identity = _identity(provider="torchaudio", dataset_id="SPEECHCOMMANDS")
    first_source = first.raw_dir("torchaudio", "SPEECHCOMMANDS", None) / "source"
    audio = first_source / "yes" / "sample.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFF-audio-one")
    dataset = LoadedDataset(
        train=Split(
            X=np.asarray([str(audio)], dtype=object),
            y=np.asarray(["yes"], dtype=object),
        ),
        meta={"dataset_fingerprint": fingerprint, "provider": "torchaudio"},
    )
    FileStorage().save(first.processed_dir(fingerprint), dataset)
    manifest = content.build_content_manifest(first, fingerprint, dataset, identity=identity)
    _write_manifest(first, fingerprint, manifest)
    content.verify_content_manifest(first, fingerprint, identity=identity, rehash=True)

    second = CacheLayout(tmp_path / "second")
    shutil.copytree(first.processed_dir(fingerprint), second.processed_dir(fingerprint))
    shutil.copytree(first_source, second.raw_dir("torchaudio", "SPEECHCOMMANDS", None) / "source")
    second.content_manifest_path(fingerprint).parent.mkdir(parents=True)
    shutil.copy2(
        first.content_manifest_path(fingerprint),
        second.content_manifest_path(fingerprint),
    )
    moved = content.verify_content_manifest(second, fingerprint, identity=identity, rehash=True)
    assert moved["content_sha256"] == manifest["content_sha256"]
    stored_paths = FileStorage().load(second.processed_dir(fingerprint)).train.X.tolist()
    assert stored_paths == [str(audio)]

    moved_audio = (
        second.raw_dir("torchaudio", "SPEECHCOMMANDS", None) / "source" / "yes" / "sample.wav"
    )
    moved_audio.write_bytes(b"RIFF-audio-two")
    with pytest.raises(ManifestError, match="digest differs"):
        content.verify_content_manifest(second, fingerprint, identity=identity, rehash=True)


def test_content_manifest_rejects_escaping_paths(tmp_path) -> None:
    layout = CacheLayout(tmp_path)
    fingerprint = "c" * 64
    processed = layout.processed_dir(fingerprint)
    processed.mkdir(parents=True)
    (processed / "layout.json").write_text("{}", encoding="utf-8")
    manifest = content.build_content_manifest(
        layout,
        fingerprint,
        LoadedDataset(train=Split(X=np.array([1]), y=np.array([0]))),
        identity=_identity(),
    )
    manifest["files"][0]["path"] = "../outside"
    _write_manifest(layout, fingerprint, manifest)

    with pytest.raises(ManifestError, match="escapes"):
        content.verify_content_manifest(layout, fingerprint, identity=_identity(), rehash=False)


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("{", "Invalid dataset content manifest"),
        ("[]", "root must be a mapping"),
    ],
)
def test_read_content_manifest_rejects_invalid_json_roots(tmp_path, contents, match) -> None:
    path = tmp_path / "content.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ManifestError, match=match):
        content.read_content_manifest(path)


def test_read_content_manifest_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(ManifestError, match="Invalid dataset content manifest"):
        content.read_content_manifest(tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 999, "Unsupported"),
        ("dataset_fingerprint", "different", "fingerprint differs"),
        ("files", "not-a-list", "file table is empty"),
        ("files", [], "file table is empty"),
    ],
)
def test_validate_manifest_rejects_invalid_envelope(field, value, match) -> None:
    fingerprint = "f" * 64
    manifest = _manifest_for_records([_valid_processed_record()], fingerprint=fingerprint)
    manifest[field] = value

    with pytest.raises(ManifestError, match=match):
        content._validate_manifest(manifest, fingerprint=fingerprint)


def test_validate_manifest_rejects_non_mapping_record() -> None:
    fingerprint = "f" * 64
    manifest = {
        "schema_version": content.CONTENT_MANIFEST_SCHEMA_VERSION,
        "dataset_fingerprint": fingerprint,
        "files": [None],
        "content_sha256": "0" * 64,
    }

    with pytest.raises(ManifestError, match="file record"):
        content._validate_manifest(manifest, fingerprint=fingerprint)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("kind", "other", "file kind"),
        ("path", None, "relative path"),
        ("path", "", "relative path"),
        ("path", "/absolute", "relative path"),
        ("path", "../outside", "escapes"),
        ("size_bytes", True, "file size"),
        ("size_bytes", "1", "file size"),
        ("size_bytes", -1, "file size"),
        ("sha256", None, "file digest"),
        ("sha256", "short", "file digest"),
        ("storage_sha256", None, "storage digest"),
        ("storage_sha256", "short", "storage digest"),
        ("digest_mode", "unknown", "digest mode"),
    ],
)
def test_validate_manifest_rejects_invalid_record_fields(field, value, match) -> None:
    fingerprint = "f" * 64
    record = _valid_processed_record()
    record[field] = value
    manifest = _manifest_for_records([record], fingerprint=fingerprint)

    with pytest.raises(ManifestError, match=match):
        content._validate_manifest(manifest, fingerprint=fingerprint)


def test_validate_manifest_rejects_duplicate_and_semantically_tampered_records() -> None:
    fingerprint = "f" * 64
    record = _valid_processed_record()
    duplicate = _manifest_for_records([record, dict(record)], fingerprint=fingerprint)
    with pytest.raises(ManifestError, match="Duplicate"):
        content._validate_manifest(duplicate, fingerprint=fingerprint)

    tampered = _manifest_for_records([record], fingerprint=fingerprint)
    tampered["content_sha256"] = "0" * 64
    with pytest.raises(ManifestError, match="semantic digest differs"):
        content._validate_manifest(tampered, fingerprint=fingerprint)


def test_record_path_rejects_source_for_non_path_dataset_and_escape(tmp_path) -> None:
    layout = CacheLayout(tmp_path)
    fingerprint = "f" * 64

    with pytest.raises(ManifestError, match="non-path dataset"):
        content._record_path(
            {"kind": "source", "path": "audio.wav"},
            layout=layout,
            fingerprint=fingerprint,
            identity=_identity(),
        )

    with pytest.raises(ManifestError, match="escapes"):
        content._record_path(
            {"kind": "processed", "path": "../outside"},
            layout=layout,
            fingerprint=fingerprint,
            identity=_identity(),
        )


def test_verify_content_manifest_supports_cheap_attestation(tmp_path) -> None:
    layout, fingerprint, manifest = _prepare_numeric_manifest(tmp_path)

    evidence = content.verify_content_manifest(
        layout, fingerprint, identity=_identity(), rehash=False
    )

    assert evidence["content_sha256"] == manifest["content_sha256"]
    assert evidence["cache_fingerprint"] == fingerprint


def test_verify_content_manifest_rejects_missing_and_resized_files(tmp_path) -> None:
    missing_layout, fingerprint, missing_manifest = _prepare_numeric_manifest(tmp_path / "missing")
    missing_record = missing_manifest["files"][0]
    missing_path = missing_layout.processed_dir(fingerprint) / missing_record["path"]
    missing_path.unlink()
    with pytest.raises(ManifestError, match="file is missing"):
        content.verify_content_manifest(
            missing_layout, fingerprint, identity=_identity(), rehash=False
        )

    resized_layout, fingerprint, resized_manifest = _prepare_numeric_manifest(tmp_path / "resized")
    resized_record = resized_manifest["files"][0]
    resized_path = resized_layout.processed_dir(fingerprint) / resized_record["path"]
    resized_path.write_bytes(resized_path.read_bytes() + b"changed")
    with pytest.raises(ManifestError, match="file size differs"):
        content.verify_content_manifest(
            resized_layout, fingerprint, identity=_identity(), rehash=False
        )


def test_verify_content_manifest_rejects_storage_digest_tampering(tmp_path) -> None:
    layout, fingerprint, manifest = _prepare_numeric_manifest(tmp_path)
    processed = next(record for record in manifest["files"] if record["kind"] == "processed")
    processed["storage_sha256"] = "0" * 64
    _write_manifest(layout, fingerprint, manifest)

    with pytest.raises(ManifestError, match="storage file digest differs"):
        content.verify_content_manifest(layout, fingerprint, identity=_identity(), rehash=True)


def test_content_manifest_json_is_versioned() -> None:
    payload = json.loads(
        content.content_manifest_json(
            {"schema_version": 1, "dataset_fingerprint": "fp", "files": []}
        )
    )
    assert payload["schema_version"] == content.CONTENT_MANIFEST_SCHEMA_VERSION
