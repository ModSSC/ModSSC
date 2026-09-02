from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import modssc.preprocess.cache as cache_module
from modssc.preprocess.errors import PreprocessCacheError


def _manager_with_value(tmp_path: Path) -> cache_module.CacheManager:
    manager = cache_module.CacheManager(root=tmp_path, dataset_fingerprint="dataset")
    manager.save_step_outputs(
        step_fingerprint="step",
        split="train",
        produced={"value": np.array([1, 2], dtype=np.int64)},
        manifest={"producer": "test"},
    )
    return manager


def _write_pointer(
    manager: cache_module.CacheManager,
    payload: dict[str, object],
) -> None:
    signed = cache_module._signed_manifest(payload)
    manager._manifest_path("step").write_text(
        cache_module.stable_json_dumps(signed), encoding="utf-8"
    )


def _mutate_pointer(
    manager: cache_module.CacheManager,
    mutation: object,
) -> None:
    pointer = json.loads(manager._manifest_path("step").read_text(encoding="utf-8"))
    mutation(pointer)
    _write_pointer(manager, pointer)


def _mutate_generation(
    manager: cache_module.CacheManager,
    mutation: object,
) -> Path:
    generation_dir = manager.split_dir("step", "train")
    generation_path = generation_dir / "generation.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    mutation(generation)
    signed = cache_module._signed_manifest(generation)
    raw = cache_module.stable_json_dumps(signed).encode("utf-8")
    generation_path.write_bytes(raw)

    pointer = json.loads(manager._manifest_path("step").read_text(encoding="utf-8"))
    pointer["splits"]["train"]["generation_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    _write_pointer(manager, pointer)
    return generation_path


def test_low_level_manifest_and_descriptor_guards(tmp_path: Path) -> None:
    valid = cache_module._signed_manifest(
        {
            "schema_version": 2,
            "cache_kind": "modssc.preprocess.step",
        }
    )
    wrong_kind = cache_module._signed_manifest({**valid, "cache_kind": "other"})
    with pytest.raises(PreprocessCacheError, match="Invalid test cache kind"):
        cache_module._verify_signed_manifest(wrong_kind, label="test")

    invalid_digest = {**valid, "manifest_sha256": "short"}
    with pytest.raises(PreprocessCacheError, match="Invalid test manifest digest"):
        cache_module._verify_signed_manifest(invalid_digest, label="test")

    class BadShape:
        shape = ("not-an-integer",)

    assert cache_module._value_shape(BadShape()) is None

    with pytest.raises(PreprocessCacheError, match="missing 'path'"):
        cache_module._authenticate_descriptor(tmp_path, {}, value=np.array([1]))
    with pytest.raises(PreprocessCacheError, match="escapes"):
        cache_module._authenticate_descriptor(
            tmp_path, {"path": "../outside.npy"}, value=np.array([1])
        )
    with pytest.raises(PreprocessCacheError, match="missing or is not a regular file"):
        cache_module._authenticate_descriptor(
            tmp_path, {"path": "missing.npy"}, value=np.array([1])
        )

    with pytest.raises(PreprocessCacheError, match="missing 'path'"):
        cache_module._verified_descriptor_path(tmp_path, {})
    with pytest.raises(PreprocessCacheError, match="must be relative"):
        cache_module._verified_descriptor_path(
            tmp_path,
            {"path": str((tmp_path / "absolute.npy").resolve())},
        )
    with pytest.raises(PreprocessCacheError, match="missing or is not a regular file"):
        cache_module._verified_descriptor_path(tmp_path, {"path": "missing.npy"})

    value_path = tmp_path / "value.npy"
    np.save(value_path, np.array([1], dtype=np.int64), allow_pickle=False)
    size = value_path.stat().st_size
    digest = cache_module._sha256_file(value_path)
    with pytest.raises(PreprocessCacheError, match="Invalid cached value sha256"):
        cache_module._verified_descriptor_path(
            tmp_path, {"path": value_path.name, "sha256": "bad", "size_bytes": size}
        )
    with pytest.raises(PreprocessCacheError, match="Invalid cached value size_bytes"):
        cache_module._verified_descriptor_path(
            tmp_path,
            {"path": value_path.name, "sha256": digest, "size_bytes": True},
        )
    with pytest.raises(PreprocessCacheError, match="Cached value size differs"):
        cache_module._verified_descriptor_path(
            tmp_path,
            {"path": value_path.name, "sha256": digest, "size_bytes": size + 1},
        )

    with pytest.raises(PreprocessCacheError, match="dtype differs"):
        cache_module._verify_loaded_value(
            np.array([1], dtype=np.int64),
            {"dtype": "float32", "shape": [1]},
            key="value",
        )
    with pytest.raises(PreprocessCacheError, match="Invalid cached value shape"):
        cache_module._verify_loaded_value(np.array([1]), {"shape": [True]}, key="value")
    with pytest.raises(PreprocessCacheError, match="shape differs"):
        cache_module._verify_loaded_value(np.array([1]), {"shape": [2]}, key="value")


def test_descriptor_detects_mutation_during_hash(tmp_path: Path, monkeypatch) -> None:
    value_path = tmp_path / "value.bin"
    value_path.write_bytes(b"aa")
    expected = "a" * 64

    def mutate_while_hashing(path: Path) -> str:
        path.write_bytes(b"bb")
        return expected

    monkeypatch.setattr(cache_module, "_sha256_file", mutate_while_hashing)
    with pytest.raises(PreprocessCacheError, match="changed while hashing"):
        cache_module._verified_descriptor_path(
            tmp_path,
            {"path": value_path.name, "sha256": expected, "size_bytes": 2},
        )


def test_split_dir_uses_historical_fallback_for_invalid_pointer(tmp_path: Path) -> None:
    manager = cache_module.CacheManager(root=tmp_path, dataset_fingerprint="dataset")
    assert manager.split_dir("step", "train") == manager.step_dir("step") / "train"

    manager.step_dir("step").mkdir(parents=True)
    _write_pointer(
        manager,
        {
            "schema_version": 2,
            "cache_kind": "modssc.preprocess.step",
            "dataset_fingerprint": "dataset",
            "step_fingerprint": "step",
            "splits": {"train": {"generation": 12}},
        },
    )
    assert manager.split_dir("step", "train") == manager.step_dir("step") / "train"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(dataset_fingerprint="other"), "dataset fingerprint differs"),
        (lambda value: value.update(step_fingerprint="other"), "step fingerprint differs"),
        (lambda value: value.update(splits=[]), "Invalid cache manifest structure"),
    ],
)
def test_pointer_identity_guards(tmp_path: Path, mutation: object, message: str) -> None:
    manager = _manager_with_value(tmp_path)
    _mutate_pointer(manager, mutation)
    with pytest.raises(PreprocessCacheError, match=message):
        manager.load_step_outputs(step_fingerprint="step", split="train")


def test_pointer_read_failure_is_normalized(tmp_path: Path) -> None:
    manager = _manager_with_value(tmp_path)
    with (
        patch.object(Path, "read_text", side_effect=UnicodeError("bad encoding")),
        pytest.raises(PreprocessCacheError, match="Invalid JSON manifest"),
    ):
        manager._read_pointer("step")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["splits"]["train"].update(generation=12),
            "Invalid preprocess cache generation",
        ),
        (
            lambda value: value["splits"]["train"].update(generation_manifest_sha256="bad"),
            "Invalid preprocess generation manifest digest",
        ),
    ],
)
def test_generation_pointer_guards(tmp_path: Path, mutation: object, message: str) -> None:
    manager = _manager_with_value(tmp_path)
    _mutate_pointer(manager, mutation)
    with pytest.raises(PreprocessCacheError, match=message):
        manager.load_step_outputs(step_fingerprint="step", split="train")


def test_generation_path_and_file_guards(tmp_path: Path) -> None:
    missing_dir_manager = _manager_with_value(tmp_path / "missing-dir")
    shutil.rmtree(missing_dir_manager.split_dir("step", "train"))
    with pytest.raises(PreprocessCacheError, match="generation is missing"):
        missing_dir_manager.load_step_outputs(step_fingerprint="step", split="train")

    missing_manifest_manager = _manager_with_value(tmp_path / "missing-manifest")
    generation_path = missing_manifest_manager.split_dir("step", "train") / "generation.json"
    generation_path.unlink()
    with pytest.raises(PreprocessCacheError, match="generation manifest is missing"):
        missing_manifest_manager.load_step_outputs(step_fingerprint="step", split="train")

    unreadable_manager = _manager_with_value(tmp_path / "unreadable")
    original_read_bytes = Path.read_bytes

    def fail_generation_read(path: Path) -> bytes:
        if path.name == "generation.json":
            raise OSError("unreadable")
        return original_read_bytes(path)

    with (
        patch.object(Path, "read_bytes", fail_generation_read),
        pytest.raises(PreprocessCacheError, match="manifest is unreadable"),
    ):
        unreadable_manager.load_step_outputs(step_fingerprint="step", split="train")


def test_generation_bytes_and_payload_guards(tmp_path: Path) -> None:
    hash_manager = _manager_with_value(tmp_path / "hash")
    hash_path = hash_manager.split_dir("step", "train") / "generation.json"
    hash_path.write_bytes(hash_path.read_bytes() + b" ")
    with pytest.raises(PreprocessCacheError, match="manifest file sha256 differs"):
        hash_manager.load_step_outputs(step_fingerprint="step", split="train")

    unicode_manager = _manager_with_value(tmp_path / "unicode")
    unicode_path = unicode_manager.split_dir("step", "train") / "generation.json"
    raw = b"\xff"
    unicode_path.write_bytes(raw)
    pointer = json.loads(unicode_manager._manifest_path("step").read_text(encoding="utf-8"))
    pointer["splits"]["train"]["generation_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    _write_pointer(unicode_manager, pointer)
    with pytest.raises(PreprocessCacheError, match="Invalid preprocess generation manifest"):
        unicode_manager.load_step_outputs(step_fingerprint="step", split="train")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(dataset_fingerprint="other"), "dataset fingerprint differs"),
        (lambda value: value.update(step_fingerprint="other"), "step fingerprint differs"),
        (lambda value: value.update(split="test"), "split differs"),
        (lambda value: value.update(saved=[]), "Invalid cache manifest structure"),
    ],
)
def test_generation_identity_guards(tmp_path: Path, mutation: object, message: str) -> None:
    manager = _manager_with_value(tmp_path)
    _mutate_generation(manager, mutation)
    with pytest.raises(PreprocessCacheError, match=message):
        manager.load_step_outputs(step_fingerprint="step", split="train")


def test_generation_saved_descriptor_and_path_change_guards(tmp_path: Path, monkeypatch) -> None:
    descriptor_manager = _manager_with_value(tmp_path / "descriptor")
    _mutate_generation(
        descriptor_manager,
        lambda value: value["saved"].update(value="not-a-mapping"),
    )
    with pytest.raises(PreprocessCacheError, match="Invalid cache manifest structure"):
        descriptor_manager.load_step_outputs(step_fingerprint="step", split="train")

    path_manager = _manager_with_value(tmp_path / "path")
    value_path = path_manager.split_dir("step", "train") / "value.npy"
    changed_path = value_path.with_name("changed.npy")
    calls = iter((value_path, changed_path))
    monkeypatch.setattr(
        cache_module, "_verified_descriptor_path", lambda *_args, **_kwargs: next(calls)
    )
    monkeypatch.setattr(cache_module, "_load_value", lambda *_args, **_kwargs: np.array([1, 2]))
    with pytest.raises(PreprocessCacheError, match="path changed while loading"):
        path_manager.load_step_outputs(step_fingerprint="step", split="train")


def test_duplicate_publication_is_idempotent_but_collision_fails(tmp_path: Path) -> None:
    manager = _manager_with_value(tmp_path)
    generations = manager.step_dir("step") / "generations"
    before = {path.name for path in generations.iterdir()}

    manager.save_step_outputs(
        step_fingerprint="step",
        split="train",
        produced={"value": np.array([1, 2], dtype=np.int64)},
        manifest={"producer": "test"},
    )
    assert {path.name for path in generations.iterdir()} == before
    assert not list(generations.glob(".staging-*"))

    with pytest.raises(PreprocessCacheError, match="Different outputs"):
        manager.save_step_outputs(
            step_fingerprint="step",
            split="train",
            produced={"value": np.array([2, 1], dtype=np.int64)},
            manifest={"producer": "test"},
        )
    assert not list(generations.glob(".staging-*"))
