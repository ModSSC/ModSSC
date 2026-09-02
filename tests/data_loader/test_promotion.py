from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

import modssc.data_loader as data_loader
import modssc.data_loader.api as api
import modssc.data_loader.promotion as promotion
from modssc.cli.datasets import app as datasets_app
from modssc.data_loader import cache, content
from modssc.data_loader.cache import CacheLayout
from modssc.data_loader.errors import CachePromotionError, ManifestError
from modssc.data_loader.manifest import Manifest, write_manifest
from modssc.data_loader.storage import FileStorage
from modssc.data_loader.types import LoadedDataset, Split


def _toy_staging(root: Path) -> tuple[CacheLayout, str]:
    api.download_dataset("toy", cache_dir=root, force=True)
    return CacheLayout(root), api.dataset_fingerprint("toy")


def _empty_destination(root: Path) -> CacheLayout:
    root.mkdir(parents=True)
    return CacheLayout(root)


def _expectation(fingerprint: str) -> promotion.CacheEntryExpectation:
    return promotion.CacheEntryExpectation(fingerprint=fingerprint)


def _promote(staging: CacheLayout, destination: CacheLayout, fingerprint: str, tx: str):
    return promotion.promote_cache_entries(
        staging_dir=staging.root,
        cache_dir=destination.root,
        entries=[_expectation(fingerprint)],
        transaction_id=tx,
    )


def test_promotes_main_manifest_last_and_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    observations: dict[str, bool] = {}

    def observe(phase: str) -> None:
        observations[phase] = destination.manifest_path(fingerprint).exists()

    monkeypatch.setattr(promotion, "_promotion_checkpoint", observe)
    report = _promote(staging, destination, fingerprint, "toy-transaction")

    assert observations == {
        "source-records": False,
        "processed": False,
        "content-manifests": False,
        "main-manifests": True,
        "live-rehash": True,
        "index": True,
        "receipt": True,
    }
    assert report.items[0].disposition == "promoted"
    assert report.items[0].dataset_id == "toy"
    assert report.receipt_path.is_file()
    assert not staging.manifest_path(fingerprint).exists()
    assert destination.manifest_path(fingerprint).is_file()
    assert cache.index_list(destination)[0]["fingerprint"] == fingerprint
    receipt_before = report.receipt_path.read_bytes()

    second = _promote(staging, destination, fingerprint, "toy-transaction")

    assert second.items[0].disposition == "reused"
    assert second.receipt_sha256 == report.receipt_sha256
    assert second.receipt_path.read_bytes() == receipt_before


def test_recovery_resumes_exact_intent_after_content_manifest(monkeypatch, tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    failed = False

    def fail_once(phase: str) -> None:
        nonlocal failed
        if phase == "content-manifests" and not failed:
            failed = True
            raise RuntimeError("injected crash")

    monkeypatch.setattr(promotion, "_promotion_checkpoint", fail_once)
    with pytest.raises(RuntimeError, match="injected crash"):
        _promote(staging, destination, fingerprint, "recoverable")

    assert destination.processed_dir(fingerprint).is_dir()
    assert destination.content_manifest_path(fingerprint).is_file()
    assert not destination.manifest_path(fingerprint).exists()
    assert not (destination.root / ".transactions/recoverable/receipt.json").exists()

    monkeypatch.setattr(promotion, "_promotion_checkpoint", lambda _phase: None)
    report = _promote(staging, destination, fingerprint, "recoverable")
    assert report.receipt_path.is_file()
    assert destination.manifest_path(fingerprint).is_file()


def test_preflight_rejects_unlisted_processed_file_before_intent(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    (staging.processed_dir(fingerprint) / "unlisted.bin").write_bytes(b"unexpected")

    with pytest.raises(CachePromotionError, match="inventory differs"):
        _promote(staging, destination, fingerprint, "extra-file")

    assert staging.manifest_path(fingerprint).is_file()
    assert not (destination.root / ".transactions/extra-file/intent.json").exists()


def test_preflight_rejects_hardlinked_processed_file(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    artifact = next(
        path for path in staging.processed_dir(fingerprint).rglob("*") if path.is_file()
    )
    os.link(artifact, tmp_path / "second-link")

    with pytest.raises(CachePromotionError, match="one hard link"):
        _promote(staging, destination, fingerprint, "hardlink")


def test_preflight_rejects_symlinked_processed_file(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    artifact = next(
        path for path in staging.processed_dir(fingerprint).rglob("*") if path.is_file()
    )
    payload = artifact.read_bytes()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(payload)
    artifact.unlink()
    artifact.symlink_to(outside)

    with pytest.raises(CachePromotionError, match="content preflight failed"):
        _promote(staging, destination, fingerprint, "symlink")


@pytest.mark.parametrize(
    "field",
    ["content_sha256", "content_manifest_sha256"],
)
def test_preflight_enforces_declared_digests(tmp_path: Path, field: str) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    expectation = _expectation(fingerprint)
    expectation = replace(expectation, **{field: "0" * 64})

    with pytest.raises(CachePromotionError, match="digest differs"):
        promotion.promote_cache_entries(
            staging_dir=staging.root,
            cache_dir=destination.root,
            entries=[expectation],
            transaction_id=f"wrong-{field}",
        )


def test_no_clobber_rejects_divergent_destination(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    destination.processed_dir(fingerprint).mkdir(parents=True)
    (destination.processed_dir(fingerprint) / "conflict.bin").write_bytes(b"different")

    with pytest.raises(CachePromotionError, match="conflicts with intent"):
        _promote(staging, destination, fingerprint, "no-clobber")

    assert staging.manifest_path(fingerprint).is_file()
    assert not destination.manifest_path(fingerprint).exists()


def test_preflight_rejects_incomplete_published_destination(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    destination.manifests_root.mkdir(parents=True)
    shutil.copy2(staging.manifest_path(fingerprint), destination.manifest_path(fingerprint))

    with pytest.raises(CachePromotionError, match="Published destination cache is incomplete"):
        _promote(staging, destination, fingerprint, "published-incomplete")

    assert staging.processed_dir(fingerprint).is_dir()
    assert staging.content_manifest_path(fingerprint).is_file()
    assert staging.manifest_path(fingerprint).is_file()
    assert not (destination.root / ".transactions/published-incomplete/intent.json").exists()


def test_exact_existing_entry_is_reused_without_deleting_staging(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    shutil.copytree(staging.processed_dir(fingerprint), destination.processed_dir(fingerprint))
    destination.manifests_root.mkdir(parents=True)
    shutil.copy2(
        staging.content_manifest_path(fingerprint), destination.content_manifest_path(fingerprint)
    )
    shutil.copy2(staging.manifest_path(fingerprint), destination.manifest_path(fingerprint))

    report = _promote(staging, destination, fingerprint, "already-there")

    assert report.items[0].disposition == "reused"
    assert staging.processed_dir(fingerprint).is_dir()
    assert staging.content_manifest_path(fingerprint).is_file()
    assert staging.manifest_path(fingerprint).is_file()


def _audio_staging(root: Path) -> tuple[CacheLayout, str, Path]:
    layout = CacheLayout(root)
    cache.ensure_layout(layout)
    fingerprint = "b" * 64
    identity = {
        "canonical_uri": "torchaudio:SPEECHCOMMANDS",
        "provider": "torchaudio",
        "dataset_id": "SPEECHCOMMANDS",
        "version": None,
        "modality": "audio",
        "task": "classification",
    }
    audio = layout.raw_dir("torchaudio", "SPEECHCOMMANDS", None) / "source/yes/sample.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFF-native-source")
    dataset = LoadedDataset(
        train=Split(
            X=np.asarray([str(audio)], dtype=object),
            y=np.asarray(["yes"], dtype=object),
        ),
        meta={"dataset_fingerprint": fingerprint},
    )
    FileStorage().save(layout.processed_dir(fingerprint), dataset)
    manifest = Manifest(
        schema_version=1,
        fingerprint=fingerprint,
        created_at="2026-08-31T00:00:00+00:00",
        identity=identity,
        dataset={},
        meta={},
        environment={"python": "3.12"},
    )
    write_manifest(layout.manifest_path(fingerprint), manifest)
    content_manifest = content.build_content_manifest(
        layout,
        fingerprint,
        dataset,
        identity=identity,
    )
    cache.atomic_write_text(
        layout.content_manifest_path(fingerprint),
        content.content_manifest_json(content_manifest),
    )
    return layout, fingerprint, audio


def test_promotes_native_source_records_and_rehashes_live_cache(tmp_path: Path) -> None:
    staging, fingerprint, staging_audio = _audio_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")

    report = _promote(staging, destination, fingerprint, "audio-source")

    live_audio = destination.raw_dir("torchaudio", "SPEECHCOMMANDS", None) / "source/yes/sample.wav"
    assert not staging_audio.exists()
    assert live_audio.read_bytes() == b"RIFF-native-source"
    assert report.items[0].source_file_count == 1
    assert (
        report.items[0].content_sha256
        == content.verify_content_manifest(
            destination,
            fingerprint,
            identity=cache.read_cached_manifest(destination, fingerprint).identity,
            rehash=True,
        )["content_sha256"]
    )


def test_recovery_reuses_source_moved_before_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging, fingerprint, staging_audio = _audio_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")

    def stop_after_source(phase: str) -> None:
        if phase == "source-records":
            raise RuntimeError("stop after source")

    monkeypatch.setattr(promotion, "_promotion_checkpoint", stop_after_source)
    with pytest.raises(RuntimeError, match="stop after source"):
        _promote(staging, destination, fingerprint, "audio-source-recovery")
    assert not staging_audio.exists()

    monkeypatch.setattr(promotion, "_promotion_checkpoint", lambda _phase: None)
    report = _promote(staging, destination, fingerprint, "audio-source-recovery")
    assert report.items[0].source_file_count == 1


def test_transaction_id_cannot_be_reused_for_a_different_request(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    _promote(staging, destination, fingerprint, "fixed-id")

    with pytest.raises(CachePromotionError, match="another promotion request"):
        promotion.promote_cache_entries(
            staging_dir=staging.root,
            cache_dir=destination.root,
            entries=[
                promotion.CacheEntryExpectation(
                    fingerprint=fingerprint,
                    content_sha256="0" * 64,
                )
            ],
            transaction_id="fixed-id",
        )


def test_atomic_index_failure_keeps_previous_index(tmp_path: Path) -> None:
    layout = CacheLayout(tmp_path / "cache")
    cache.ensure_layout(layout)
    before = layout.index_path.read_bytes()
    fingerprint = "c" * 64
    layout.processed_dir(fingerprint).mkdir(parents=True)
    layout.manifest_path(fingerprint).write_text("not-json", encoding="utf-8")

    with pytest.raises(ManifestError, match="Invalid published"):
        cache.rebuild_index_atomic(layout, strict=True)

    assert layout.index_path.read_bytes() == before
    assert not list(layout.root.glob(".index.sqlite.*.tmp"))


def test_public_exports_and_cli_repeated_fingerprints(tmp_path: Path) -> None:
    assert data_loader.promote_cache_entries is api.promote_cache_entries
    assert data_loader.CacheEntryExpectation is promotion.CacheEntryExpectation
    runner = CliRunner()
    first = "a" * 64
    second = "b" * 64
    report = SimpleNamespace(to_json=lambda: json.dumps({"items": 2}) + "\n")
    with patch("modssc.cli.datasets.api.promote_cache_entries", return_value=report) as called:
        result = runner.invoke(
            datasets_app,
            [
                "cache",
                "promote",
                "--staging-dir",
                str(tmp_path),
                "--cache-dir",
                str(tmp_path / "live"),
                "--transaction-id",
                "cli",
                "--fingerprint",
                first,
                "--fingerprint",
                second,
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"items": 2}
    assert [item.fingerprint for item in called.call_args.kwargs["entries"]] == [first, second]


def test_cli_all_staged_delegates_selection_to_native_api(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    fingerprint = "d" * 64
    (manifests / f"{fingerprint}.json").write_text("{}", encoding="utf-8")
    (manifests / f"{fingerprint}.content.json").write_text("{}", encoding="utf-8")
    (manifests / "notes.json").write_text("{}", encoding="utf-8")
    report = SimpleNamespace(to_json=lambda: json.dumps({"items": 1}) + "\n")
    with patch("modssc.cli.datasets.api.promote_cache_entries", return_value=report) as called:
        result = CliRunner().invoke(
            datasets_app,
            [
                "cache",
                "promote",
                "--staging-dir",
                str(tmp_path),
                "--cache-dir",
                str(tmp_path / "live"),
                "--transaction-id",
                "cli-all",
                "--all-staged",
            ],
        )

    assert result.exit_code == 0, result.output
    assert called.call_args.kwargs["entries"] is None


def test_cli_all_staged_replays_completed_transaction(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    arguments = [
        "cache",
        "promote",
        "--staging-dir",
        str(staging.root),
        "--cache-dir",
        str(destination.root),
        "--transaction-id",
        "all-complete",
        "--all-staged",
    ]

    first = CliRunner().invoke(datasets_app, arguments)
    assert first.exit_code == 0, first.output
    assert not staging.manifest_path(fingerprint).exists()

    replay = CliRunner().invoke(datasets_app, arguments)
    assert replay.exit_code == 0, replay.output
    assert json.loads(replay.stdout)["items"][0]["disposition"] == "reused"


def test_cli_all_staged_recovers_after_main_manifest_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    failed = False

    def fail_after_main(phase: str) -> None:
        nonlocal failed
        if phase == "main-manifests" and not failed:
            failed = True
            raise RuntimeError("crash after publication")

    monkeypatch.setattr(promotion, "_promotion_checkpoint", fail_after_main)
    arguments = [
        "cache",
        "promote",
        "--staging-dir",
        str(staging.root),
        "--cache-dir",
        str(destination.root),
        "--transaction-id",
        "all-crash",
        "--all-staged",
    ]
    first = CliRunner().invoke(datasets_app, arguments)
    assert first.exit_code == 1
    assert isinstance(first.exception, RuntimeError)
    assert destination.manifest_path(fingerprint).is_file()
    assert not staging.manifest_path(fingerprint).exists()
    assert (destination.root / ".transactions/all-crash/intent.json").is_file()
    assert not (destination.root / ".transactions/all-crash/receipt.json").exists()

    monkeypatch.setattr(promotion, "_promotion_checkpoint", lambda _phase: None)
    replay = CliRunner().invoke(datasets_app, arguments)
    assert replay.exit_code == 0, replay.output
    assert json.loads(replay.stdout)["items"][0]["fingerprint"] == fingerprint


def test_receipt_hash_matches_immutable_bytes(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")

    report = _promote(staging, destination, fingerprint, "receipt-hash")

    assert hashlib.sha256(report.receipt_path.read_bytes()).hexdigest() == report.receipt_sha256


def test_tampered_receipt_is_rejected_on_idempotent_replay(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    report = _promote(staging, destination, fingerprint, "receipt-tamper")
    receipt = json.loads(report.receipt_path.read_text(encoding="utf-8"))
    receipt["items"][0]["dataset_id"] = "tampered"
    report.receipt_path.write_bytes(promotion._canonical_json(receipt) + b"\n")

    with pytest.raises(CachePromotionError, match="receipt evidence differs"):
        _promote(staging, destination, fingerprint, "receipt-tamper")


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("extra-key", "receipt keys differ"),
        ("missing-key", "receipt keys differ"),
        ("schema", "receipt schema"),
        ("completed-invalid", "completion time is invalid"),
        ("completed-offset", "completion time is not canonical UTC"),
    ],
)
def test_receipt_rejects_noncanonical_envelope(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    report = _promote(staging, destination, fingerprint, f"receipt-{case}")
    receipt = json.loads(report.receipt_path.read_text(encoding="utf-8"))
    if case == "extra-key":
        receipt["unexpected"] = True
    elif case == "missing-key":
        del receipt["completed_at"]
    elif case == "schema":
        receipt["schema_version"] = True
    elif case == "completed-invalid":
        receipt["completed_at"] = "not-a-time"
    else:
        receipt["completed_at"] = "2026-08-31T02:00:00+02:00"
    report.receipt_path.write_bytes(promotion._canonical_json(receipt) + b"\n")

    with pytest.raises(CachePromotionError, match=match):
        _promote(staging, destination, fingerprint, f"receipt-{case}")


def test_receipt_rejects_noncanonical_json_bytes(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    report = _promote(staging, destination, fingerprint, "receipt-json")
    receipt = json.loads(report.receipt_path.read_text(encoding="utf-8"))
    report.receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(CachePromotionError, match="not canonical JSON"):
        _promote(staging, destination, fingerprint, "receipt-json")


def test_receipt_rehashes_current_live_index(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    _promote(staging, destination, fingerprint, "receipt-index")
    destination.index_path.write_bytes(destination.index_path.read_bytes() + b"tampered")

    with pytest.raises(CachePromotionError, match="index digest differs from live index"):
        _promote(staging, destination, fingerprint, "receipt-index")


def _stat_like(value: os.stat_result, **changes: int) -> SimpleNamespace:
    fields = {
        "st_dev": value.st_dev,
        "st_ino": value.st_ino,
        "st_mode": value.st_mode,
        "st_uid": value.st_uid,
        "st_gid": value.st_gid,
        "st_nlink": value.st_nlink,
        "st_size": value.st_size,
        "st_mtime_ns": value.st_mtime_ns,
        "st_ctime_ns": value.st_ctime_ns,
    }
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_secure_reader_supports_platform_without_nofollow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"payload")
    monkeypatch.delattr(promotion.os, "O_NOFOLLOW")

    payload, snapshot = promotion._read_regular_file(artifact)

    assert payload == b"payload"
    assert snapshot.sha256 == hashlib.sha256(payload).hexdigest()


def test_secure_reader_rejects_open_error_and_non_regular_file(tmp_path: Path) -> None:
    with (
        patch("modssc.data_loader.promotion.os.open", side_effect=PermissionError),
        pytest.raises(CachePromotionError, match="Cannot securely open"),
    ):
        promotion._read_regular_file(tmp_path / "missing")

    with pytest.raises(CachePromotionError, match="not a regular file"):
        promotion._read_regular_file(tmp_path)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("opening-missing", "disappeared while reading"),
        ("opening-changed", "changed while opening"),
        ("reading-changed", "changed while reading"),
        ("ending-missing", "disappeared while reading"),
        ("ending-replaced", "replaced while reading"),
    ],
)
def test_secure_reader_rejects_concurrent_path_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"payload")
    original_lstat = os.lstat
    original_fstat = os.fstat
    observed = original_lstat(artifact)
    changed = _stat_like(observed, st_ino=observed.st_ino + 1)
    lstat_calls = 0
    fstat_calls = 0

    def fake_lstat(path: os.PathLike[str] | str) -> os.stat_result | SimpleNamespace:
        nonlocal lstat_calls
        lstat_calls += 1
        if case == "opening-missing" and lstat_calls == 1:
            raise FileNotFoundError
        if case == "opening-changed" and lstat_calls == 1:
            return changed
        if case == "ending-missing" and lstat_calls == 2:
            raise FileNotFoundError
        if case == "ending-replaced" and lstat_calls == 2:
            return changed
        return original_lstat(path)

    def fake_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal fstat_calls
        fstat_calls += 1
        value = original_fstat(descriptor)
        if case == "reading-changed" and fstat_calls == 2:
            return _stat_like(value, st_size=value.st_size + 1)
        return value

    monkeypatch.setattr(promotion.os, "lstat", fake_lstat)
    monkeypatch.setattr(promotion.os, "fstat", fake_fstat)
    with pytest.raises(CachePromotionError, match=match):
        promotion._read_regular_file(artifact)


def test_canonical_root_rejects_missing_symlink_uninspectable_and_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(CachePromotionError, match="must already exist"):
        promotion._canonical_root(tmp_path / "missing", label="root")

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(CachePromotionError, match="symlink components"):
        promotion._canonical_root(link, label="root")

    with (
        patch.object(Path, "resolve", return_value=target),
        patch("modssc.data_loader.promotion.os.lstat", side_effect=PermissionError),
        pytest.raises(CachePromotionError, match="Cannot inspect"),
    ):
        promotion._canonical_root(target, label="root")

    artifact = tmp_path / "file"
    artifact.write_bytes(b"x")
    with pytest.raises(CachePromotionError, match="is not a directory"):
        promotion._canonical_root(artifact, label="root")


@pytest.mark.parametrize("value", ["", "/absolute", "safe/../escape"])
def test_relative_path_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(CachePromotionError, match="Invalid test relative path"):
        promotion._relative_path(value, purpose="test")


def test_relative_to_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(CachePromotionError, match="escapes cache root"):
        promotion._relative_to(tmp_path.parent, tmp_path, purpose="test")


def test_symlink_chain_rejects_missing_inspection_error_and_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(CachePromotionError, match="artifact is missing"):
        promotion._assert_no_symlink_chain(tmp_path, Path("missing"), allow_missing=False)

    with (
        patch("modssc.data_loader.promotion.os.lstat", side_effect=PermissionError),
        pytest.raises(CachePromotionError, match="Cannot inspect cache path"),
    ):
        promotion._assert_no_symlink_chain(tmp_path, Path("blocked"), allow_missing=False)

    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "link").symlink_to(target)
    with pytest.raises(CachePromotionError, match="contains a symlink"):
        promotion._assert_no_symlink_chain(tmp_path, Path("link"), allow_missing=False)


def test_scan_directory_rejects_missing_file_nested_fs_and_symlinked_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    missing = root / "missing"
    monkeypatch.setattr(promotion, "_assert_no_symlink_chain", lambda *_args, **_kwargs: None)
    with pytest.raises(CachePromotionError, match="directory is missing"):
        promotion._scan_directory(missing, cache_root=root)

    artifact = root / "file"
    artifact.write_bytes(b"x")
    with pytest.raises(CachePromotionError, match="is not a directory"):
        promotion._scan_directory(artifact, cache_root=root)

    directory = root / "directory"
    directory.mkdir()
    original_lstat = os.lstat

    def nested_lstat(path: os.PathLike[str] | str):
        value = original_lstat(path)
        if Path(path) == root:
            return _stat_like(value, st_dev=value.st_dev + 1)
        return value

    monkeypatch.setattr(promotion.os, "lstat", nested_lstat)
    with pytest.raises(CachePromotionError, match="nested filesystem"):
        promotion._scan_directory(directory, cache_root=root)
    monkeypatch.setattr(promotion.os, "lstat", original_lstat)

    child_target = root / "child-target"
    child_target.mkdir()
    (directory / "link").symlink_to(child_target)
    with pytest.raises(CachePromotionError, match="contains a non-directory"):
        promotion._scan_directory(directory, cache_root=root)


def test_scan_directory_accepts_real_subdirectories(tmp_path: Path) -> None:
    root = tmp_path / "root"
    child = root / "processed" / "nested"
    child.mkdir(parents=True)
    (child / "data.bin").write_bytes(b"data")

    assert promotion._scan_directory(root / "processed", cache_root=root) == [
        {
            "path": "nested/data.bin",
            "sha256": hashlib.sha256(b"data").hexdigest(),
            "size_bytes": 4,
        }
    ]


def test_expectation_validation_rejects_invalid_duplicate_and_empty() -> None:
    with pytest.raises(CachePromotionError, match="content_sha256"):
        promotion._expect_sha256("BAD", field="content_sha256")
    with pytest.raises(CachePromotionError, match="fingerprint must be"):
        promotion._expectation_payload([promotion.CacheEntryExpectation("BAD")])
    valid = promotion.CacheEntryExpectation("a" * 64)
    with pytest.raises(CachePromotionError, match="Duplicate"):
        promotion._expectation_payload([valid, valid])
    with pytest.raises(CachePromotionError, match="At least one"):
        promotion._expectation_payload([])


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda value: b"not-json", "Invalid main dataset manifest"),
        (
            lambda value: json.dumps({**value, "schema_version": 2}).encode(),
            "Unsupported main dataset manifest schema",
        ),
        (
            lambda value: json.dumps({**value, "fingerprint": "b" * 64}).encode(),
            "Main manifest fingerprint differs",
        ),
        (
            lambda value: json.dumps(
                {**value, "identity": {**value["identity"], "provider": ""}}
            ).encode(),
            "invalid identity field",
        ),
    ],
)
def test_main_manifest_validation_rejects_invalid_variants(
    tmp_path: Path,
    mutator,
    match: str,
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    value = json.loads(staging.manifest_path(fingerprint).read_text(encoding="utf-8"))

    with pytest.raises(CachePromotionError, match=match):
        promotion._parse_main_manifest(
            mutator(value),
            fingerprint=fingerprint,
            path=staging.manifest_path(fingerprint),
        )


def test_file_plan_rejects_nested_filesystem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "file"
    artifact.write_bytes(b"x")
    _, snapshot = promotion._read_regular_file(artifact)
    monkeypatch.setattr(
        promotion.os,
        "lstat",
        lambda _path: _stat_like(os.stat(root), st_dev=snapshot.device + 1),
    )

    with pytest.raises(CachePromotionError, match="nested filesystem"):
        promotion._file_plan(artifact, root=root, snapshot=snapshot)


def _entry_plan(
    staging: CacheLayout,
    destination: CacheLayout,
    fingerprint: str,
) -> dict:
    request = promotion._expectation_payload([_expectation(fingerprint)])
    return promotion._build_entry_plan(staging, destination, request[0])


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"not-json", "Invalid dataset content manifest"),
        (b"[]", "root is not a mapping"),
    ],
)
def test_entry_plan_rejects_invalid_content_manifest(
    tmp_path: Path,
    payload: bytes,
    match: str,
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    staging.content_manifest_path(fingerprint).write_bytes(payload)

    with pytest.raises(CachePromotionError, match=match):
        _entry_plan(staging, destination, fingerprint)


def test_entry_plan_rejects_manifest_changed_during_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    verify = content.verify_content_manifest

    def changed(*args, **kwargs):
        evidence = verify(*args, **kwargs)
        return {**evidence, "content_manifest_sha256": "0" * 64}

    monkeypatch.setattr(promotion.content, "verify_content_manifest", changed)
    with pytest.raises(CachePromotionError, match="changed during preflight"):
        _entry_plan(staging, destination, fingerprint)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("size_bytes", "size differs"),
        ("sha256", "storage digest differs"),
    ],
)
def test_entry_plan_rejects_processed_inventory_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    match: str,
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    scan = promotion._scan_directory

    def changed(*args, **kwargs):
        records = scan(*args, **kwargs)
        replacement = "0" * 64 if field == "sha256" else records[0][field] + 1
        records[0] = {**records[0], field: replacement}
        return records

    monkeypatch.setattr(promotion, "_scan_directory", changed)
    with pytest.raises(CachePromotionError, match=match):
        _entry_plan(staging, destination, fingerprint)


def test_source_plan_rejects_missing_native_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging, fingerprint, _ = _audio_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    manifest = cache.read_cached_manifest(staging, fingerprint)
    evidence = content.verify_content_manifest(
        staging,
        fingerprint,
        identity=manifest.identity,
        rehash=True,
    )
    monkeypatch.setattr(
        promotion.content,
        "verify_content_manifest",
        lambda *_args, **_kwargs: evidence,
    )
    monkeypatch.setattr(promotion.content, "_source_root", lambda *_args, **_kwargs: None)

    with pytest.raises(CachePromotionError, match="no native cache root"):
        _entry_plan(staging, destination, fingerprint)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("device", "nested filesystem"),
        ("sha256", "differs from content manifest"),
    ],
)
def test_source_plan_rejects_changed_source_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: str,
    match: str,
) -> None:
    staging, fingerprint, source = _audio_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    secure_read = promotion._read_regular_file

    def changed(path: Path, *, capture_bytes: bool = True):
        payload, snapshot = secure_read(path, capture_bytes=capture_bytes)
        if Path(path) == source:
            if change == "device":
                snapshot = replace(snapshot, device=snapshot.device + 1)
            else:
                snapshot = replace(snapshot, sha256="0" * 64)
        return payload, snapshot

    monkeypatch.setattr(promotion, "_read_regular_file", changed)
    with pytest.raises(CachePromotionError, match=match):
        _entry_plan(staging, destination, fingerprint)


def test_intent_rejects_source_destination_disagreement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = CacheLayout(tmp_path / "staging")
    destination = CacheLayout(tmp_path / "live")
    first = {
        "fingerprint": "a" * 64,
        "source_files": [{"destination": "raw/shared", "sha256": "1" * 64, "size_bytes": 1}],
    }
    second = {
        "fingerprint": "b" * 64,
        "source_files": [{"destination": "raw/shared", "sha256": "2" * 64, "size_bytes": 1}],
    }
    plans = iter([first, second])
    monkeypatch.setattr(promotion, "_build_entry_plan", lambda *_args: next(plans))
    monkeypatch.setattr(
        promotion,
        "_reject_incomplete_published_destination",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(CachePromotionError, match="Source records disagree"):
        promotion._build_intent(
            staging=staging,
            destination=destination,
            transaction_id="conflict",
            request=[{"fingerprint": "a" * 64}, {"fingerprint": "b" * 64}],
        )


def _copy_plan_part(
    staging: CacheLayout,
    destination: CacheLayout,
    part: dict,
    *,
    destination_key: str = "path",
) -> None:
    source = staging.root / part["path"]
    target = destination.root / part[destination_key]
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def test_published_destination_requires_content_manifest(tmp_path: Path) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    plan = _entry_plan(staging, destination, fingerprint)
    _copy_plan_part(staging, destination, plan["processed"], destination_key="destination")
    _copy_plan_part(staging, destination, plan["main_manifest"])

    with pytest.raises(CachePromotionError, match="incomplete"):
        promotion._reject_incomplete_published_destination(plan, destination=destination)


def test_published_destination_requires_and_accepts_source_records(tmp_path: Path) -> None:
    staging, fingerprint, _ = _audio_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    plan = _entry_plan(staging, destination, fingerprint)
    _copy_plan_part(staging, destination, plan["processed"], destination_key="destination")
    _copy_plan_part(staging, destination, plan["content_manifest"])
    _copy_plan_part(staging, destination, plan["main_manifest"])

    with pytest.raises(CachePromotionError, match="incomplete"):
        promotion._reject_incomplete_published_destination(plan, destination=destination)

    for source in plan["source_files"]:
        _copy_plan_part(staging, destination, source, destination_key="destination")
    promotion._reject_incomplete_published_destination(plan, destination=destination)


def test_write_exclusive_rejects_missing_existing_and_unwritable_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(CachePromotionError, match="directory is missing"):
        promotion._write_exclusive(tmp_path / "missing" / "record", b"x")

    existing = tmp_path / "record"
    existing.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        promotion._write_exclusive(existing, b"new")

    target = tmp_path / "new-record"
    with (
        patch("modssc.data_loader.promotion.os.open", side_effect=PermissionError),
        pytest.raises(CachePromotionError, match="Cannot create immutable"),
    ):
        promotion._write_exclusive(target, b"x")

    monkeypatch.delattr(promotion.os, "O_NOFOLLOW")
    fallback = tmp_path / "fallback"
    promotion._write_exclusive(fallback, b"payload")
    assert fallback.read_bytes() == b"payload"


@pytest.mark.parametrize(
    ("payload", "match"),
    [(b"not-json", "Invalid record"), (b"[]", "root is not a mapping")],
)
def test_json_record_rejects_invalid_payload(tmp_path: Path, payload: bytes, match: str) -> None:
    path = tmp_path / "record"
    path.write_bytes(payload)
    with pytest.raises(CachePromotionError, match=match):
        promotion._read_json_record(path, label="record")


def test_file_lock_supports_fallback_and_rejects_open_or_hardlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delattr(promotion.os, "O_NOFOLLOW")
    with promotion._file_lock(tmp_path / "fallback.lock"):
        pass

    with (
        patch("modssc.data_loader.promotion.os.open", side_effect=PermissionError),
        pytest.raises(CachePromotionError, match="Cannot securely open cache lock"),
        promotion._file_lock(tmp_path / "blocked.lock"),
    ):
        pass

    first = tmp_path / "hardlink.lock"
    first.write_bytes(b"")
    os.link(first, tmp_path / "hardlink-copy.lock")
    with (
        pytest.raises(CachePromotionError, match="single-link regular file"),
        promotion._file_lock(first),
    ):
        pass


def _intent_envelope(staging: CacheLayout, destination: CacheLayout, transaction_id: str) -> dict:
    return {
        "cache_root": str(destination.root),
        "entries": [{"fingerprint": "a" * 64}],
        "request": promotion._expectation_payload([promotion.CacheEntryExpectation("a" * 64)]),
        "schema_version": 1,
        "staging_root": str(staging.root),
        "transaction_id": transaction_id,
    }


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("keys", "keys differ"),
        ("schema", "intent schema"),
        ("transaction", "transaction differs"),
        ("staging", "staging root differs"),
        ("destination", "destination root differs"),
    ],
)
def test_intent_envelope_rejects_tampering(tmp_path: Path, case: str, match: str) -> None:
    staging = CacheLayout(tmp_path / "staging")
    destination = CacheLayout(tmp_path / "live")
    intent = _intent_envelope(staging, destination, "transaction")
    if case == "keys":
        intent["extra"] = True
    elif case == "schema":
        intent["schema_version"] = True
    elif case == "transaction":
        intent["transaction_id"] = "other"
    elif case == "staging":
        intent["staging_root"] = "other"
    else:
        intent["cache_root"] = "other"

    with pytest.raises(CachePromotionError, match=match):
        promotion._validate_intent_envelope(
            intent,
            staging=staging,
            destination=destination,
            transaction_id="transaction",
        )


@pytest.mark.parametrize(
    ("raw_request", "match"),
    [
        (None, "request is empty"),
        ([], "request is empty"),
        ([{"fingerprint": "a" * 64}], "item keys differ"),
        (
            [
                {
                    "content_manifest_sha256": None,
                    "content_sha256": None,
                    "fingerprint": "b" * 64,
                },
                {
                    "content_manifest_sha256": None,
                    "content_sha256": None,
                    "fingerprint": "a" * 64,
                },
            ],
            "request is not canonical",
        ),
    ],
)
def test_intent_request_rejects_invalid_or_noncanonical(raw_request, match: str) -> None:
    with pytest.raises(CachePromotionError, match=match):
        promotion._normalize_intent_request(raw_request)


def test_file_state_rejects_conflicting_existing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"actual")
    with pytest.raises(CachePromotionError, match="conflicts with intent"):
        promotion._file_state(
            path,
            root=tmp_path,
            expected={"size_bytes": 1, "sha256": "0" * 64},
        )


def test_mkdir_safe_rejects_nested_filesystem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent = tmp_path / "parent"
    original_lstat = os.lstat

    def changed(path: os.PathLike[str] | str):
        value = original_lstat(path)
        if Path(path) == parent:
            return _stat_like(value, st_dev=value.st_dev + 1)
        return value

    monkeypatch.setattr(promotion.os, "lstat", changed)
    with pytest.raises(CachePromotionError, match="nested filesystem"):
        promotion._mkdir_safe(tmp_path, parent)


class _FakeRename:
    def __init__(self, result: int):
        self.result = result
        self.argtypes = None
        self.restype = None

    def __call__(self, *_args) -> int:
        return self.result


def _force_linux_rename_einval(monkeypatch: pytest.MonkeyPatch) -> None:
    rename = _FakeRename(-1)
    monkeypatch.setattr(promotion.sys, "platform", "linux")
    monkeypatch.setattr(
        promotion.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(renameat2=rename),
    )
    monkeypatch.setattr(promotion.ctypes, "get_errno", lambda: 22)


def test_linux_atomic_rename_rejects_missing_primitive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(promotion.sys, "platform", "linux")
    monkeypatch.setattr(
        promotion.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(renameat2=None),
    )
    with pytest.raises(CachePromotionError, match="renameat2"):
        promotion._atomic_rename_noreplace(tmp_path / "source", tmp_path / "destination")


@pytest.mark.parametrize(
    ("result", "error", "match"),
    [
        (0, 0, None),
        (-1, 17, "appeared concurrently"),
        (-1, 5, "Atomic cache promotion failed"),
    ],
)
def test_linux_atomic_rename_result_handling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result: int,
    error: int,
    match: str | None,
) -> None:
    rename = _FakeRename(result)
    monkeypatch.setattr(promotion.sys, "platform", "linux")
    monkeypatch.setattr(
        promotion.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(renameat2=rename),
    )
    monkeypatch.setattr(promotion.ctypes, "get_errno", lambda: error)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    if match is None:
        promotion._atomic_rename_noreplace(source, destination)
        assert rename.argtypes is not None
        assert rename.restype is promotion.ctypes.c_int
    else:
        with pytest.raises(CachePromotionError, match=match):
            promotion._atomic_rename_noreplace(source, destination)


def test_einval_fallback_publishes_exclusively_with_main_manifest_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    observations: dict[str, bool] = {}
    _force_linux_rename_einval(monkeypatch)
    monkeypatch.setattr(
        promotion.os,
        "rename",
        lambda *_args, **_kwargs: pytest.fail("fallback must not use os.rename"),
    )
    monkeypatch.setattr(
        promotion,
        "_promotion_checkpoint",
        lambda phase: observations.setdefault(
            phase,
            destination.manifest_path(fingerprint).exists(),
        ),
    )

    report = _promote(staging, destination, fingerprint, "lustre-einval")

    assert observations["processed"] is False
    assert observations["content-manifests"] is False
    assert observations["main-manifests"] is True
    assert report.items[0].disposition == "promoted"
    assert staging.manifest_path(fingerprint).is_file()
    assert destination.manifest_path(fingerprint).is_file()
    assert all(path.stat().st_nlink == 1 for path in destination.root.rglob("*") if path.is_file())


def test_einval_fallback_resumes_claimed_partial_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    _force_linux_rename_einval(monkeypatch)
    original = promotion._publish_file_fallback
    interrupted = False

    def interrupt_after_one(*args, **kwargs):
        nonlocal interrupted
        published = original(*args, **kwargs)
        target = Path(kwargs["destination"])
        if destination.processed_dir(fingerprint) in target.parents and not interrupted:
            interrupted = True
            raise RuntimeError("injected fallback interruption")
        return published

    monkeypatch.setattr(promotion, "_publish_file_fallback", interrupt_after_one)
    with pytest.raises(RuntimeError, match="fallback interruption"):
        _promote(staging, destination, fingerprint, "lustre-replay")

    assert destination.processed_dir(fingerprint).is_dir()
    assert any(destination.processed_dir(fingerprint).rglob("*"))
    assert not destination.manifest_path(fingerprint).exists()
    assert list(
        (destination.root / ".transactions/lustre-replay/publish").glob("*.tree-claim.json")
    )

    monkeypatch.setattr(promotion, "_publish_file_fallback", original)
    report = _promote(staging, destination, fingerprint, "lustre-replay")
    assert report.receipt_path.is_file()
    assert destination.manifest_path(fingerprint).is_file()


def test_fallback_recovers_linked_temp_after_interruption(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    destination_root = tmp_path / "live"
    publish_root = destination_root / ".transactions/replay/publish"
    source = staging_root / "source.bin"
    destination = destination_root / "processed/artifact.bin"
    source.parent.mkdir(parents=True)
    destination.parent.mkdir(parents=True)
    publish_root.mkdir(parents=True)
    payload = b"interrupted-exclusive-publication"
    source.write_bytes(payload)
    plan = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    temp = promotion._publish_temp_path(
        publish_root,
        destination,
        destination_root=destination_root,
        expected=plan,
    )
    promotion._copy_regular_file_exclusive(source, temp, expected=plan)
    os.link(temp, destination, follow_symlinks=False)
    assert temp.stat().st_nlink == destination.stat().st_nlink == 2

    promotion._reconcile_publish_temp(
        temp,
        destination,
        destination_root=destination_root,
        expected=plan,
    )

    assert not temp.exists()
    assert destination.stat().st_nlink == 1
    assert promotion._file_state(destination, root=destination_root, expected=plan) == "exact"


def test_directory_progress_rejects_invalid_partial_and_unexpected_trees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    directory = root / "processed"
    directory.mkdir(parents=True)
    record = {
        "path": "nested/artifact.bin",
        "sha256": hashlib.sha256(b"payload").hexdigest(),
        "size_bytes": 7,
    }

    with pytest.raises(CachePromotionError, match="intent files are invalid"):
        promotion._directory_progress(directory, root=root, expected={"files": None})
    with pytest.raises(CachePromotionError, match="duplicate files"):
        promotion._directory_progress(
            directory,
            root=root,
            expected={"files": [record, record]},
        )
    with pytest.raises(CachePromotionError, match="conflicts with intent"):
        promotion._directory_state(directory, root=root, expected={"files": [record]})

    artifact = directory / record["path"]
    artifact.parent.mkdir()
    artifact.write_bytes(b"payload")
    assert (
        promotion._directory_progress(directory, root=root, expected={"files": [record]}) == "exact"
    )
    (directory / "unexpected").mkdir()
    with pytest.raises(CachePromotionError, match="conflicts with intent"):
        promotion._directory_progress(directory, root=root, expected={"files": [record]})
    (directory / "unexpected").rmdir()

    original_is_dir = Path.is_dir
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda self: False if self == artifact.parent else original_is_dir(self),
    )
    with pytest.raises(CachePromotionError, match="conflicts with intent"):
        promotion._directory_progress(directory, root=root, expected={"files": [record]})


def test_tree_claim_reuse_and_conflicts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    claim = tmp_path / "claim.json"
    expected = {"destination": "processed/item", "plan_sha256": "a" * 64, "schema_version": 1}
    assert promotion._tree_claim_state(claim, expected=expected) == "absent"
    assert promotion._write_tree_claim(claim, expected=expected) is True
    assert promotion._write_tree_claim(claim, expected=expected) is False

    conflict = tmp_path / "conflict.json"
    promotion._write_exclusive(conflict, promotion._canonical_json({"different": True}) + b"\n")
    with pytest.raises(CachePromotionError, match="claim differs"):
        promotion._tree_claim_state(conflict, expected=expected)

    monkeypatch.setattr(promotion, "_tree_claim_state", lambda *_args, **_kwargs: "absent")
    with pytest.raises(CachePromotionError, match="failed verification"):
        promotion._write_tree_claim(claim, expected=expected)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("unsafe", "unsafe"),
        ("nested", "nested filesystem"),
        ("wrong-linked-destination", "unexpected hard link"),
        ("linked-elsewhere", "unexpected hard link"),
        ("destination-vanished", "failed verification"),
    ],
)
def test_reconcile_publish_temp_rejects_unsafe_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    root = tmp_path / "live"
    publish = root / ".transactions/tx/publish"
    destination = root / "processed/artifact.bin"
    temp = publish / "temp.partial"
    publish.mkdir(parents=True)
    destination.parent.mkdir(parents=True)
    payload = b"payload"
    expected = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    if case == "unsafe":
        temp.mkdir()
    else:
        temp.write_bytes(payload)
    if case == "nested":
        original_lstat = os.lstat

        def nested_lstat(path):
            value = original_lstat(path)
            return _stat_like(value, st_dev=value.st_dev + 1) if Path(path) == temp else value

        monkeypatch.setattr(promotion.os, "lstat", nested_lstat)
    elif case == "wrong-linked-destination":
        os.link(temp, tmp_path / "other-link")
        destination.write_bytes(payload)
    elif case == "linked-elsewhere":
        os.link(temp, tmp_path / "other-link")
    elif case == "destination-vanished":
        destination.write_bytes(payload)
        monkeypatch.setattr(promotion, "_file_state", lambda *_args, **_kwargs: "absent")

    with pytest.raises(CachePromotionError, match=match):
        promotion._reconcile_publish_temp(
            temp,
            destination,
            destination_root=root,
            expected=expected,
        )


def test_reconcile_publish_temp_cleans_stale_exact_temp(tmp_path: Path) -> None:
    root = tmp_path / "live"
    publish = root / ".transactions/tx/publish"
    destination = root / "processed/artifact.bin"
    temp = publish / "temp.partial"
    publish.mkdir(parents=True)
    destination.parent.mkdir(parents=True)
    payload = b"payload"
    temp.write_bytes(payload)
    destination.write_bytes(payload)
    expected = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}

    promotion._reconcile_publish_temp(
        temp,
        destination,
        destination_root=root,
        expected=expected,
    )

    assert not temp.exists()

    lonely_temp = publish / "lonely.partial"
    lonely_destination = root / "processed/lonely.bin"
    lonely_temp.write_bytes(payload)
    promotion._reconcile_publish_temp(
        lonely_temp,
        lonely_destination,
        destination_root=root,
        expected=expected,
    )
    assert lonely_temp.is_file()


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing", "securely open"),
        ("unsafe-source", "source is unsafe"),
        ("changed-opening", "changed while opening"),
        ("temp-exists", "create publish temporary"),
        ("write-zero", "Cannot write"),
        ("unsafe-cleanup", "Cannot write"),
        ("changed-copying", "changed while copying"),
        ("replaced-copying", "replaced while copying"),
        ("digest", "differs from intent"),
    ],
)
def test_copy_regular_file_exclusive_rejects_changes_and_io_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    source = tmp_path / "source.bin"
    temp = tmp_path / "temp.partial"
    payload = b"payload"
    if case != "missing":
        source.write_bytes(payload)
    expected = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    if case == "unsafe-source":
        os.link(source, tmp_path / "other-link")
    elif case == "changed-opening":
        original_lstat = os.lstat

        def changed_opening(path):
            value = original_lstat(path)
            return _stat_like(value, st_ino=value.st_ino + 1) if Path(path) == source else value

        monkeypatch.setattr(promotion.os, "lstat", changed_opening)
    elif case == "temp-exists":
        temp.write_bytes(b"already here")
    elif case == "write-zero":
        monkeypatch.setattr(promotion.os, "write", lambda *_args, **_kwargs: 0)
    elif case == "unsafe-cleanup":
        original_lstat = os.lstat
        monkeypatch.setattr(promotion.os, "write", lambda *_args, **_kwargs: 0)

        def unsafe_cleanup(path):
            value = original_lstat(path)
            if Path(path) == temp:
                return _stat_like(value, st_mode=stat.S_IFDIR | 0o700)
            return value

        monkeypatch.setattr(promotion.os, "lstat", unsafe_cleanup)
    elif case == "changed-copying":
        original_fstat = os.fstat
        calls = 0

        def changed_fstat(descriptor):
            nonlocal calls
            value = original_fstat(descriptor)
            calls += 1
            return _stat_like(value, st_mtime_ns=value.st_mtime_ns + 1) if calls == 2 else value

        monkeypatch.setattr(promotion.os, "fstat", changed_fstat)
    elif case == "replaced-copying":
        original_lstat = os.lstat
        calls = 0

        def replaced_lstat(path):
            nonlocal calls
            value = original_lstat(path)
            if Path(path) == source:
                calls += 1
                if calls == 2:
                    return _stat_like(value, st_ino=value.st_ino + 1)
            return value

        monkeypatch.setattr(promotion.os, "lstat", replaced_lstat)
    elif case == "digest":
        expected["sha256"] = "0" * 64

    with pytest.raises(CachePromotionError, match=match):
        promotion._copy_regular_file_exclusive(source, temp, expected=expected)


def test_copy_regular_file_exclusive_without_nofollow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.bin"
    temp = tmp_path / "temp.partial"
    payload = b"payload"
    source.write_bytes(payload)
    expected = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    monkeypatch.delattr(promotion.os, "O_NOFOLLOW")

    promotion._copy_regular_file_exclusive(source, temp, expected=expected)

    assert temp.read_bytes() == payload


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing-source", "both absent"),
        ("unsafe-temp", "not a regular file"),
        ("copied-temp-vanishes", "temporary file failed verification"),
        ("link-race-vanished", "appeared concurrently"),
        ("link-error", "Exclusive cache publication failed"),
        ("final-vanished", "Published cache file failed verification"),
    ],
)
def test_publish_file_fallback_rejects_edge_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    staging_root = tmp_path / "staging"
    destination_root = tmp_path / "live"
    publish_root = destination_root / ".transactions/tx/publish"
    source = staging_root / "source.bin"
    destination = destination_root / "processed/artifact.bin"
    staging_root.mkdir()
    destination_root.mkdir()
    payload = b"payload"
    plan = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    if case != "missing-source":
        source.write_bytes(payload)
    temp = promotion._publish_temp_path(
        publish_root,
        destination,
        destination_root=destination_root,
        expected=plan,
    )
    if case == "unsafe-temp":
        publish_root.mkdir(parents=True)
        temp.mkdir()
        monkeypatch.setattr(promotion, "_reconcile_publish_temp", lambda *_args, **_kwargs: None)
    elif case == "copied-temp-vanishes":
        monkeypatch.setattr(
            promotion,
            "_copy_regular_file_exclusive",
            lambda *_args, **_kwargs: None,
        )
    elif case == "link-race-vanished":
        monkeypatch.setattr(
            promotion.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(FileExistsError()),
        )
    elif case == "link-error":
        monkeypatch.setattr(
            promotion.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("link failed")),
        )
    elif case == "final-vanished":
        original_file_state = promotion._file_state
        destination_checks = 0

        def vanish_at_final(path, **kwargs):
            nonlocal destination_checks
            if Path(path) == destination:
                destination_checks += 1
                if destination_checks == 2:
                    return "absent"
            return original_file_state(path, **kwargs)

        monkeypatch.setattr(promotion, "_file_state", vanish_at_final)

    with pytest.raises(CachePromotionError, match=match):
        promotion._publish_file_fallback(
            plan,
            source=source,
            destination=destination,
            staging_root=staging_root,
            destination_root=destination_root,
            publish_root=publish_root,
        )


@pytest.mark.parametrize("case", ["exact-temp", "conflicting-temp", "link-race-exact"])
def test_publish_file_fallback_recovers_reusable_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    staging_root = tmp_path / "staging"
    destination_root = tmp_path / "live"
    publish_root = destination_root / ".transactions/tx/publish"
    source = staging_root / "source.bin"
    destination = destination_root / "processed/artifact.bin"
    source.parent.mkdir()
    destination_root.mkdir()
    payload = b"payload"
    source.write_bytes(payload)
    plan = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    temp = promotion._publish_temp_path(
        publish_root,
        destination,
        destination_root=destination_root,
        expected=plan,
    )
    if case in {"exact-temp", "conflicting-temp"}:
        publish_root.mkdir(parents=True)
        temp.write_bytes(payload if case == "exact-temp" else b"conflict")
    if case == "link-race-exact":

        def publish_then_race(*_args, **_kwargs):
            destination.write_bytes(payload)
            raise FileExistsError

        monkeypatch.setattr(promotion.os, "link", publish_then_race)

    published = promotion._publish_file_fallback(
        plan,
        source=source,
        destination=destination,
        staging_root=staging_root,
        destination_root=destination_root,
        publish_root=publish_root,
    )

    assert destination.read_bytes() == payload
    assert published is (case != "link-race-exact")


def _simple_directory_plan(
    staging_root: Path,
    destination_root: Path,
    *,
    with_file: bool,
) -> tuple[dict[str, object], Path, Path]:
    source = staging_root / "processed/source"
    destination = destination_root / "processed/destination"
    source.mkdir(parents=True)
    files: list[dict[str, object]] = []
    if with_file:
        payload = b"payload"
        (source / "artifact.bin").write_bytes(payload)
        files.append(
            {
                "path": "artifact.bin",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return (
        {
            "destination": "processed/destination",
            "files": files,
            "path": "processed/source",
        },
        source,
        destination,
    )


def test_promote_directory_rejects_invalid_files(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    destination_root = tmp_path / "live"
    staging_root.mkdir()
    destination_root.mkdir()
    plan = {"destination": "processed/destination", "files": None, "path": "processed/source"}

    with pytest.raises(CachePromotionError, match="intent files are invalid"):
        promotion._promote_directory(
            plan,
            staging_root=staging_root,
            destination_root=destination_root,
            publish_root=destination_root / ".transactions/tx/publish",
        )


def test_promote_directory_rejects_destination_created_after_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging_root = tmp_path / "staging"
    destination_root = tmp_path / "live"
    destination_root.mkdir()
    plan, _, destination = _simple_directory_plan(
        staging_root,
        destination_root,
        with_file=True,
    )
    publish_root = destination_root / ".transactions/tx/publish"
    monkeypatch.setattr(
        promotion,
        "_atomic_rename_noreplace",
        lambda *_args: (_ for _ in ()).throw(promotion._AtomicNoReplaceUnsupported("unsupported")),
    )

    def claim_then_race(*_args, **_kwargs):
        destination.mkdir(parents=True)
        return True

    monkeypatch.setattr(promotion, "_write_tree_claim", claim_then_race)

    with pytest.raises(CachePromotionError, match="appeared before it was claimed"):
        promotion._promote_directory(
            plan,
            staging_root=staging_root,
            destination_root=destination_root,
            publish_root=publish_root,
        )


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (FileExistsError(), "appeared concurrently"),
        (OSError("mkdir failed"), "Cannot create claimed"),
    ],
)
def test_promote_directory_rejects_claimed_mkdir_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: OSError,
    match: str,
) -> None:
    staging_root = tmp_path / "staging"
    destination_root = tmp_path / "live"
    destination_root.mkdir()
    plan, _, destination = _simple_directory_plan(
        staging_root,
        destination_root,
        with_file=False,
    )
    publish_root = destination_root / ".transactions/tx/publish"
    publish_root.mkdir(parents=True)
    monkeypatch.setattr(
        promotion,
        "_atomic_rename_noreplace",
        lambda *_args: (_ for _ in ()).throw(promotion._AtomicNoReplaceUnsupported("unsupported")),
    )
    original_mkdir = os.mkdir

    def fail_destination(path, mode=0o777, *, dir_fd=None):
        if Path(path) == destination:
            raise error
        return original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(promotion.os, "mkdir", fail_destination)

    with pytest.raises(CachePromotionError, match=match):
        promotion._promote_directory(
            plan,
            staging_root=staging_root,
            destination_root=destination_root,
            publish_root=publish_root,
        )


def test_promote_directory_rejects_failed_fallback_postcheck(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging_root = tmp_path / "staging"
    destination_root = tmp_path / "live"
    destination_root.mkdir()
    plan, source, destination = _simple_directory_plan(
        staging_root,
        destination_root,
        with_file=False,
    )
    publish_root = destination_root / ".transactions/tx/publish"
    monkeypatch.setattr(
        promotion,
        "_atomic_rename_noreplace",
        lambda *_args: (_ for _ in ()).throw(promotion._AtomicNoReplaceUnsupported("unsupported")),
    )
    original_directory_state = promotion._directory_state

    def fail_final(path, **kwargs):
        if Path(path) == destination:
            return "absent"
        assert Path(path) == source
        return original_directory_state(path, **kwargs)

    monkeypatch.setattr(promotion, "_directory_state", fail_final)

    with pytest.raises(CachePromotionError, match="failed verification"):
        promotion._promote_directory(
            plan,
            staging_root=staging_root,
            destination_root=destination_root,
            publish_root=publish_root,
        )


@pytest.mark.parametrize("preexisting", ["empty", "exact-subset"])
def test_fallback_rejects_unclaimed_partial_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preexisting: str,
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    target = destination.processed_dir(fingerprint)
    target.mkdir(parents=True)
    if preexisting == "exact-subset":
        source_file = next(
            path for path in staging.processed_dir(fingerprint).rglob("*") if path.is_file()
        )
        relative = source_file.relative_to(staging.processed_dir(fingerprint))
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        (target / relative).write_bytes(source_file.read_bytes())
    _force_linux_rename_einval(monkeypatch)

    with pytest.raises(CachePromotionError, match="no exact transaction claim"):
        _promote(staging, destination, fingerprint, f"unclaimed-{preexisting}")

    assert staging.manifest_path(fingerprint).is_file()
    assert not destination.manifest_path(fingerprint).exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_promote_helpers_reject_missing_source_and_failed_postcheck(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
) -> None:
    plan = {
        "destination": "destination",
        "files": [],
        "path": "source",
        "sha256": hashlib.sha256(b"").hexdigest(),
        "size_bytes": 0,
    }
    helper = promotion._promote_file if kind == "file" else promotion._promote_directory
    state_name = "_file_state" if kind == "file" else "_directory_state"
    match = "source and destination are both absent"
    monkeypatch.setattr(promotion, state_name, lambda *_args, **_kwargs: "absent")
    with pytest.raises(CachePromotionError, match=match):
        helper(
            plan,
            staging_root=tmp_path,
            destination_root=tmp_path,
            publish_root=tmp_path / "publish",
        )

    states = iter(["exact", "absent"] if kind == "directory" else ["absent", "exact", "absent"])
    monkeypatch.setattr(promotion, state_name, lambda *_args, **_kwargs: next(states))
    monkeypatch.setattr(promotion, "_mkdir_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(promotion, "_atomic_rename_noreplace", lambda *_args: None)
    monkeypatch.setattr(promotion, "_fsync_directory", lambda *_args: None)
    with pytest.raises(CachePromotionError, match="failed verification"):
        helper(
            plan,
            staging_root=tmp_path,
            destination_root=tmp_path,
            publish_root=tmp_path / "publish",
        )


@pytest.mark.parametrize(
    ("entries", "match"),
    [
        (None, "has no entries"),
        ([], "has no entries"),
        ([{"fingerprint": "b" * 64}], "entry order differs"),
    ],
)
def test_validate_intent_rejects_missing_or_reordered_entries(
    tmp_path: Path,
    entries,
    match: str,
) -> None:
    staging = CacheLayout(tmp_path / "staging")
    destination = CacheLayout(tmp_path / "live")
    intent = _intent_envelope(staging, destination, "transaction")
    intent["entries"] = entries
    request = promotion._expectation_payload([promotion.CacheEntryExpectation("a" * 64)])
    with pytest.raises(CachePromotionError, match=match):
        promotion._validate_intent(
            intent,
            staging=staging,
            destination=destination,
            transaction_id="transaction",
            request=request,
        )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("processed", "processed cache is absent"),
        ("source", "source record is absent"),
        ("manifest", "manifest is absent"),
    ],
)
def test_final_entry_rejects_missing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    entry = {
        "content_manifest": {"path": "manifests/content", "sha256": "0" * 64, "size_bytes": 0},
        "content_sha256": "0" * 64,
        "fingerprint": "a" * 64,
        "main_manifest": {"path": "manifests/main", "sha256": "0" * 64, "size_bytes": 0},
        "processed": {"destination": "processed/item", "files": []},
        "source_files": (
            [{"destination": "raw/source", "sha256": "0" * 64, "size_bytes": 0}]
            if case == "source"
            else []
        ),
    }
    monkeypatch.setattr(
        promotion,
        "_directory_state",
        lambda *_args, **_kwargs: "absent" if case == "processed" else "exact",
    )
    monkeypatch.setattr(promotion, "_file_state", lambda *_args, **_kwargs: "absent")

    with pytest.raises(CachePromotionError, match=match):
        promotion._verify_final_entry(entry, destination=CacheLayout(tmp_path))


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("verification", "Live cache verification failed"),
        ("digest", "Live content digest differs"),
    ],
)
def test_final_entry_rejects_invalid_live_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    plan = _entry_plan(staging, destination, fingerprint)
    _copy_plan_part(staging, destination, plan["processed"], destination_key="destination")
    _copy_plan_part(staging, destination, plan["content_manifest"])
    _copy_plan_part(staging, destination, plan["main_manifest"])
    if case == "verification":
        monkeypatch.setattr(
            promotion.content,
            "verify_content_manifest",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ManifestError("bad")),
        )
    else:
        monkeypatch.setattr(
            promotion.content,
            "verify_content_manifest",
            lambda *_args, **_kwargs: {"content_sha256": "0" * 64},
        )

    with pytest.raises(CachePromotionError, match=match):
        promotion._verify_final_entry(plan, destination=destination)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("transaction", "receipt transaction differs"),
        ("intent", "intent digest differs"),
        ("staging", "staging root differs"),
        ("destination", "destination root differs"),
        ("completed-type", "completion time is invalid"),
        ("index", "index digest is invalid"),
        ("items-type", "receipt entries differ"),
        ("items-count", "receipt entries differ"),
        ("item-type", "receipt item is invalid"),
        ("fingerprint", "receipt entries differ"),
        ("disposition", "receipt disposition is invalid"),
    ],
)
def test_receipt_rejects_tampered_fields(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    report = _promote(staging, destination, fingerprint, f"receipt-field-{case}")
    receipt = json.loads(report.receipt_path.read_text(encoding="utf-8"))
    if case == "transaction":
        receipt["transaction_id"] = "other"
    elif case == "intent":
        receipt["intent_sha256"] = "0" * 64
    elif case == "staging":
        receipt["staging_root"] = "other"
    elif case == "destination":
        receipt["cache_root"] = "other"
    elif case == "completed-type":
        receipt["completed_at"] = None
    elif case == "index":
        receipt["index_sha256"] = "BAD"
    elif case == "items-type":
        receipt["items"] = None
    elif case == "items-count":
        receipt["items"] = []
    elif case == "item-type":
        receipt["items"] = [None]
    elif case == "fingerprint":
        receipt["items"][0]["fingerprint"] = "b" * 64
    else:
        receipt["items"][0]["disposition"] = "unknown"
    report.receipt_path.write_bytes(promotion._canonical_json(receipt) + b"\n")

    with pytest.raises(CachePromotionError, match=match):
        _promote(staging, destination, fingerprint, f"receipt-field-{case}")


@pytest.mark.parametrize("transaction_id", ["", ".", "..", "bad/id"])
def test_promotion_rejects_invalid_transaction_id(tmp_path: Path, transaction_id: str) -> None:
    with pytest.raises(CachePromotionError, match="Invalid cache-promotion transaction id"):
        promotion.promote_cache_entries(
            staging_dir=tmp_path,
            cache_dir=tmp_path,
            entries=[promotion.CacheEntryExpectation("a" * 64)],
            transaction_id=transaction_id,
        )


def test_promotion_rejects_same_or_cross_filesystem_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(CachePromotionError, match="roots must differ"):
        promotion.promote_cache_entries(
            staging_dir=tmp_path,
            cache_dir=tmp_path,
            entries=[promotion.CacheEntryExpectation("a" * 64)],
            transaction_id="same",
        )

    staging = tmp_path / "staging"
    destination = tmp_path / "live"
    staging.mkdir()
    destination.mkdir()
    roots = iter([staging, destination])
    monkeypatch.setattr(promotion, "_canonical_root", lambda *_args, **_kwargs: next(roots))
    original_lstat = os.lstat

    def changed(path: os.PathLike[str] | str):
        value = original_lstat(path)
        if Path(path) == destination:
            return _stat_like(value, st_dev=value.st_dev + 1)
        return value

    monkeypatch.setattr(promotion.os, "lstat", changed)
    with pytest.raises(CachePromotionError, match="share a filesystem"):
        promotion.promote_cache_entries(
            staging_dir=staging,
            cache_dir=destination,
            entries=[promotion.CacheEntryExpectation("a" * 64)],
            transaction_id="cross-device",
        )


def test_promotion_recovers_intent_creation_race(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    write = promotion._write_exclusive

    def race(path: Path, payload: bytes) -> None:
        if path.name == "intent.json":
            path.write_bytes(payload)
            raise FileExistsError
        write(path, payload)

    monkeypatch.setattr(promotion, "_write_exclusive", race)
    report = _promote(staging, destination, fingerprint, "intent-race")
    assert report.receipt_path.is_file()


def test_promotion_recovers_receipt_creation_race(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    write = promotion._write_exclusive

    def race(path: Path, payload: bytes) -> None:
        if path.name == "receipt.json":
            path.write_bytes(payload)
            raise FileExistsError
        write(path, payload)

    monkeypatch.setattr(promotion, "_write_exclusive", race)
    report = _promote(staging, destination, fingerprint, "receipt-race")
    assert report.receipt_path.is_file()


def test_promotion_rejects_conflicting_sources_in_existing_intent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging, fingerprint = _toy_staging(tmp_path / "staging")
    destination = _empty_destination(tmp_path / "live")
    request = promotion._expectation_payload([_expectation(fingerprint)])
    plan = _entry_plan(staging, destination, fingerprint)
    plan["source_files"] = [
        {"destination": "raw/shared", "path": "raw/first", "sha256": "1" * 64, "size_bytes": 1},
        {"destination": "raw/shared", "path": "raw/second", "sha256": "2" * 64, "size_bytes": 1},
    ]
    intent = {
        "cache_root": str(destination.root),
        "entries": [plan],
        "request": request,
        "schema_version": 1,
        "staging_root": str(staging.root),
        "transaction_id": "intent-source-conflict",
    }
    monkeypatch.setattr(promotion, "_build_intent", lambda **_kwargs: intent)

    with pytest.raises(CachePromotionError, match="intent contains conflicting sources"):
        _promote(staging, destination, fingerprint, "intent-source-conflict")


def test_cache_lock_reader_rejects_hardlink_and_supports_no_nofollow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock = tmp_path / "lock"
    lock.write_bytes(b"")
    os.link(lock, tmp_path / "copy")
    with pytest.raises(ManifestError, match="single-link regular file"):
        cache._open_lock_file(lock)

    lock.unlink()
    monkeypatch.delattr(cache.os, "O_NOFOLLOW")
    descriptor = cache._open_lock_file(tmp_path / "fallback")
    os.close(descriptor)


def test_atomic_index_rejects_symlink_fingerprint_and_missing_processed(
    tmp_path: Path,
) -> None:
    layout = CacheLayout(tmp_path / "cache")
    layout.root.mkdir(parents=True)
    layout.locks_root.mkdir()
    target = tmp_path / "target-index"
    target.write_bytes(b"")
    layout.index_path.symlink_to(target)
    with pytest.raises(ManifestError, match="must not be a symlink"):
        cache.rebuild_index_atomic(layout)
    layout.index_path.unlink()

    layout.manifests_root.mkdir()
    named = "a" * 64
    manifest = Manifest(
        schema_version=1,
        fingerprint="b" * 64,
        created_at="2026-08-31T00:00:00+00:00",
        identity={"provider": "toy", "dataset_id": "toy"},
        dataset={},
        meta={},
        environment={},
    )
    write_manifest(layout.manifest_path(named), manifest)
    with pytest.raises(ManifestError, match="fingerprint differs"):
        cache.rebuild_index_atomic(layout)

    manifest = replace(manifest, fingerprint=named)
    write_manifest(layout.manifest_path(named), manifest)
    with pytest.raises(ManifestError, match="no processed cache"):
        cache.rebuild_index_atomic(layout)


def test_atomic_index_non_strict_skips_invalid_manifest(tmp_path: Path) -> None:
    layout = CacheLayout(tmp_path / "cache")
    cache.ensure_layout(layout)
    (layout.manifests_root / f"{'a' * 64}.json").write_text("not-json", encoding="utf-8")

    digest = cache.rebuild_index_atomic(layout, strict=False)

    assert len(digest) == 64
    assert cache.index_list(layout) == []


def test_atomic_index_rejects_failed_integrity_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = CacheLayout(tmp_path / "cache")
    cache.ensure_layout(layout)
    connect = cache.sqlite3.connect

    class ConnectionProxy:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, statement, *args):
            if statement == "PRAGMA integrity_check":
                return SimpleNamespace(fetchone=lambda: ("corrupt",))
            return self.connection.execute(statement, *args)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    monkeypatch.setattr(
        cache.sqlite3,
        "connect",
        lambda *args, **kwargs: ConnectionProxy(connect(*args, **kwargs)),
    )
    with pytest.raises(ManifestError, match="integrity check failed"):
        cache.rebuild_index_atomic(layout)


@pytest.mark.parametrize(
    "arguments",
    [
        ["cache", "promote", "--staging-dir", ".", "--transaction-id", "missing"],
        [
            "cache",
            "promote",
            "--staging-dir",
            ".",
            "--transaction-id",
            "both",
            "--fingerprint",
            "a" * 64,
            "--all-staged",
        ],
    ],
)
def test_cli_promote_requires_exact_selection(arguments: list[str]) -> None:
    result = CliRunner().invoke(datasets_app, arguments)
    assert result.exit_code == 2
    assert "repeated --fingerprint" in result.output


def test_cli_promote_configures_logging_and_uses_default_cache(tmp_path: Path) -> None:
    report = SimpleNamespace(to_json=lambda: "{}\n")
    with (
        patch("modssc.cli.datasets.configure_logging") as configure,
        patch("modssc.cli.datasets.api.cache_dir", return_value=tmp_path / "live") as default,
        patch("modssc.cli.datasets.api.promote_cache_entries", return_value=report) as called,
    ):
        result = CliRunner().invoke(
            datasets_app,
            [
                "cache",
                "promote",
                "--staging-dir",
                str(tmp_path),
                "--transaction-id",
                "default-cache",
                "--fingerprint",
                "a" * 64,
                "--log-level",
                "detailed",
            ],
        )

    assert result.exit_code == 0, result.output
    configure.assert_called_once()
    default.assert_called_once_with()
    assert called.call_args.kwargs["cache_dir"] == tmp_path / "live"


@pytest.mark.parametrize("debug", [False, True])
def test_cli_promote_reports_native_error(
    tmp_path: Path,
    debug: bool,
) -> None:
    with (
        patch(
            "modssc.cli.datasets.api.promote_cache_entries",
            side_effect=CachePromotionError("promotion failed"),
        ),
        patch("modssc.cli.datasets.logger.isEnabledFor", return_value=debug),
        patch("modssc.cli.datasets.logger.exception") as logged,
    ):
        result = CliRunner().invoke(
            datasets_app,
            [
                "cache",
                "promote",
                "--staging-dir",
                str(tmp_path),
                "--cache-dir",
                str(tmp_path / "live"),
                "--transaction-id",
                "failure",
                "--fingerprint",
                "a" * 64,
            ],
        )

    assert result.exit_code == 2
    assert "promotion failed" in result.output
    assert logged.called is debug
