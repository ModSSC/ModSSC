from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.campaign.model_artifacts import (
    ModelArtifactError,
    build_model_artifact_lock,
    discover_model_ids,
    model_artifact_lock_sha256,
    verify_model_artifact_attestations,
    verify_model_artifact_lock,
)


def _hf_snapshot(root: Path) -> Path:
    repository = root / "hf" / "models--sentence-transformers--all-MiniLM-L6-v2"
    snapshot = repository / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    (snapshot / "config.json").write_text('{"hidden_size": 2}\n', encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    return snapshot


def test_discovers_nested_model_ids_deterministically(tmp_path) -> None:
    config = tmp_path / "configs" / "cell.yaml"
    config.parent.mkdir()
    config.write_text(
        "preprocess:\n"
        "  plan:\n"
        "    steps:\n"
        "      - params:\n"
        "          model_id: st:all-MiniLM-L6-v2\n"
        "      - params:\n"
        "          model_id_vision: torchvision:resnet18\n",
        encoding="utf-8",
    )

    assert discover_model_ids([config.parent]) == [
        "st:all-MiniLM-L6-v2",
        "torchvision:resnet18",
    ]


def test_stub_model_is_locked_without_weights() -> None:
    lock = build_model_artifact_lock(["stub:text"])

    assert lock["models"][0]["artifact_free"] is True
    assert lock["models"][0]["files"] == []
    assert len(model_artifact_lock_sha256(lock)) == 64
    assert verify_model_artifact_lock(lock, ["stub:text"]) == []


def test_hf_snapshot_revision_and_files_are_rehashed(tmp_path) -> None:
    snapshot = _hf_snapshot(tmp_path)
    lock = build_model_artifact_lock(["st:all-MiniLM-L6-v2"], model_cache_root=tmp_path)
    entry = lock["models"][0]

    assert entry["revision"] == "abc123"
    assert [record["path"] for record in entry["files"]] == [
        "config.json",
        "model.safetensors",
    ]
    attestations = verify_model_artifact_lock(
        lock,
        ["st:all-MiniLM-L6-v2"],
        model_cache_root=tmp_path,
    )
    assert len(attestations) == 2
    verify_model_artifact_attestations(attestations)

    weights = snapshot / "model.safetensors"
    original_stat = weights.stat()
    weights.write_bytes(b"changed")
    os.utime(weights, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1))
    with pytest.raises(ModelArtifactError, match="changed after preflight"):
        verify_model_artifact_attestations(attestations)
    with pytest.raises(ModelArtifactError, match="cached artifacts differ"):
        verify_model_artifact_lock(
            lock,
            ["st:all-MiniLM-L6-v2"],
            model_cache_root=tmp_path,
        )


def test_torchvision_and_torchaudio_checkpoint_files_are_locked(tmp_path) -> None:
    checkpoints = tmp_path / "torch" / "hub" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "resnet18-f37072fd.pth").write_bytes(b"resnet")
    (checkpoints / "wav2vec2_fairseq_base_ls960.pth").write_bytes(b"wav2vec")

    lock = build_model_artifact_lock(
        ["torchvision:resnet18", "wav2vec2:base"],
        model_cache_root=tmp_path,
    )

    by_id = {entry["model_id"]: entry for entry in lock["models"]}
    assert by_id["torchvision:resnet18"]["files"][0]["size"] == 6
    assert by_id["wav2vec2:base"]["files"][0]["size"] == 7
    assert (
        len(
            verify_model_artifact_lock(
                lock,
                ["torchvision:resnet18", "wav2vec2:base"],
                model_cache_root=tmp_path,
            )
        )
        == 2
    )


def test_model_lock_refuses_missing_cache_and_missing_entry(tmp_path) -> None:
    with pytest.raises(ModelArtifactError, match="cached checkpoint.*is missing"):
        build_model_artifact_lock(
            ["torchvision:resnet18"],
            model_cache_root=tmp_path,
        )

    lock = build_model_artifact_lock(["stub:text"])
    with pytest.raises(ModelArtifactError, match="lock is missing"):
        verify_model_artifact_lock(lock, ["stub:audio"])
