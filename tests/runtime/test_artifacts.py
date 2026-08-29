from __future__ import annotations

import os
from pathlib import Path

import pytest

from modssc.runtime import (
    ArtifactAttestation,
    ArtifactContract,
    ArtifactContractError,
    ArtifactIntegrityError,
    artifact_sha256,
    revalidate_artifact,
    verify_artifact,
)


def _file_contract(root: Path, relative_path: str) -> ArtifactContract:
    return ArtifactContract(
        path=relative_path,
        kind="file",
        sha256=artifact_sha256(root, path=relative_path, kind="file"),
    )


def test_file_contract_is_portable_and_attestation_round_trips(tmp_path: Path) -> None:
    artifact = tmp_path / "models" / "weights.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"model-weights")
    contract = _file_contract(tmp_path, "models/weights.bin")

    assert str(tmp_path) not in str(contract.to_dict())
    assert ArtifactContract.from_dict(contract.to_dict()) == contract

    attestation = verify_artifact(contract, root=tmp_path)
    restored = ArtifactAttestation.from_dict(attestation.to_dict())

    assert restored == attestation
    assert len(attestation.state_sha256) == 64
    assert revalidate_artifact(restored, root=tmp_path) == restored


def test_initial_verification_rehashes_and_rejects_wrong_digest(tmp_path: Path) -> None:
    (tmp_path / "weights.bin").write_bytes(b"current")
    contract = ArtifactContract(path="weights.bin", kind="file", sha256="0" * 64)

    with pytest.raises(ArtifactIntegrityError, match="SHA-256 differs"):
        verify_artifact(contract, root=tmp_path)


def test_revalidation_detects_mtime_only_change(tmp_path: Path) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"unchanged")
    attestation = verify_artifact(_file_contract(tmp_path, weights.name), root=tmp_path)
    original = weights.stat()

    os.utime(
        weights,
        ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000),
    )

    with pytest.raises(ArtifactIntegrityError, match="changed after preflight"):
        revalidate_artifact(attestation, root=tmp_path)


def test_revalidation_detects_weight_size_change(tmp_path: Path) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    attestation = verify_artifact(_file_contract(tmp_path, weights.name), root=tmp_path)

    weights.write_bytes(b"weights-changed-and-longer")

    with pytest.raises(ArtifactIntegrityError, match="SHA-256 differs"):
        revalidate_artifact(attestation, root=tmp_path)


def test_revalidation_rehashes_same_size_content_with_restored_mtime(tmp_path: Path) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"1234567")
    attestation = verify_artifact(_file_contract(tmp_path, weights.name), root=tmp_path)
    original = weights.stat()

    weights.write_bytes(b"7654321")
    os.utime(weights, ns=(original.st_atime_ns, original.st_mtime_ns))

    with pytest.raises(ArtifactIntegrityError, match="SHA-256 differs"):
        revalidate_artifact(attestation, root=tmp_path)


def test_tree_digest_is_ordered_portable_and_covers_membership(tmp_path: Path) -> None:
    first = tmp_path / "first" / "model"
    second = tmp_path / "second" / "model"
    (first / "nested").mkdir(parents=True)
    (first / "b.bin").write_bytes(b"b")
    (first / "nested" / "a.bin").write_bytes(b"a")
    (second / "nested").mkdir(parents=True)
    (second / "nested" / "a.bin").write_bytes(b"a")
    (second / "b.bin").write_bytes(b"b")

    first_digest = artifact_sha256(tmp_path / "first", path="model", kind="tree")
    second_digest = artifact_sha256(tmp_path / "second", path="model", kind="tree")
    assert first_digest == second_digest

    contract = ArtifactContract(path="model", kind="tree", sha256=first_digest)
    attestation = verify_artifact(contract, root=tmp_path / "first")
    (first / "new-empty-directory").mkdir()

    with pytest.raises(ArtifactIntegrityError, match="changed after preflight"):
        revalidate_artifact(attestation, root=tmp_path / "first")


@pytest.mark.parametrize("path", ["/absolute/file", "../escape", "nested/../../escape"])
def test_contract_rejects_nonportable_paths(path: str) -> None:
    with pytest.raises(ArtifactContractError, match="escape|absolute"):
        ArtifactContract(path=path, kind="file", sha256="a" * 64)


def test_artifact_cannot_escape_runtime_root_through_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (root / "weights.bin").symlink_to(outside)

    with pytest.raises(ArtifactIntegrityError, match="escapes"):
        artifact_sha256(root, path="weights.bin", kind="file")
