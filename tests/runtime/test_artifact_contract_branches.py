from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from modssc.runtime import artifacts as artifacts_module
from modssc.runtime.artifacts import (
    ArtifactAttestation,
    ArtifactContract,
    ArtifactContractError,
    ArtifactIntegrityError,
    ArtifactPathState,
    artifact_sha256,
    revalidate_artifact,
    verify_artifact,
)

_DIGEST = "a" * 64


def _state(**overrides: Any) -> ArtifactPathState:
    values: dict[str, Any] = {
        "path": ".",
        "kind": "file",
        "size_bytes": 1,
        "mtime_ns": 2,
        "ctime_ns": 3,
        "mode": 0o644,
        "content_sha256": _DIGEST,
    }
    values.update(overrides)
    return ArtifactPathState(**values)


@pytest.mark.parametrize("value", [None, "x", "g" * 64])
def test_digest_validation_rejects_type_length_and_alphabet(value: Any) -> None:
    with pytest.raises(ArtifactContractError, match="SHA-256"):
        artifacts_module._sha256(value, field="digest")
    assert artifacts_module._sha256(_DIGEST, field="digest") == _DIGEST


@pytest.mark.parametrize("value", [None, "", "bad\x00path", "bad\\path"])
def test_relative_path_rejects_each_nonportable_spelling(value: Any) -> None:
    with pytest.raises(ArtifactContractError, match="portable relative path"):
        artifacts_module._relative_path(value, field="path")


def test_integer_and_contract_validation_cover_all_guards() -> None:
    for value in (True, "1"):
        with pytest.raises(ArtifactContractError, match="integer"):
            artifacts_module._integer(value, field="number")
    with pytest.raises(ArtifactContractError, match=">= 0"):
        artifacts_module._integer(-1, field="number", minimum=0)
    assert artifacts_module._integer(1, field="number") == 1
    assert artifacts_module._integer(1, field="number", minimum=0) == 1

    for kind in (1, "socket"):
        with pytest.raises(ArtifactContractError, match="artifact kind"):
            ArtifactContract(path="file", sha256=_DIGEST, kind=kind)  # type: ignore[arg-type]
    for value in (
        [],
        {"schema_version": 1},
        {"schema_version": 2, "path": "x", "kind": "file", "sha256": _DIGEST},
    ):
        with pytest.raises(ArtifactContractError):
            ArtifactContract.from_dict(value)  # type: ignore[arg-type]


def test_path_state_validation_covers_directory_file_and_symlink_contracts() -> None:
    for kind in (1, "socket"):
        with pytest.raises(ArtifactContractError, match="state kind"):
            _state(kind=kind)  # type: ignore[arg-type]
    with pytest.raises(ArtifactContractError, match=">= 0"):
        _state(size_bytes=-1)
    with pytest.raises(ArtifactContractError, match="integer"):
        _state(mtime_ns=True)

    directory = _state(kind="directory", content_sha256=None)
    assert directory.kind == "directory"
    with pytest.raises(ArtifactContractError, match="file identity"):
        _state(kind="directory", link_target="target")
    with pytest.raises(ArtifactContractError, match="link metadata"):
        _state(kind="directory", content_sha256=None, link_mtime_ns=1)

    with pytest.raises(ArtifactContractError, match="link target"):
        _state(link_target="target")
    with pytest.raises(ArtifactContractError, match="link metadata"):
        _state(link_mtime_ns=1)

    for target in (None, "", "/absolute"):
        with pytest.raises(ArtifactContractError, match="relative link target"):
            _state(
                kind="symlink",
                link_target=target,
                link_mtime_ns=1,
                link_ctime_ns=2,
            )
    with pytest.raises(ArtifactContractError, match="integer"):
        _state(
            kind="symlink",
            link_target="target",
            link_mtime_ns=None,
            link_ctime_ns=2,
        )
    symlink = _state(
        kind="symlink",
        link_target="target",
        link_mtime_ns=1,
        link_ctime_ns=2,
    )
    assert ArtifactPathState.from_dict(symlink.to_dict()) == symlink
    for value in ([], {"path": "."}):
        with pytest.raises(ArtifactContractError):
            ArtifactPathState.from_dict(value)  # type: ignore[arg-type]


def test_attestation_validation_and_deserialization_guards() -> None:
    contract = ArtifactContract(path="file", sha256=_DIGEST)
    state = _state()
    for kwargs in (
        {"contract": object(), "observed_sha256": _DIGEST, "paths": (state,)},
        {"contract": contract, "observed_sha256": "b" * 64, "paths": (state,)},
        {"contract": contract, "observed_sha256": _DIGEST, "paths": []},
        {"contract": contract, "observed_sha256": _DIGEST, "paths": (object(),)},
        {
            "contract": contract,
            "observed_sha256": _DIGEST,
            "paths": (_state(path="other"),),
        },
        {
            "contract": contract,
            "observed_sha256": _DIGEST,
            "paths": (state, state),
        },
    ):
        with pytest.raises(ArtifactContractError):
            ArtifactAttestation(**kwargs)  # type: ignore[arg-type]

    attestation = ArtifactAttestation(contract=contract, observed_sha256=_DIGEST, paths=(state,))
    serialized = attestation.to_dict()
    malformed = [
        [],
        {"schema_version": 1},
        {**serialized, "schema_version": 2},
        {**serialized, "paths": {}},
        {**serialized, "state_sha256": "b" * 64},
    ]
    for value in malformed:
        with pytest.raises(ArtifactContractError):
            ArtifactAttestation.from_dict(value)  # type: ignore[arg-type]


def test_root_and_target_validation_cover_missing_wrong_kind_and_tree_symlink(
    tmp_path: Path,
) -> None:
    with pytest.raises(ArtifactIntegrityError, match="root is missing"):
        artifacts_module._root_path(tmp_path / "missing")
    root_file = tmp_path / "root-file"
    root_file.write_text("x", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="not a directory"):
        artifacts_module._root_path(root_file)
    with pytest.raises(ArtifactIntegrityError, match="escapes"):
        artifacts_module._ensure_inside_root(root_file, root=tmp_path / "elsewhere")

    contract = ArtifactContract(path="missing", sha256=_DIGEST)
    with pytest.raises(ArtifactIntegrityError, match="missing or escapes"):
        artifacts_module._target_path(tmp_path, contract)

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ArtifactIntegrityError, match="not a file"):
        artifacts_module._target_path(
            tmp_path, ArtifactContract(path="directory", kind="file", sha256=_DIGEST)
        )
    regular = tmp_path / "regular"
    regular.write_text("x", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="not a regular directory"):
        artifacts_module._target_path(
            tmp_path, ArtifactContract(path="regular", kind="tree", sha256=_DIGEST)
        )
    link = tmp_path / "tree-link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ArtifactIntegrityError, match="not a regular directory"):
        artifacts_module._target_path(
            tmp_path, ArtifactContract(path="tree-link", kind="tree", sha256=_DIGEST)
        )


def test_stable_hash_detects_io_and_stat_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ArtifactIntegrityError, match="cannot read"):
        artifacts_module._hash_file_stable(tmp_path / "missing")

    value = tmp_path / "value"
    value.write_bytes(b"value")
    calls = iter([(1, 1, 1, 1), (2, 1, 1, 1)])
    monkeypatch.setattr(artifacts_module, "_stat_signature", lambda _value: next(calls))
    with pytest.raises(ArtifactIntegrityError, match="changed while"):
        artifacts_module._hash_file_stable(value)


def test_capture_path_covers_symlink_directory_fifo_and_race_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ArtifactIntegrityError, match="path is missing"):
        artifacts_module._capture_path(tmp_path / "missing", logical_path=".", root=tmp_path)

    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    absolute = tmp_path / "absolute"
    absolute.symlink_to(target)
    with pytest.raises(ArtifactIntegrityError, match="absolute symlink"):
        artifacts_module._capture_path(absolute, logical_path="absolute", root=tmp_path)

    directory = tmp_path / "directory"
    directory.mkdir()
    directory_link = tmp_path / "directory-link"
    directory_link.symlink_to("directory", target_is_directory=True)
    with pytest.raises(ArtifactIntegrityError, match="regular file"):
        artifacts_module._capture_path(directory_link, logical_path="directory-link", root=tmp_path)

    relative = tmp_path / "relative"
    relative.symlink_to("target.bin")
    state = artifacts_module._capture_path(relative, logical_path="relative", root=tmp_path)
    assert state.kind == "symlink" and state.link_target == "target.bin"

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ArtifactIntegrityError, match="unsupported filesystem"):
        artifacts_module._capture_path(fifo, logical_path="fifo", root=tmp_path)

    original_lstat = Path.lstat
    lstat_calls = 0

    def disappearing(path: Path) -> os.stat_result:
        nonlocal lstat_calls
        if path == relative:
            lstat_calls += 1
            if lstat_calls == 2:
                raise OSError("gone")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", disappearing)
    with pytest.raises(ArtifactIntegrityError, match="symlink disappeared"):
        artifacts_module._capture_path(relative, logical_path="relative", root=tmp_path)
    monkeypatch.setattr(Path, "lstat", original_lstat)

    signatures = iter([(1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (2, 1, 1, 1)])
    monkeypatch.setattr(artifacts_module, "_stat_signature", lambda _value: next(signatures))
    with pytest.raises(ArtifactIntegrityError, match="symlink changed"):
        artifacts_module._capture_path(relative, logical_path="relative", root=tmp_path)


def test_tree_scan_covers_relative_symlink_and_enumeration_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "tree"
    nested = tree / "nested"
    nested.mkdir(parents=True)
    target = nested / "target.bin"
    target.write_bytes(b"target")
    (tree / "link.bin").symlink_to("nested/target.bin")

    digest = artifact_sha256(tmp_path, path="tree", kind="tree")
    attestation = verify_artifact(
        ArtifactContract(path="tree", kind="tree", sha256=digest), root=tmp_path
    )
    assert {state.kind for state in attestation.paths} == {"directory", "file", "symlink"}

    original_iterdir = Path.iterdir

    def fail_iterdir(path: Path):
        if path == tree:
            raise OSError("denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_iterdir)
    with pytest.raises(ArtifactIntegrityError, match="cannot enumerate"):
        artifacts_module._scan_artifact(
            ArtifactContract(path="tree", kind="tree", sha256=_DIGEST), root=tmp_path
        )


def test_file_scan_invalid_state_and_public_type_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = tmp_path / "value"
    value.write_bytes(b"value")
    directory_state = _state(kind="directory", content_sha256=None)
    monkeypatch.setattr(artifacts_module, "_capture_path", lambda *args, **kwargs: directory_state)
    with pytest.raises(ArtifactIntegrityError, match="file artifact is invalid"):
        artifacts_module._scan_artifact(
            ArtifactContract(path="value", kind="file", sha256=_DIGEST), root=tmp_path
        )

    with pytest.raises(TypeError, match="ArtifactContract"):
        verify_artifact(object(), root=tmp_path)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ArtifactAttestation"):
        revalidate_artifact(object(), root=tmp_path)  # type: ignore[arg-type]
