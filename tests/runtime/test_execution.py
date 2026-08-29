from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from modssc.runtime.execution import (
    ExecutionContext,
    RunIdentity,
    normalize_resume_policy,
)

CONFIG_SHA256 = "a" * 64
CODE_SHA256 = "b" * 64


def test_run_identity_is_stable_and_round_trips() -> None:
    identity = RunIdentity(
        config_sha256=CONFIG_SHA256,
        seed=17,
        code_sha256=CODE_SHA256,
    )
    payload = identity.to_dict()
    expected = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    assert identity.sha256 == expected
    assert identity.short_id == expected[:20]
    assert RunIdentity.from_dict(dict(reversed(list(payload.items())))) == identity
    assert RunIdentity(CONFIG_SHA256, 17, CODE_SHA256).sha256 == identity.sha256
    assert RunIdentity(CONFIG_SHA256, 18, CODE_SHA256).sha256 != identity.sha256
    assert RunIdentity(CONFIG_SHA256, 17).sha256 != identity.sha256


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"config_sha256": "short", "seed": 0}, "config_sha256"),
        ({"config_sha256": "A" * 64, "seed": 0}, "config_sha256"),
        ({"config_sha256": CONFIG_SHA256, "seed": True}, "seed"),
        ({"config_sha256": CONFIG_SHA256, "seed": 1.5}, "seed"),
        ({"config_sha256": CONFIG_SHA256, "seed": -1}, "seed"),
        (
            {"config_sha256": CONFIG_SHA256, "seed": 0, "code_sha256": "bad"},
            "code_sha256",
        ),
    ],
)
def test_run_identity_rejects_invalid_values(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RunIdentity(**kwargs)  # type: ignore[arg-type]


def test_run_identity_rejects_invalid_serialized_forms() -> None:
    identity = RunIdentity(CONFIG_SHA256, 3)

    with pytest.raises(ValueError, match="mapping"):
        RunIdentity.from_dict([])  # type: ignore[arg-type]

    payload = identity.to_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        RunIdentity.from_dict(payload)

    payload = identity.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        RunIdentity.from_dict(payload)


def test_resume_policy_validation() -> None:
    assert normalize_resume_policy("never") == "never"
    assert normalize_resume_policy("auto") == "auto"
    assert normalize_resume_policy("required") == "required"

    with pytest.raises(ValueError, match="auto, never, required"):
        normalize_resume_policy("sometimes")
    with pytest.raises(ValueError, match="resume_policy"):
        normalize_resume_policy(None)  # type: ignore[arg-type]


def test_execution_context_derives_identity_scoped_checkpoint_dir(tmp_path: Path) -> None:
    identity = RunIdentity(CONFIG_SHA256, 4)
    context = ExecutionContext(identity, tmp_path / "outputs", resume_policy="auto")

    assert context.output_dir == (tmp_path / "outputs").resolve()
    assert context.checkpoint_root is None
    assert (
        context.checkpoint_dir
        == (tmp_path / "outputs" / ".checkpoints" / identity.sha256).resolve()
    )
    assert not context.should_resume(checkpoint_exists=False)
    assert context.should_resume(checkpoint_exists=True)


def test_execution_context_supports_explicit_checkpoint_root(tmp_path: Path) -> None:
    identity = RunIdentity(CONFIG_SHA256, 5)
    context = ExecutionContext(
        identity,
        tmp_path / "outputs",
        resume_policy="required",
        checkpoint_root=tmp_path / "state",
    )

    assert context.checkpoint_root == (tmp_path / "state").resolve()
    assert context.checkpoint_dir == (tmp_path / "state" / identity.sha256).resolve()
    assert context.should_resume(checkpoint_exists=True)
    with pytest.raises(FileNotFoundError, match="required"):
        context.should_resume(checkpoint_exists=False)


def test_execution_context_never_policy_ignores_existing_checkpoint(tmp_path: Path) -> None:
    context = ExecutionContext(RunIdentity(CONFIG_SHA256, 6), tmp_path)
    assert not context.should_resume(checkpoint_exists=False)
    assert not context.should_resume(checkpoint_exists=True)


def test_execution_context_rejects_invalid_identity(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="RunIdentity"):
        ExecutionContext("invalid", tmp_path)  # type: ignore[arg-type]
