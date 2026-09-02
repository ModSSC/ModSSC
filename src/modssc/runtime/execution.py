from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

ResumePolicy = Literal["never", "auto", "required"]

_RUN_IDENTITY_SCHEMA_VERSION = 1
_RESUME_POLICIES = frozenset({"never", "auto", "required"})


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def normalize_resume_policy(value: str) -> ResumePolicy:
    if not isinstance(value, str) or value not in _RESUME_POLICIES:
        allowed = ", ".join(sorted(_RESUME_POLICIES))
        raise ValueError(f"resume_policy must be one of: {allowed}")
    return cast(ResumePolicy, value)


@dataclass(frozen=True)
class RunIdentity:
    """Stable identity for one effective configuration and seed.

    The optional code digest lets callers distinguish two executions of the
    same configuration without coupling the runtime to Git or a repository.
    """

    config_sha256: str
    seed: int
    code_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "config_sha256",
            _sha256(self.config_sha256, field="config_sha256"),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.code_sha256 is not None:
            object.__setattr__(
                self,
                "code_sha256",
                _sha256(self.code_sha256, field="code_sha256"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _RUN_IDENTITY_SCHEMA_VERSION,
            "config_sha256": self.config_sha256,
            "seed": self.seed,
            "code_sha256": self.code_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunIdentity:
        if not isinstance(value, Mapping):
            raise ValueError("run identity must be a mapping")
        if value.get("schema_version") != _RUN_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported run identity schema_version")
        expected_fields = {"schema_version", "config_sha256", "seed", "code_sha256"}
        if set(value) != expected_fields:
            raise ValueError("run identity fields differ from the runtime schema")
        return cls(
            config_sha256=value["config_sha256"],
            seed=value["seed"],
            code_sha256=value["code_sha256"],
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()

    @property
    def short_id(self) -> str:
        """Return the portable identifier used in human-facing run names."""

        return self.sha256[:20]


@dataclass(frozen=True)
class ExecutionContext:
    """Execution-owned paths and resume semantics passed explicitly to methods."""

    identity: RunIdentity
    output_dir: Path
    resume_policy: ResumePolicy = "never"
    checkpoint_root: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RunIdentity):
            raise TypeError("identity must be a RunIdentity")
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        object.__setattr__(
            self,
            "resume_policy",
            normalize_resume_policy(self.resume_policy),
        )
        if self.checkpoint_root is not None:
            object.__setattr__(
                self,
                "checkpoint_root",
                Path(self.checkpoint_root).expanduser().resolve(),
            )

    @property
    def checkpoint_dir(self) -> Path:
        root = self.checkpoint_root or self.output_dir / ".checkpoints"
        return root / self.identity.sha256

    def should_resume(self, *, checkpoint_exists: bool) -> bool:
        if self.resume_policy == "never":
            return False
        if self.resume_policy == "required" and not checkpoint_exists:
            raise FileNotFoundError("resume_policy='required' but no checkpoint exists")
        return checkpoint_exists


__all__ = [
    "ExecutionContext",
    "ResumePolicy",
    "RunIdentity",
    "normalize_resume_policy",
]
