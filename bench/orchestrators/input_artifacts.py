"""Thin YAML-runner orchestration for native external artifact contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modssc.runtime.artifacts import (
    ArtifactAttestation,
    ArtifactContract,
    ArtifactIntegrityError,
    revalidate_artifact,
    verify_artifact,
)

from ..errors import BenchRuntimeError


@dataclass(frozen=True)
class InputArtifactPreflight:
    """Machine-local root plus portable attestations returned by ``src``."""

    root: Path
    attestations: tuple[ArtifactAttestation, ...]
    revalidated_before_success: bool = False

    def report_payload(self) -> dict[str, Any]:
        return {
            "attestations": [attestation.to_dict() for attestation in self.attestations],
            "revalidated_before_success": self.revalidated_before_success,
        }


def _resolve_root(value: str, *, config_path: Path) -> Path:
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    return root.resolve()


def preflight(
    contracts: Sequence[ArtifactContract],
    *,
    artifact_root: str,
    config_path: Path,
) -> InputArtifactPreflight:
    """Resolve the operational root and fully verify every declared input."""

    root = _resolve_root(artifact_root, config_path=config_path)
    try:
        attestations = tuple(verify_artifact(contract, root=root) for contract in contracts)
    except ArtifactIntegrityError as exc:
        raise BenchRuntimeError("E_BENCH_INPUT_ARTIFACT_INTEGRITY", str(exc)) from exc
    return InputArtifactPreflight(root=root, attestations=attestations)


def revalidate(preflight_result: InputArtifactPreflight) -> InputArtifactPreflight:
    """Re-hash every input immediately before the runner reports success."""

    if not isinstance(preflight_result, InputArtifactPreflight):
        raise TypeError("preflight_result must be an InputArtifactPreflight")
    try:
        attestations = tuple(
            revalidate_artifact(attestation, root=preflight_result.root)
            for attestation in preflight_result.attestations
        )
    except ArtifactIntegrityError as exc:
        raise BenchRuntimeError("E_BENCH_INPUT_ARTIFACT_INTEGRITY", str(exc)) from exc
    return InputArtifactPreflight(
        root=preflight_result.root,
        attestations=attestations,
        revalidated_before_success=True,
    )


__all__ = ["InputArtifactPreflight", "preflight", "revalidate"]
