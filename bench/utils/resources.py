from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


def _resolve_path(value: Any, *, resource_root: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((resource_root / path).resolve())


def resolve_sampling_plan_resources(
    plan: Mapping[str, Any], *, resource_root: Path
) -> dict[str, Any]:
    """Resolve authenticated sampling inputs without mutating the declared plan.

    Only fields whose schema explicitly denotes a content-authenticated input are
    resolved. Cache and output paths intentionally remain outside this function.
    """

    resolved = deepcopy(dict(plan))
    partition = resolved.get("partition")
    if isinstance(partition, dict):
        artifact = partition.get("ordered_indices_artifact")
        if isinstance(artifact, dict) and "path" in artifact:
            artifact["path"] = _resolve_path(artifact["path"], resource_root=resource_root)

    labeling = resolved.get("labeling")
    if isinstance(labeling, dict):
        artifact = labeling.get("fixed_indices_artifact")
        if isinstance(artifact, dict) and "path" in artifact:
            artifact["path"] = _resolve_path(artifact["path"], resource_root=resource_root)
    return resolved


def resolve_graph_spec_resources(spec: Mapping[str, Any], *, resource_root: Path) -> dict[str, Any]:
    """Resolve authenticated graph inputs relative to the declaring YAML file."""

    resolved = deepcopy(dict(spec))
    if "precomputed_path" in resolved:
        resolved["precomputed_path"] = _resolve_path(
            resolved["precomputed_path"], resource_root=resource_root
        )
    return resolved
