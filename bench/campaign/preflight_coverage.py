from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any


def _coverage_payload(*, task_ids: list[str], architecture: str | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "all" if architecture is None else "architecture",
        "architecture": architecture,
        "task_count": len(task_ids),
        "task_ids": task_ids,
    }


def _coverage_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_task_coverage(task_ids: Iterable[str], *, architecture: str | None) -> dict[str, Any]:
    """Build the immutable identity of the tasks inspected by one preflight."""

    values = list(task_ids)
    if any(not isinstance(task_id, str) or not task_id for task_id in values):
        raise ValueError("preflight coverage task ids must be non-empty strings")
    ordered = sorted(values)
    if len(ordered) != len(set(ordered)):
        raise ValueError("preflight coverage task ids must be unique")
    payload = _coverage_payload(task_ids=ordered, architecture=architecture)
    return {**payload, "sha256": _coverage_sha256(payload)}


def validate_task_coverage(value: Any) -> dict[str, Any]:
    """Validate and normalize a serialized preflight task coverage payload."""

    if not isinstance(value, Mapping):
        raise ValueError("preflight task coverage is missing")
    task_ids = value.get("task_ids")
    if not isinstance(task_ids, list) or any(
        not isinstance(task_id, str) or not task_id for task_id in task_ids
    ):
        raise ValueError("preflight task coverage has invalid task_ids")
    architecture = value.get("architecture")
    if architecture is not None and (not isinstance(architecture, str) or not architecture.strip()):
        raise ValueError("preflight task coverage has an invalid architecture")
    expected = build_task_coverage(task_ids, architecture=architecture)
    for field in ("schema_version", "scope", "task_count", "task_ids", "sha256"):
        if value.get(field) != expected[field]:
            raise ValueError(f"preflight task coverage {field} differs")
    return expected


__all__ = ["build_task_coverage", "validate_task_coverage"]
