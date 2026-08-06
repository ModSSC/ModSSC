"""Operational resource and allocation checks for campaign preflight.

The scientific checks remain in :mod:`bench.campaign.governance`.  This adapter
translates scheduler site files into the normalized resource contract accepted
by that module, then adds allocation accounting to the resulting report.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from bench.campaign.errors import CampaignError
from bench.campaign.governance import run_preflight as run_scientific_preflight
from bench.campaign.manifest import load_manifest
from bench.utils.io import atomic_write_json

from .execution_context import execution_metadata
from .resources import load_execution_site, profile_resource_metadata


@dataclass(frozen=True)
class HPCPreflightReport:
    campaign_id: str
    status: str
    task_count: int
    report_path: str
    error_count: int
    planned_gpu_hours: dict[str, float]


def _read_mapping(path: Path, *, code: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CampaignError(code, f"cannot read {path}") from exc
    if not isinstance(raw, dict):
        raise CampaignError(code, f"root must be a mapping: {path}")
    return raw


def load_allocation_snapshot(path: Path) -> dict[str, Any]:
    raw = _read_mapping(path, code="E_CAMPAIGN_ALLOCATION_INVALID")
    if raw.get("schema_version") != 1:
        raise CampaignError("E_CAMPAIGN_ALLOCATION_INVALID", "schema_version must equal 1")
    reserve = raw.get("reserve_fraction")
    if (
        isinstance(reserve, bool)
        or not isinstance(reserve, int | float)
        or not 0.15 <= float(reserve) < 1.0
    ):
        raise CampaignError(
            "E_CAMPAIGN_ALLOCATION_INVALID", "reserve_fraction must be at least 0.15"
        )
    updated_at = raw.get("updated_at")
    try:
        parsed_updated_at = (
            updated_at if isinstance(updated_at, datetime) else datetime.fromisoformat(updated_at)
        )
    except (TypeError, ValueError) as exc:
        raise CampaignError(
            "E_CAMPAIGN_ALLOCATION_INVALID", "updated_at must be a valid ISO-8601 timestamp"
        ) from exc
    if parsed_updated_at.tzinfo is None or parsed_updated_at.utcoffset() is None:
        raise CampaignError(
            "E_CAMPAIGN_ALLOCATION_INVALID", "updated_at must include an explicit timezone"
        )
    architectures = raw.get("architectures")
    if not isinstance(architectures, Mapping) or not architectures:
        raise CampaignError("E_CAMPAIGN_ALLOCATION_INVALID", "architectures are required")
    normalized: dict[str, dict[str, float]] = {}
    for architecture, payload in architectures.items():
        if not isinstance(architecture, str) or not architecture.strip():
            raise CampaignError("E_CAMPAIGN_ALLOCATION_INVALID", "invalid architecture name")
        if not isinstance(payload, Mapping):
            raise CampaignError(
                "E_CAMPAIGN_ALLOCATION_INVALID", f"{architecture} allocation must be a mapping"
            )
        values: dict[str, float] = {}
        for field in ("total_hours", "consumed_hours", "other_committed_hours"):
            value = payload.get(field, 0.0)
            if isinstance(value, bool) or not isinstance(value, int | float) or float(value) < 0:
                raise CampaignError(
                    "E_CAMPAIGN_ALLOCATION_INVALID",
                    f"{architecture}.{field} must be a non-negative number",
                )
            values[field] = float(value)
        if values["consumed_hours"] > values["total_hours"]:
            raise CampaignError(
                "E_CAMPAIGN_ALLOCATION_INVALID",
                f"{architecture}.consumed_hours exceeds total_hours",
            )
        key = architecture.upper()
        if key in normalized:
            raise CampaignError("E_CAMPAIGN_ALLOCATION_INVALID", f"duplicate architecture: {key}")
        normalized[key] = values
    return {
        "schema_version": 1,
        "updated_at": parsed_updated_at.astimezone(UTC).isoformat(),
        "reserve_fraction": float(reserve),
        "architectures": normalized,
    }


def load_site_resources(site_paths: Sequence[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    resources: dict[tuple[str, str], dict[str, Any]] = {}
    for path in site_paths:
        site = load_execution_site(path)
        site_id = str(site["site_id"])
        for profile_id, profile in site["profiles"].items():
            if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
                raise CampaignError("E_CAMPAIGN_SITE_INVALID", "invalid profile mapping")
            key = (site_id, profile_id)
            if key in resources:
                raise CampaignError(
                    "E_CAMPAIGN_SITE_INVALID", f"duplicate resource profile {site_id}.{profile_id}"
                )
            resources[key] = profile_resource_metadata(
                site_id=site_id,
                profile_id=profile_id,
                profile=profile,
                executor=str(site["scheduler"]),
            )
    return resources


def _runtime_estimates(path: Path | None) -> dict[tuple[str | None, str], float]:
    if path is None:
        return {}
    raw = _read_mapping(path, code="E_CAMPAIGN_ESTIMATES_INVALID")
    estimates: dict[tuple[str | None, str], float] = {}
    profiles = raw.get("profiles")
    if isinstance(profiles, Mapping):
        for name, payload in profiles.items():
            if not isinstance(name, str) or not isinstance(payload, Mapping):
                raise CampaignError("E_CAMPAIGN_ESTIMATES_INVALID", "invalid profile estimate")
            value = payload.get("p95_seconds")
            if isinstance(value, bool) or not isinstance(value, int | float) or float(value) <= 0:
                raise CampaignError(
                    "E_CAMPAIGN_ESTIMATES_INVALID", f"{name}.p95_seconds must be positive"
                )
            estimates[tuple(name.split(".", 1)) if "." in name else (None, name)] = float(value)
        return estimates
    rows = raw.get("by_resource_site")
    if not isinstance(rows, list):
        raise CampaignError(
            "E_CAMPAIGN_ESTIMATES_INVALID", "expected profiles or daily by_resource_site"
        )
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        site = row.get("site")
        profile = row.get("resource_profile")
        value = row.get("success_runtime_p95_seconds", row.get("runtime_p95_seconds"))
        if (
            isinstance(site, str)
            and isinstance(profile, str)
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            and float(value) > 0
        ):
            estimates[(site, profile)] = float(value)
    return estimates


def _resource_budget(
    *,
    tasks: Sequence[Any],
    resources: Mapping[tuple[str, str], Mapping[str, Any]],
    allocation: Mapping[str, Any],
    estimates: Mapping[tuple[str | None, str], float],
) -> tuple[list[dict[str, Any]], dict[str, float], list[str]]:
    counts = Counter((task.assigned_site, task.resource_profile) for task in tasks)
    planned: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for (site, profile), count in sorted(counts.items()):
        resource = resources.get((site, profile))
        if resource is None:
            errors.append(f"missing resource metadata for {site}.{profile}")
            continue
        architecture = str(resource["architecture"]).upper()
        accelerators = int(resource["accelerators_per_task"])
        p95 = estimates.get((site, profile), estimates.get((None, profile)))
        seconds = float(p95 or resource["configured_walltime_seconds"])
        hours = count * seconds * accelerators / 3600.0
        if accelerators > 0:
            planned[architecture] += hours
        requested = 1.25 * seconds if p95 is not None else seconds
        if requested > float(resource["max_walltime_seconds"]):
            errors.append(f"{site}.{profile}: requested walltime exceeds its configured cap")
        rows.append(
            {
                "site": site,
                "resource_profile": profile,
                "architecture": architecture,
                "task_count": count,
                "planned_gpu_hours": hours,
                "p95_seconds": None if p95 is None else seconds,
                "estimate_basis": (
                    "calibrated_p95" if p95 is not None else "configured_walltime_upper_bound"
                ),
                "runtime_estimate_seconds": seconds,
                "requested_walltime_seconds": requested,
                "max_walltime_seconds": resource["max_walltime_seconds"],
            }
        )
    allocation_architectures = allocation["architectures"]
    for architecture in sorted(
        {
            str(resource["architecture"]).upper()
            for resource in resources.values()
            if int(resource["accelerators_per_task"]) > 0
        }
    ):
        if architecture not in allocation_architectures:
            errors.append(f"exact allocation is missing for {architecture}")
    reserve = float(allocation["reserve_fraction"])
    for architecture, hours in sorted(planned.items()):
        record = allocation_architectures.get(architecture)
        if record is None:
            continue
        spendable = (
            float(record["total_hours"]) * (1.0 - reserve)
            - float(record["consumed_hours"])
            - float(record["other_committed_hours"])
        )
        if hours > spendable:
            errors.append(
                f"{architecture}: planned {hours:.3f} GPU-h exceeds spendable "
                f"{spendable:.3f} GPU-h after the {reserve:.0%} reserve"
            )
    return rows, dict(sorted(planned.items())), errors


def run_preflight(
    manifest_path: Path,
    *,
    allocation_path: Path,
    site_paths: Sequence[Path],
    repo_root: Path,
    output_path: Path,
    runtime_estimates_path: Path | None = None,
    max_allocation_age_hours: float = 24.0,
    now_provider: Callable[[], datetime] = lambda: datetime.now(UTC),
    **scientific_options: Any,
) -> HPCPreflightReport:
    if (
        isinstance(max_allocation_age_hours, bool)
        or not isinstance(max_allocation_age_hours, int | float)
        or not math.isfinite(float(max_allocation_age_hours))
        or float(max_allocation_age_hours) <= 0
    ):
        raise CampaignError(
            "E_CAMPAIGN_ALLOCATION_INVALID",
            "max_allocation_age_hours must be a finite positive number",
        )
    observed_at = now_provider()
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise CampaignError(
            "E_CAMPAIGN_ALLOCATION_INVALID", "preflight clock must include a timezone"
        )
    allocation = load_allocation_snapshot(allocation_path)
    resources = load_site_resources(site_paths)
    updated_at = datetime.fromisoformat(str(allocation["updated_at"]))
    expires_at = updated_at + timedelta(hours=float(max_allocation_age_hours))
    _, tasks = load_manifest(manifest_path, verify_digest=True)
    profile_rows, planned, budget_errors = _resource_budget(
        tasks=tasks,
        resources=resources,
        allocation=allocation,
        estimates=_runtime_estimates(runtime_estimates_path),
    )
    scientific = run_scientific_preflight(
        manifest_path,
        repo_root=repo_root,
        output_path=output_path,
        resources=resources,
        authorization_created_at=observed_at,
        authorization_expires_at=expires_at,
        max_authorization_age_hours=max_allocation_age_hours,
        now_provider=lambda: observed_at,
        **scientific_options,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    allocation_age = observed_at.astimezone(UTC) - updated_at
    freshness_errors: list[str] = []
    if allocation_age.total_seconds() < 0:
        freshness_errors.append("allocation updated_at is in the future")
    elif allocation_age >= timedelta(hours=float(max_allocation_age_hours)):
        freshness_errors.append("allocation snapshot is stale")
    freshness_check = {
        "name": "allocation_freshness",
        "status": "pass" if not freshness_errors else "fail",
        "errors": freshness_errors,
        "updated_at": allocation["updated_at"],
        "observed_at": observed_at.astimezone(UTC).isoformat(),
        "age_hours": allocation_age.total_seconds() / 3600.0,
        "max_age_hours": float(max_allocation_age_hours),
        "expires_at": expires_at.isoformat(),
    }
    allocation_check = {
        "name": "allocation_reserve",
        "status": "pass" if not budget_errors else "fail",
        "errors": budget_errors,
        "profiles": profile_rows,
    }
    payload["checks"].extend((freshness_check, allocation_check))
    payload["allocation"] = allocation
    payload["scheduler"] = execution_metadata()
    payload["planned_gpu_hours"] = planned
    payload["max_allocation_age_hours"] = float(max_allocation_age_hours)
    payload["error_count"] = (
        int(payload["error_count"]) + len(freshness_errors) + len(budget_errors)
    )
    payload["status"] = "pass" if payload["error_count"] == 0 else "blocked"
    atomic_write_json(output_path, payload)
    return HPCPreflightReport(
        campaign_id=scientific.campaign_id,
        status=str(payload["status"]),
        task_count=scientific.task_count,
        report_path=str(output_path),
        error_count=int(payload["error_count"]),
        planned_gpu_hours=planned,
    )


__all__ = [
    "HPCPreflightReport",
    "load_allocation_snapshot",
    "load_site_resources",
    "run_preflight",
]
