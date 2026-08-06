from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from bench.campaign.errors import CampaignError
from bench.campaign.identifiers import validate_safe_identifier
from bench.campaign.manifest import sha256_bytes, sha256_file, write_text_atomic
from bench.campaign.models import CampaignTask
from bench.utils.io import atomic_write_json

_DURATION_RE = re.compile(
    r"^(?:(?P<days>[0-9]+)-)?(?P<hours>[0-9]+):(?P<minutes>[0-9]{2}):(?P<seconds>[0-9]{2})$"
)
_SUPPORTED_EXECUTORS = {"local", "slurm"}


def parse_duration(value: str) -> int:
    """Return a ``D-HH:MM:SS``/``HH:MM:SS`` duration in seconds."""

    match = _DURATION_RE.fullmatch(value.strip())
    if match is None:
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", f"invalid duration: {value!r}")
    days = int(match.group("days") or 0)
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", f"invalid duration: {value!r}")
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def format_duration(seconds: int) -> str:
    """Format a positive duration without losing hours above 24."""

    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", "duration must be positive")
    hours, remainder = divmod(seconds, 3600)
    minutes, second = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{second:02d}"


def _template_placeholder_paths(value: Any, *, prefix: str = "") -> list[str]:
    if isinstance(value, str) and value.startswith("REPLACE_WITH_"):
        return [prefix or "<root>"]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_template_placeholder_paths(child, prefix=child_prefix))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_template_placeholder_paths(child, prefix=f"{prefix}[{index}]"))
        return paths
    return []


def load_execution_site(path: Path, *, allow_template_placeholders: bool = False) -> dict[str, Any]:
    """Load a resource site without producing scheduler-specific artifacts."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", f"cannot read site: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", f"invalid site schema: {path}")
    executor = raw.get("scheduler")
    if executor not in _SUPPORTED_EXECUTORS:
        raise CampaignError(
            "E_CAMPAIGN_SITE_INVALID",
            f"scheduler must be one of {sorted(_SUPPORTED_EXECUTORS)}",
        )
    placeholders = _template_placeholder_paths(raw)
    if placeholders and not allow_template_placeholders:
        raise CampaignError(
            "E_CAMPAIGN_TEMPLATE_PLACEHOLDER",
            f"site profile {path} contains template values at {placeholders}",
        )
    site_id = validate_safe_identifier(
        raw.get("site_id"), field="site_id", code="E_CAMPAIGN_SITE_INVALID"
    )
    profiles = raw.get("profiles")
    if not isinstance(profiles, Mapping):
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", "site profiles must be a mapping")
    for profile_id, profile in profiles.items():
        validate_safe_identifier(
            profile_id,
            field="profile_id",
            code="E_CAMPAIGN_SITE_INVALID",
        )
        if not isinstance(profile, Mapping):
            raise CampaignError("E_CAMPAIGN_SITE_INVALID", "site profile must be a mapping")
        if executor == "local":
            architecture = profile.get("architecture")
            accelerators = profile.get("accelerators_per_task")
            if not isinstance(architecture, str) or architecture.upper() != "CPU":
                raise CampaignError(
                    "E_CAMPAIGN_SITE_INVALID",
                    f"{site_id}.{profile_id}: scheduler=local requires architecture=CPU",
                )
            if (
                isinstance(accelerators, bool)
                or not isinstance(accelerators, int)
                or accelerators != 0
            ):
                raise CampaignError(
                    "E_CAMPAIGN_SITE_INVALID",
                    f"{site_id}.{profile_id}: scheduler=local requires accelerators_per_task=0",
                )
    raw["site_id"] = site_id
    return raw


def _positive_int_field(
    profile: Mapping[str, Any], field: str, *, default: int, allow_zero: bool = False
) -> int:
    value = profile.get(field, default)
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        qualifier = ">= 0" if allow_zero else "> 0"
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", f"{field} must be {qualifier}")
    return value


def _accelerators_from_profile(profile: Mapping[str, Any]) -> int:
    directives = profile.get("directives")
    if not isinstance(directives, Mapping):
        return 0
    accelerator_request = directives.get("gres")
    if not isinstance(accelerator_request, str):
        return 0
    parts = accelerator_request.split(":")
    if not parts or parts[0].strip().lower() != "gpu":
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 1


def profile_resource_metadata(
    *,
    site_id: str,
    profile_id: str,
    profile: Mapping[str, Any],
    executor: str = "slurm",
) -> dict[str, Any]:
    """Normalize the scheduler-independent resource contract of one profile."""

    if executor not in _SUPPORTED_EXECUTORS:
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", f"unsupported scheduler: {executor!r}")
    raw_directives = profile.get("directives")
    if executor == "slurm":
        if not isinstance(raw_directives, Mapping):
            raise CampaignError("E_CAMPAIGN_SITE_INVALID", "profile directives must be a mapping")
        directives: Mapping[str, Any] = raw_directives
    else:
        if raw_directives is not None and not isinstance(raw_directives, Mapping):
            raise CampaignError("E_CAMPAIGN_SITE_INVALID", "profile directives must be a mapping")
        directives = raw_directives or {}
    accelerators = _positive_int_field(
        profile,
        "accelerators_per_task",
        default=_accelerators_from_profile(profile),
        allow_zero=True,
    )
    architecture = profile.get("architecture")
    if architecture is None:
        constraint = directives.get("constraint")
        architecture = str(constraint).split("-")[0].upper() if constraint else "CPU"
    if not isinstance(architecture, str) or not architecture.strip():
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", "architecture must be non-empty")
    if accelerators == 0 and architecture.upper() != "CPU":
        raise CampaignError(
            "E_CAMPAIGN_SITE_INVALID",
            f"{site_id}.{profile_id} declares architecture={architecture} without an accelerator",
        )
    if executor == "local" and (architecture.upper() != "CPU" or accelerators != 0):
        raise CampaignError(
            "E_CAMPAIGN_SITE_INVALID",
            f"{site_id}.{profile_id}: scheduler=local supports CPU resources only",
        )

    configured_time = (
        directives.get("time")
        if executor == "slurm"
        else profile.get("walltime", profile.get("max_walltime"))
    )
    if not isinstance(configured_time, str):
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", f"{profile_id}.walltime is required")
    configured_walltime = parse_duration(configured_time)
    raw_max_walltime = profile.get("max_walltime", configured_time)
    if not isinstance(raw_max_walltime, str):
        raise CampaignError(
            "E_CAMPAIGN_SITE_INVALID", f"{profile_id}.max_walltime must be a string"
        )
    max_walltime = parse_duration(raw_max_walltime)
    if configured_walltime > max_walltime:
        raise CampaignError(
            "E_CAMPAIGN_SITE_INVALID",
            f"{profile_id}.configured walltime exceeds max_walltime",
        )
    fixed_walltime = profile.get("fixed_walltime", False)
    if not isinstance(fixed_walltime, bool):
        raise CampaignError(
            "E_CAMPAIGN_SITE_INVALID",
            f"{profile_id}.fixed_walltime must be a boolean",
        )

    initial = _positive_int_field(
        profile,
        "initial_concurrency",
        default=_positive_int_field(profile, "concurrency", default=1),
    )
    if initial != profile.get("concurrency"):
        raise CampaignError(
            "E_CAMPAIGN_SITE_INVALID",
            f"{profile_id}.initial_concurrency must equal concurrency for generated arrays",
        )
    promoted = _positive_int_field(profile, "promoted_concurrency", default=initial)
    if promoted < initial:
        raise CampaignError(
            "E_CAMPAIGN_SITE_INVALID", "promoted_concurrency must be >= initial_concurrency"
        )
    minimum_successes = _positive_int_field(
        profile, "promotion_min_successes", default=200, allow_zero=True
    )
    maximum_failure_rate = profile.get("promotion_max_failure_rate", 0.02)
    if (
        isinstance(maximum_failure_rate, bool)
        or not isinstance(maximum_failure_rate, int | float)
        or not 0.0 <= float(maximum_failure_rate) <= 1.0
    ):
        raise CampaignError(
            "E_CAMPAIGN_SITE_INVALID", "promotion_max_failure_rate must be in [0, 1]"
        )
    metadata = {
        "site_id": site_id,
        "profile_id": profile_id,
        "architecture": architecture.upper(),
        "accelerators_per_task": accelerators,
        "configured_walltime_seconds": configured_walltime,
        "max_walltime_seconds": max_walltime,
        "initial_concurrency": initial,
        "promoted_concurrency": promoted,
        "promotion_min_successes": minimum_successes,
        "promotion_max_failure_rate": float(maximum_failure_rate),
    }
    if fixed_walltime:
        metadata["fixed_walltime"] = True
    return metadata


def plan_resource_sites(
    *,
    site_paths: Sequence[Path],
    tasks: list[CampaignTask],
    campaign_dir: Path,
    allow_template_placeholders: bool = False,
) -> Path:
    """Write neutral resource metadata and deterministic task index groups."""

    manifest_sha256 = sha256_file(campaign_dir / "manifest.jsonl")
    resource_catalog: list[dict[str, Any]] = []
    array_indices: list[dict[str, Any]] = []
    coverage: Counter[int] = Counter()
    for site_path in site_paths:
        site = load_execution_site(
            Path(site_path),
            allow_template_placeholders=allow_template_placeholders,
        )
        site_id = str(site["site_id"])
        executor = str(site["scheduler"])
        profiles = site["profiles"]
        grouped: dict[str, list[CampaignTask]] = defaultdict(list)
        for task in tasks:
            if task.assigned_site in {site_id, "any"}:
                grouped[task.resource_profile].append(task)
        for profile_id, profile_tasks in sorted(grouped.items()):
            raw_profile = profiles.get(profile_id)
            if not isinstance(raw_profile, Mapping):
                raise CampaignError(
                    "E_CAMPAIGN_SITE_INVALID",
                    f"site {site_id} has no profile {profile_id}",
                )
            resource_catalog.append(
                profile_resource_metadata(
                    site_id=site_id,
                    profile_id=profile_id,
                    profile=raw_profile,
                    executor=executor,
                )
            )
            coverage.update(task.task_index for task in profile_tasks)
            if executor == "local":
                continue
            block_size = raw_profile.get("array_block_size", 500)
            if (
                isinstance(block_size, bool)
                or not isinstance(block_size, int)
                or not 1 <= block_size <= 500
            ):
                raise CampaignError(
                    "E_CAMPAIGN_SITE_INVALID",
                    f"{profile_id}.array_block_size must be in [1, 500]",
                )
            blocks = [
                profile_tasks[start : start + block_size]
                for start in range(0, len(profile_tasks), block_size)
            ]
            for block_index, block_tasks in enumerate(blocks):
                suffix = "" if len(blocks) == 1 else f".block{block_index:03d}"
                index_filename = f"{site_id}.{profile_id}{suffix}.indices"
                index_text = "".join(f"{task.task_index}\n" for task in block_tasks)
                index_sha256 = sha256_bytes(index_text.encode("utf-8"))
                write_text_atomic(campaign_dir / "profiles" / index_filename, index_text)
                array_indices.append(
                    {
                        "site_id": site_id,
                        "profile_id": profile_id,
                        "block": block_index,
                        "path": f"profiles/{index_filename}",
                        "sha256": index_sha256,
                        "task_count": len(block_tasks),
                    }
                )
    uncovered = sorted(task.task_index for task in tasks if coverage[task.task_index] != 1)
    if uncovered:
        raise CampaignError(
            "E_CAMPAIGN_RETRY_SITE_COVERAGE",
            "resource profiles must cover every campaign task exactly once; "
            f"invalid task indices: {uncovered}",
        )
    catalog_path = campaign_dir / "profiles" / "resources.json"
    atomic_write_json(
        catalog_path,
        {
            "schema_version": 1,
            "manifest_sha256": manifest_sha256,
            "array_indices": sorted(
                array_indices,
                key=lambda item: (str(item["site_id"]), str(item["path"])),
            ),
            "resources": sorted(
                resource_catalog,
                key=lambda item: (str(item["site_id"]), str(item["profile_id"])),
            ),
        },
    )
    return catalog_path
