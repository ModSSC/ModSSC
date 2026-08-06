from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bench.campaign.errors import CampaignError
from bench.campaign.identifiers import validate_safe_identifier
from bench.campaign.manifest import load_manifest, sha256_file, write_text_atomic

from .resources import load_execution_site, plan_resource_sites, profile_resource_metadata

_DIRECTIVE_ORDER = (
    "account",
    "constraint",
    "partition",
    "qos",
    "nodes",
    "ntasks",
    "gres",
    "cpus-per-task",
    "mem",
    "time",
    "signal",
    "hint",
)


def _render_directives(profile: Mapping[str, Any]) -> list[str]:
    directives = profile.get("directives")
    if not isinstance(directives, Mapping):
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", "profile directives must be a mapping")
    unknown = sorted(set(directives) - set(_DIRECTIVE_ORDER))
    if unknown:
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", f"unsupported Slurm directives: {unknown}")
    lines: list[str] = []
    for key in _DIRECTIVE_ORDER:
        if key not in directives:
            continue
        value = directives[key]
        if (
            not isinstance(value, str | int)
            or str(value).strip() == ""
            or any(character in str(value) for character in ("\n", "\r", "\0"))
        ):
            raise CampaignError(
                "E_CAMPAIGN_SITE_INVALID", f"invalid Slurm directive {key}={value!r}"
            )
        lines.append(f"#SBATCH --{key}={value}")
    return lines


def render_slurm_wrapper(
    *,
    campaign_id: str,
    site: Mapping[str, Any],
    profile_id: str,
    profile: Mapping[str, Any],
    task_count: int,
    index_filename: str,
    index_sha256: str,
    manifest_sha256: str,
    resource_profile_id: str,
    architecture: str,
) -> str:
    """Render one Slurm array wrapper from an already neutral campaign plan."""

    campaign_id = validate_safe_identifier(
        campaign_id, field="campaign_id", code="E_CAMPAIGN_SITE_INVALID"
    )
    profile_id = validate_safe_identifier(
        profile_id, field="profile_id", code="E_CAMPAIGN_SITE_INVALID"
    )
    resource_profile_id = validate_safe_identifier(
        resource_profile_id,
        field="resource_profile_id",
        code="E_CAMPAIGN_SITE_INVALID",
    )
    concurrency = profile.get("concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", f"{profile_id}.concurrency must be > 0")
    site_setup = site.get("setup", [])
    profile_setup = profile.get("setup", [])
    if not isinstance(site_setup, list) or any(not isinstance(line, str) for line in site_setup):
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", "site setup must be a list[str]")
    if not isinstance(profile_setup, list) or any(
        not isinstance(line, str) for line in profile_setup
    ):
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", "profile setup must be a list[str]")
    setup = [*site_setup, *profile_setup]
    if any(any(character in line for character in ("\n", "\r", "\0")) for line in setup):
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", "setup entries must be single shell lines")
    environment_digest = site.get("environment_lock_sha256", "from_environment")
    if (
        not isinstance(environment_digest, str)
        or not environment_digest
        or any(character in environment_digest for character in ("\n", "\r", "\0"))
    ):
        raise CampaignError("E_CAMPAIGN_SITE_INVALID", "environment_lock_sha256 must be non-empty")
    site_id = validate_safe_identifier(
        site["site_id"], field="site_id", code="E_CAMPAIGN_SITE_INVALID"
    )
    environment_lines = (
        [': "${MODSSC_ENVIRONMENT_LOCK_SHA256:?Pin the runtime environment digest}"']
        if environment_digest == "from_environment"
        else [f"export MODSSC_ENVIRONMENT_LOCK_SHA256={shlex.quote(environment_digest)}"]
    )
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={campaign_id[:40]}-{profile_id[:24]}",
        "#SBATCH --output=logs/%x_%A_%a.log",
        f"#SBATCH --array=0-{task_count - 1}%{concurrency}",
        *_render_directives(profile),
        "",
        "set -euo pipefail",
        ': "${SLURM_JOB_ID:?This campaign wrapper must run inside a Slurm allocation}"',
        ': "${SLURMD_NODENAME:?This campaign wrapper must run on a Slurm compute node}"',
        ': "${MODSSC_ROOT:?Set MODSSC_ROOT to the repository root}"',
        ': "${MODSSC_CAMPAIGN_DIR:?Set MODSSC_CAMPAIGN_DIR to the generated campaign}"',
        ': "${MODSSC_CAMPAIGN_RESULTS:?Set MODSSC_CAMPAIGN_RESULTS to persistent storage}"',
        f"export MODSSC_CAMPAIGN_ID={shlex.quote(campaign_id)}",
        f"export MODSSC_CAMPAIGN_SITE_ID={shlex.quote(site_id)}",
        "export MODSSC_CAMPAIGN_SCHEDULER=slurm",
        f"export MODSSC_RESOURCE_PROFILE={shlex.quote(resource_profile_id)}",
        f"export MODSSC_EXPECTED_ACCELERATOR_ARCH={shlex.quote(architecture.lower())}",
        *setup,
        *environment_lines,
        f'export MODSSC_ARRAY_INDEX_FILE="$MODSSC_CAMPAIGN_DIR/profiles/{index_filename}"',
        f"export MODSSC_ARRAY_INDEX_SHA256={shlex.quote(index_sha256)}",
        'export MODSSC_CAMPAIGN_MANIFEST="$MODSSC_CAMPAIGN_DIR/manifest.jsonl"',
        f"export MODSSC_CAMPAIGN_MANIFEST_SHA256={shlex.quote(manifest_sha256)}",
        'export MODSSC_CAMPAIGN_META="$MODSSC_CAMPAIGN_DIR/manifest.meta.json"',
        'export MODSSC_CAMPAIGN_RESULT_ROOT="$MODSSC_CAMPAIGN_RESULTS/$MODSSC_CAMPAIGN_ID"',
        (
            'export MODSSC_CAMPAIGN_CHECKPOINT_ROOT="'
            "${MODSSC_CAMPAIGN_CHECKPOINTS:-$MODSSC_CAMPAIGN_RESULTS/checkpoints}/"
            '$MODSSC_CAMPAIGN_ID"'
        ),
        'exec "$MODSSC_ROOT/tools/hpc/slurm/array-task.sh"',
        "",
    ]
    return "\n".join(lines)


def render_slurm_sites(
    *,
    site_paths: Sequence[Path],
    campaign_dir: Path,
    submission_dir: Path | None = None,
    allow_template_placeholders: bool = False,
) -> list[Path]:
    """Render scheduler wrappers around an immutable neutral campaign."""

    campaign_dir = Path(campaign_dir).resolve(strict=True)
    meta, _ = load_manifest(campaign_dir / "manifest.jsonl", verify_digest=True)
    campaign_id = validate_safe_identifier(
        meta.get("campaign_id"), field="campaign_id", code="E_CAMPAIGN_SITE_INVALID"
    )
    catalog_path = campaign_dir / "profiles" / "resources.json"
    if not catalog_path.is_file():
        _, tasks = load_manifest(campaign_dir / "manifest.jsonl", verify_digest=True)
        plan_resource_sites(
            site_paths=site_paths,
            tasks=tasks,
            campaign_dir=campaign_dir,
            allow_template_placeholders=allow_template_placeholders,
        )
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignError(
            "E_CAMPAIGN_RESOURCE_CATALOG_INVALID",
            f"cannot read neutral resource plan: {catalog_path}",
        ) from exc
    manifest_sha256 = sha256_file(campaign_dir / "manifest.jsonl")
    if (
        not isinstance(catalog, dict)
        or catalog.get("schema_version") != 1
        or catalog.get("manifest_sha256") != manifest_sha256
        or not isinstance(catalog.get("resources"), list)
        or not isinstance(catalog.get("array_indices"), list)
    ):
        raise CampaignError(
            "E_CAMPAIGN_RESOURCE_CATALOG_INVALID",
            "neutral resource plan does not authenticate this campaign",
        )
    sites = {
        str(site["site_id"]): site
        for site in (
            load_execution_site(
                Path(path),
                allow_template_placeholders=allow_template_placeholders,
            )
            for path in site_paths
        )
    }
    resources = {(str(row["site_id"]), str(row["profile_id"])): row for row in catalog["resources"]}
    for (site_id, profile_id), resource in resources.items():
        site = sites.get(site_id)
        profile = None if site is None else site["profiles"].get(profile_id)
        if not isinstance(profile, Mapping):
            raise CampaignError(
                "E_CAMPAIGN_SITE_INVALID", f"site {site_id} has no profile {profile_id}"
            )
        expected = profile_resource_metadata(
            site_id=site_id,
            profile_id=profile_id,
            profile=profile,
            executor=str(site["scheduler"]),
        )
        if resource != expected:
            raise CampaignError(
                "E_CAMPAIGN_RESOURCE_CATALOG_INVALID",
                f"resource profile {site_id}.{profile_id} differs from the neutral plan",
            )
    block_counts: dict[tuple[str, str], int] = {}
    for row in catalog["array_indices"]:
        key = (str(row["site_id"]), str(row["profile_id"]))
        block_counts[key] = block_counts.get(key, 0) + 1

    generated: list[Path] = []
    output_root = (
        campaign_dir / "submit"
        if submission_dir is None
        else Path(submission_dir).resolve(strict=False)
    )
    for row in catalog["array_indices"]:
        site_id = str(row["site_id"])
        resource_profile_id = str(row["profile_id"])
        site = sites[site_id]
        if site.get("scheduler") != "slurm":
            continue
        profile = site["profiles"][resource_profile_id]
        block = int(row["block"])
        suffix = "" if block_counts[(site_id, resource_profile_id)] == 1 else f".block{block:03d}"
        wrapper_profile_id = f"{resource_profile_id}{suffix}"
        script = render_slurm_wrapper(
            campaign_id=campaign_id,
            site=site,
            profile_id=wrapper_profile_id,
            profile=profile,
            task_count=int(row["task_count"]),
            index_filename=Path(str(row["path"])).name,
            index_sha256=str(row["sha256"]),
            manifest_sha256=manifest_sha256,
            resource_profile_id=resource_profile_id,
            architecture=str(resources[(site_id, resource_profile_id)]["architecture"]),
        )
        index_path = campaign_dir / str(row["path"])
        if not index_path.is_file() or sha256_file(index_path) != row["sha256"]:
            raise CampaignError(
                "E_CAMPAIGN_RESOURCE_CATALOG_INVALID",
                f"array index is missing or changed: {row['path']}",
            )
        script_path = output_root / site_id / f"{wrapper_profile_id}.slurm"
        write_text_atomic(script_path, script)
        script_path.chmod(0o755)
        generated.append(script_path)
    output_root.parent.joinpath("logs").mkdir(parents=True, exist_ok=True)
    return generated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.hpc.slurm_renderer",
        description="Render Slurm submission wrappers for a neutral ModSSC campaign",
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--site", type=Path, action="append", required=True)
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=None,
        help="operational output root; defaults to CAMPAIGN_DIR/submit",
    )
    parser.add_argument("--allow-template-placeholders", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        scripts = render_slurm_sites(
            site_paths=args.site,
            campaign_dir=args.campaign_dir,
            submission_dir=args.submission_dir,
            allow_template_placeholders=args.allow_template_placeholders,
        )
    except CampaignError as exc:
        print(str(exc))
        return 2
    print(json.dumps({"scripts": [str(path) for path in scripts]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
