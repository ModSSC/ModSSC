from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class SubmissionError(RuntimeError):
    """Raised when an array chain cannot be validated or submitted safely."""


_ARRAY_RE = re.compile(r"#SBATCH[ \t]+--array=(0)-([0-9]+)%([1-9][0-9]*)[ \t]*")
_TIME_RE = re.compile(r"#SBATCH[ \t]+--time=([^ \t]+)[ \t]*")
_DURATION_RE = re.compile(
    r"(?:(?P<days>[0-9]+)-)?(?P<hours>[0-9]{2,}):(?P<minutes>[0-9]{2}):(?P<seconds>[0-9]{2})"
)
_CLI_DURATION_RE = re.compile(r"(?P<hours>[0-9]{2,}):(?P<minutes>[0-9]{2}):(?P<seconds>[0-9]{2})")
_SAFE_IDENTIFIER = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_EXPORT_PATTERNS = {
    "campaign_id": re.compile(rf"export[ \t]+MODSSC_CAMPAIGN_ID=({_SAFE_IDENTIFIER})[ \t]*"),
    "site_id": re.compile(rf"export[ \t]+MODSSC_CAMPAIGN_SITE_ID=({_SAFE_IDENTIFIER})[ \t]*"),
    "profile_id": re.compile(rf"export[ \t]+MODSSC_RESOURCE_PROFILE=({_SAFE_IDENTIFIER})[ \t]*"),
    "manifest_sha256": re.compile(
        r"export[ \t]+MODSSC_CAMPAIGN_MANIFEST_SHA256=([0-9a-f]{64})[ \t]*"
    ),
    "index_sha256": re.compile(r"export[ \t]+MODSSC_ARRAY_INDEX_SHA256=([0-9a-f]{64})[ \t]*"),
    "index_path": re.compile(
        r'export[ \t]+MODSSC_ARRAY_INDEX_FILE="\$MODSSC_CAMPAIGN_DIR/'
        r'(profiles/[A-Za-z0-9][A-Za-z0-9._-]*)"[ \t]*'
    ),
}
_SBATCH_ID_RE = re.compile(r"(?P<job_id>[1-9][0-9]*)(?:;[A-Za-z0-9][A-Za-z0-9._-]*)?")


@dataclass(frozen=True)
class Wrapper:
    path: Path
    campaign_root: Path
    campaign_id: str
    site_id: str
    profile_id: str
    manifest_sha256: str
    index_sha256: str
    index_path: str
    array_end: int
    source_throttle: int
    embedded_walltime_seconds: int
    block_number: int | None

    @property
    def array_range(self) -> str:
        return f"0-{self.array_end}"


@dataclass(frozen=True)
class PreflightAuthorization:
    path: Path
    report_sha256: str
    validated_at: datetime


def _single_match(pattern: re.Pattern[str], text: str, *, label: str, path: Path) -> str:
    matches = [match.group(1) for line in text.splitlines() if (match := pattern.fullmatch(line))]
    if len(matches) != 1:
        raise SubmissionError(f"{path}: expected exactly one {label}, found {len(matches)}")
    return matches[0]


def _duration_seconds(value: str, *, cli: bool) -> int:
    match = (_CLI_DURATION_RE if cli else _DURATION_RE).fullmatch(value)
    if match is None:
        expected = "HH:MM:SS" if cli else "[D-]HH:MM:SS"
        raise SubmissionError(f"invalid walltime {value!r}; expected {expected}")
    days = 0 if cli else int(match.groupdict().get("days") or 0)
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise SubmissionError(f"invalid walltime {value!r}; minutes and seconds must be < 60")
    total = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    if total <= 0:
        raise SubmissionError("walltime must be positive")
    return total


def _format_duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 60 * 60)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_wrapper(path: Path) -> Wrapper:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SubmissionError(f"cannot resolve wrapper {path}: {exc}") from exc
    if not resolved.is_file():
        raise SubmissionError(f"wrapper is not a regular file: {resolved}")
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SubmissionError(f"cannot read wrapper {resolved}: {exc}") from exc

    array_matches = [match for line in text.splitlines() if (match := _ARRAY_RE.fullmatch(line))]
    if len(array_matches) != 1:
        raise SubmissionError(
            f"{resolved}: expected exactly one generated #SBATCH --array directive, "
            f"found {len(array_matches)}"
        )
    array_end = int(array_matches[0].group(2))
    source_throttle = int(array_matches[0].group(3))
    if array_end >= 500:
        raise SubmissionError(f"{resolved}: array block contains more than 500 tasks")

    embedded_time = _single_match(_TIME_RE, text, label="#SBATCH --time directive", path=resolved)
    embedded_seconds = _duration_seconds(embedded_time, cli=False)
    fields = {
        name: _single_match(pattern, text, label=name, path=resolved)
        for name, pattern in _EXPORT_PATTERNS.items()
    }
    if any(".." in fields[name] for name in ("campaign_id", "site_id", "profile_id", "index_path")):
        raise SubmissionError(f"{resolved}: generated identifiers must not contain '..'")
    profile_id = fields["profile_id"]
    plain_name = f"{profile_id}.slurm"
    block_match = re.fullmatch(rf"{re.escape(profile_id)}\.block([0-9]{{3}})\.slurm", resolved.name)
    if resolved.name == plain_name:
        block_number = None
    elif block_match is not None:
        block_number = int(block_match.group(1))
    else:
        raise SubmissionError(
            f"{resolved}: filename does not match exported resource profile {profile_id!r}"
        )

    if resolved.parent.parent.name != "submit":
        raise SubmissionError(f"{resolved}: wrapper must be located under <campaign>/submit/<site>")
    campaign_root = resolved.parent.parent.parent
    if resolved.parent.name != fields["site_id"]:
        raise SubmissionError(f"{resolved}: directory and exported site id differ")
    return Wrapper(
        path=resolved,
        campaign_root=campaign_root,
        campaign_id=fields["campaign_id"],
        site_id=fields["site_id"],
        profile_id=profile_id,
        manifest_sha256=fields["manifest_sha256"],
        index_sha256=fields["index_sha256"],
        index_path=fields["index_path"],
        array_end=array_end,
        source_throttle=source_throttle,
        embedded_walltime_seconds=embedded_seconds,
        block_number=block_number,
    )


def _mapping_list(value: Any, *, field: str, catalog_path: Path) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SubmissionError(f"{catalog_path}: {field} must be a list of objects")
    return value


def _load_catalog(campaign_root: Path) -> tuple[Path, dict[str, Any]]:
    catalog_path = campaign_root / "profiles" / "resources.json"
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SubmissionError(
            f"cannot read generated resource catalog {catalog_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise SubmissionError(f"{catalog_path}: unsupported resource catalog schema")
    return catalog_path, raw


def _validate_preflight_report(
    path: Path,
    *,
    campaign_id: str,
    manifest_sha256: str,
    architecture: str,
    now: datetime | None = None,
) -> PreflightAuthorization:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SubmissionError(f"cannot resolve preflight report {path}: {exc}") from exc
    if not resolved.is_file():
        raise SubmissionError(f"preflight report is not a regular file: {resolved}")
    if any(character in str(resolved) for character in (",", "\n", "\r", "\0")):
        raise SubmissionError("preflight report path is unsafe for Slurm --export syntax")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"cannot read preflight report {resolved}: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise SubmissionError(f"{resolved}: preflight schema_version must equal 1")
    if report.get("status") != "pass":
        raise SubmissionError(f"{resolved}: preflight status is not pass")
    if report.get("campaign_id") != campaign_id:
        raise SubmissionError(f"{resolved}: preflight campaign_id differs from the wrappers")
    if report.get("manifest_sha256") != manifest_sha256:
        raise SubmissionError(f"{resolved}: preflight manifest digest differs from the wrappers")
    if report.get("required_architecture") != architecture:
        raise SubmissionError(
            f"{resolved}: preflight architecture differs; expected {architecture}"
        )
    try:
        created_at = datetime.fromisoformat(str(report["created_at"]))
        expires_at = datetime.fromisoformat(str(report["expires_at"]))
    except (KeyError, ValueError) as exc:
        raise SubmissionError(
            f"{resolved}: preflight created_at/expires_at must be ISO-8601 timestamps"
        ) from exc
    if (
        created_at.tzinfo is None
        or created_at.utcoffset() is None
        or expires_at.tzinfo is None
        or expires_at.utcoffset() is None
    ):
        raise SubmissionError(f"{resolved}: preflight timestamps must include a timezone")
    max_age = report.get("max_allocation_age_hours")
    if (
        isinstance(max_age, bool)
        or not isinstance(max_age, int | float)
        or not math.isfinite(float(max_age))
        or float(max_age) <= 0.0
    ):
        raise SubmissionError(f"{resolved}: preflight maximum allocation age is invalid")
    created_at = created_at.astimezone(UTC)
    expires_at = expires_at.astimezone(UTC)
    checked_at = (now or datetime.now(UTC)).astimezone(UTC)
    if expires_at <= created_at or expires_at > created_at + timedelta(hours=float(max_age)):
        raise SubmissionError(f"{resolved}: preflight validity interval is invalid")
    if created_at > checked_at:
        raise SubmissionError(f"{resolved}: preflight creation time is in the future")
    if checked_at >= expires_at:
        raise SubmissionError(f"{resolved}: preflight report has expired")
    return PreflightAuthorization(
        path=resolved,
        report_sha256=_sha256_file(resolved),
        validated_at=checked_at,
    )


def validate_chain(
    paths: list[Path], *, throttle: int, walltime: str, preflight_report: Path
) -> tuple[list[Wrapper], PreflightAuthorization]:
    """Validate an ordered, single-profile generated-wrapper chain before any submission."""

    if not paths:
        raise SubmissionError("at least one wrapper is required")
    if isinstance(throttle, bool) or not 1 <= throttle <= 10_000:
        raise SubmissionError("throttle must be an integer in [1, 10000]")
    requested_walltime_seconds = _duration_seconds(walltime, cli=True)
    wrappers = [_parse_wrapper(path) for path in paths]
    resolved_paths = [wrapper.path for wrapper in wrappers]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise SubmissionError("the wrapper list contains a duplicate path")

    first = wrappers[0]
    common_fields = (
        "campaign_root",
        "campaign_id",
        "site_id",
        "profile_id",
        "manifest_sha256",
        "source_throttle",
        "embedded_walltime_seconds",
    )
    for wrapper in wrappers[1:]:
        different = [
            field for field in common_fields if getattr(wrapper, field) != getattr(first, field)
        ]
        if different:
            raise SubmissionError(
                f"{wrapper.path}: wrapper differs from the first block in {', '.join(different)}"
            )

    block_numbers = [wrapper.block_number for wrapper in wrappers]
    if len(wrappers) > 1 and any(number is None for number in block_numbers):
        raise SubmissionError("a multi-wrapper chain must contain only numbered block wrappers")
    numeric_blocks = [number for number in block_numbers if number is not None]
    if numeric_blocks and numeric_blocks != list(
        range(numeric_blocks[0], numeric_blocks[0] + len(numeric_blocks))
    ):
        raise SubmissionError("numbered wrappers must be listed in strictly consecutive order")

    catalog_path, catalog = _load_catalog(first.campaign_root)
    if catalog.get("manifest_sha256") != first.manifest_sha256:
        raise SubmissionError(f"{catalog_path}: manifest digest differs from the wrappers")
    manifest_path = first.campaign_root / "manifest.jsonl"
    try:
        actual_manifest_sha256 = _sha256_file(manifest_path)
    except OSError as exc:
        raise SubmissionError(f"cannot hash campaign manifest {manifest_path}: {exc}") from exc
    if actual_manifest_sha256 != first.manifest_sha256:
        raise SubmissionError(f"campaign manifest SHA-256 mismatch: {manifest_path}")

    resources = _mapping_list(
        catalog.get("resources"), field="resources", catalog_path=catalog_path
    )
    matching_resources = [
        item
        for item in resources
        if item.get("site_id") == first.site_id and item.get("profile_id") == first.profile_id
    ]
    if len(matching_resources) != 1:
        raise SubmissionError(
            f"{catalog_path}: expected one resource record for {first.site_id}.{first.profile_id}"
        )
    resource = matching_resources[0]
    architecture = resource.get("architecture")
    initial = resource.get("initial_concurrency")
    maximum_walltime = resource.get("max_walltime_seconds")
    configured_walltime = resource.get("configured_walltime_seconds")
    fixed_walltime = resource.get("fixed_walltime", False)
    if isinstance(initial, bool) or not isinstance(initial, int) or initial <= 0:
        raise SubmissionError(f"{catalog_path}: invalid initial_concurrency")
    if not isinstance(architecture, str) or architecture not in {"CPU", "A100", "V100", "H100"}:
        raise SubmissionError(f"{catalog_path}: invalid resource architecture")
    if first.source_throttle != initial:
        raise SubmissionError(
            "wrapper throttle differs from the resource catalogue initial concurrency"
        )
    if throttle != initial:
        raise SubmissionError(
            f"--throttle must equal the pre-registered initial concurrency ({initial}); "
            "promote only the active job after daily-report approval"
        )
    if (
        isinstance(maximum_walltime, bool)
        or not isinstance(maximum_walltime, int)
        or maximum_walltime <= 0
    ):
        raise SubmissionError(f"{catalog_path}: invalid max_walltime_seconds")
    if not isinstance(fixed_walltime, bool):
        raise SubmissionError(f"{catalog_path}: invalid fixed_walltime")
    if configured_walltime != first.embedded_walltime_seconds:
        raise SubmissionError("wrapper walltime differs from the resource catalogue")
    if fixed_walltime and requested_walltime_seconds != configured_walltime:
        raise SubmissionError(
            f"--time must equal the fixed profile walltime "
            f"({_format_duration(configured_walltime)})"
        )
    if requested_walltime_seconds > maximum_walltime:
        raise SubmissionError(
            f"requested walltime {walltime} exceeds the profile cap encoded in resources.json"
        )

    index_records = _mapping_list(
        catalog.get("array_indices"), field="array_indices", catalog_path=catalog_path
    )
    for wrapper in wrappers:
        matches = [item for item in index_records if item.get("path") == wrapper.index_path]
        if len(matches) != 1:
            raise SubmissionError(
                f"{catalog_path}: no unique index record for {wrapper.index_path}"
            )
        record = matches[0]
        if (
            record.get("site_id") != wrapper.site_id
            or record.get("profile_id") != wrapper.profile_id
            or record.get("sha256") != wrapper.index_sha256
            or record.get("task_count") != wrapper.array_end + 1
        ):
            raise SubmissionError(f"{catalog_path}: index record differs from {wrapper.path}")
        index_path = wrapper.campaign_root / wrapper.index_path
        try:
            actual_index_sha256 = _sha256_file(index_path)
        except OSError as exc:
            raise SubmissionError(f"cannot hash array index {index_path}: {exc}") from exc
        if actual_index_sha256 != wrapper.index_sha256:
            raise SubmissionError(f"array index SHA-256 mismatch: {index_path}")
    resolved_preflight = _validate_preflight_report(
        preflight_report,
        campaign_id=first.campaign_id,
        manifest_sha256=first.manifest_sha256,
        architecture=architecture,
    )
    return wrappers, resolved_preflight


def _command_for_wrapper(
    wrapper: Wrapper,
    *,
    throttle: int,
    walltime: str,
    preflight_authorization: PreflightAuthorization,
    dependency_job_id: str | None,
) -> list[str]:
    command = [
        "sbatch",
        "--parsable",
        f"--array={wrapper.array_range}%{throttle}",
        f"--time={walltime}",
        f"--chdir={wrapper.campaign_root}",
        (f"--export=ALL,MODSSC_PREFLIGHT_REPORT={preflight_authorization.path}"),
    ]
    if dependency_job_id is not None:
        command.append(f"--dependency=afterok:{dependency_job_id}")
    command.append(str(wrapper.path))
    return command


def submit_chain(
    wrappers: list[Wrapper],
    *,
    throttle: int,
    walltime: str,
    preflight_authorization: PreflightAuthorization,
    dry_run: bool,
) -> list[str]:
    """Submit validated wrappers sequentially, or print the exact dry-run plan."""

    submitted_ids: list[str] = []
    for index, wrapper in enumerate(wrappers):
        if dry_run:
            dependency = None if index == 0 else f"<job-id-{index - 1}>"
            command = _command_for_wrapper(
                wrapper,
                throttle=throttle,
                walltime=walltime,
                preflight_authorization=preflight_authorization,
                dependency_job_id=dependency,
            )
            print(f"DRY-RUN[{index}] {shlex.join(command)}")
            continue

        dependency = submitted_ids[-1] if submitted_ids else None
        command = _command_for_wrapper(
            wrapper,
            throttle=throttle,
            walltime=walltime,
            preflight_authorization=preflight_authorization,
            dependency_job_id=dependency,
        )
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError as exc:
            raise SubmissionError(f"cannot execute sbatch for {wrapper.path}: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no scheduler output"
            raise SubmissionError(
                f"sbatch failed for {wrapper.path} with status {completed.returncode}: {detail}"
            )
        output = completed.stdout
        if output.endswith("\n"):
            output = output[:-1]
        match = _SBATCH_ID_RE.fullmatch(output)
        if match is None:
            raise SubmissionError(
                f"sbatch returned an invalid parsable job id: {completed.stdout!r}"
            )
        job_id = match.group("job_id")
        submitted_ids.append(job_id)
        dependency_text = "none" if dependency is None else f"afterok:{dependency}"
        print(
            f"submitted job_id={job_id} dependency={dependency_text} "
            f"wrapper={json.dumps(str(wrapper.path))}"
        )
    return submitted_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit generated Slurm array blocks as one fail-closed afterok chain with a single "
            "global throttle. Preflight freshness is checked again when each task starts."
        )
    )
    parser.add_argument(
        "--throttle",
        required=True,
        type=int,
        help="initial global array throttle; must match profiles/resources.json",
    )
    parser.add_argument(
        "--time",
        required=True,
        dest="walltime",
        help="calibrated walltime in HH:MM:SS, passed explicitly to every sbatch call",
    )
    parser.add_argument(
        "--preflight-report",
        required=True,
        type=Path,
        help="passing, unexpired architecture-specific preflight report exported to every job",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate everything and print commands without invoking sbatch",
    )
    parser.add_argument(
        "wrappers",
        nargs="+",
        type=Path,
        help="explicit ordered wrappers from one generated campaign/site/profile",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        wrappers, preflight_authorization = validate_chain(
            args.wrappers,
            throttle=args.throttle,
            walltime=args.walltime,
            preflight_report=args.preflight_report,
        )
        submit_chain(
            wrappers,
            throttle=args.throttle,
            walltime=args.walltime,
            preflight_authorization=preflight_authorization,
            dry_run=args.dry_run,
        )
    except SubmissionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module entry point
    raise SystemExit(main())
