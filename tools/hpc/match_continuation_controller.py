from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from bench.campaign.errors import CampaignError
from bench.campaign.manifest import load_manifest, sha256_file
from bench.campaign.reconcile import reconcile_campaign
from bench.utils.hashing import stable_json_dumps
from bench.utils.io import atomic_write_json
from tools.hpc.slurm_renderer import render_slurm_sites, render_slurm_wrapper

_JOB_ID_RE = re.compile(r"(?P<job_id>[1-9][0-9]*)(?:;[A-Za-z0-9._-]+)?")
_SAFE_JOB_ID_RE = re.compile(r"[1-9][0-9]*")
_SNAPSHOT_NAME_RE = re.compile(r"(?P<sequence>[0-9]{6})-(?P<digest>[0-9a-f]{64})\.json")
_MATCH_METHODS = {"fixmatch", "flexmatch", "free_match", "softmatch"}
_MATCH_CONTINUATION_PROFILES = {
    "h100_long": {
        "concurrency": 5,
        "walltime": "100:00:00",
        "walltime_seconds": 360_000,
        "planned_segment_seconds": 288_000,
    },
    "h100_long_adaptive": {
        "concurrency": 9,
        "walltime": "100:00:00",
        "walltime_seconds": 360_000,
        "planned_segment_seconds": 288_000,
    },
    "h100_t3_adaptive": {
        "concurrency": 9,
        "walltime": "20:00:00",
        "walltime_seconds": 72_000,
        "planned_segment_seconds": 68_400,
    },
}
_TERMINAL_STATUSES = {"complete", "blocked", "max_segments_exceeded"}
_SBATCH_DIRECTIVES = (
    "account",
    "constraint",
    "partition",
    "qos",
    "nodes",
    "ntasks",
    "gres",
    "cpus-per-task",
    "mem",
    "hint",
)


class ControllerError(RuntimeError):
    """Raised when a continuation chain cannot be advanced safely."""


class ControllerBusy(RuntimeError):
    """Raised when another controller process owns the campaign lock."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_absolute_path(value: str | Path, *, field: str, must_exist: bool) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ControllerError(f"{field} must be an absolute path")
    resolved = path.resolve()
    if any(character in str(resolved) for character in (",", "\n", "\r", "\0")):
        raise ControllerError(f"{field} is unsafe for Slurm export")
    if must_exist and not resolved.exists():
        raise ControllerError(f"{field} does not exist: {resolved}")
    return resolved


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(dict(payload)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ControllerConfig:
    schema_version: int
    controller_id: str
    repo_root: str
    campaign_dir: str
    source_manifest_sha256: str
    source_meta_sha256: str
    result_root: str
    state_dir: str
    site_path: str
    site_sha256: str
    allocation_path: str
    environment_manifest_path: str
    environment_manifest_sha256: str
    checkpoint_base: str
    max_segments: int
    controller_profile: str

    @property
    def path_fields(self) -> tuple[str, ...]:
        return (
            "repo_root",
            "campaign_dir",
            "result_root",
            "state_dir",
            "site_path",
            "allocation_path",
            "environment_manifest_path",
            "checkpoint_base",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def build(
        cls,
        *,
        repo_root: Path,
        campaign_dir: Path,
        result_root: Path,
        state_dir: Path,
        site_path: Path,
        allocation_path: Path,
        environment_manifest_path: Path,
        checkpoint_base: Path,
        max_segments: int,
        controller_profile: str,
    ) -> ControllerConfig:
        if isinstance(max_segments, bool) or max_segments < 2:
            raise ControllerError("max_segments must be an integer greater than or equal to 2")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "repo_root": str(_safe_absolute_path(repo_root, field="repo_root", must_exist=True)),
            "campaign_dir": str(
                _safe_absolute_path(campaign_dir, field="campaign_dir", must_exist=True)
            ),
            "result_root": str(
                _safe_absolute_path(result_root, field="result_root", must_exist=False)
            ),
            "state_dir": str(_safe_absolute_path(state_dir, field="state_dir", must_exist=False)),
            "site_path": str(_safe_absolute_path(site_path, field="site_path", must_exist=True)),
            "allocation_path": str(
                _safe_absolute_path(
                    allocation_path,
                    field="allocation_path",
                    must_exist=True,
                )
            ),
            "environment_manifest_path": str(
                _safe_absolute_path(
                    environment_manifest_path,
                    field="environment_manifest_path",
                    must_exist=True,
                )
            ),
            "checkpoint_base": str(
                _safe_absolute_path(
                    checkpoint_base,
                    field="checkpoint_base",
                    must_exist=False,
                )
            ),
            "max_segments": int(max_segments),
            "controller_profile": str(controller_profile),
        }
        campaign_path = Path(payload["campaign_dir"])
        try:
            payload["source_manifest_sha256"] = sha256_file(campaign_path / "manifest.jsonl")
            payload["source_meta_sha256"] = sha256_file(campaign_path / "manifest.meta.json")
        except OSError as exc:
            raise ControllerError("source campaign manifest files are unavailable") from exc
        payload["site_sha256"] = sha256_file(Path(payload["site_path"]))
        payload["environment_manifest_sha256"] = sha256_file(
            Path(payload["environment_manifest_path"])
        )
        payload["controller_id"] = _sha256_payload(payload)
        config = cls(**payload)
        config.validate()
        return config

    @classmethod
    def load(cls, path: Path) -> ControllerConfig:
        config_path = _safe_absolute_path(path, field="config", must_exist=True)
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ControllerError(f"cannot read controller config: {config_path}") from exc
        try:
            config = cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ControllerError("invalid controller config fields") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ControllerError("controller config schema_version must equal 1")
        if isinstance(self.max_segments, bool) or self.max_segments < 2:
            raise ControllerError("max_segments must be at least 2")
        if not self.controller_profile:
            raise ControllerError("controller_profile must be non-empty")
        for field in self.path_fields:
            must_exist = field not in {"result_root", "state_dir", "checkpoint_base"}
            _safe_absolute_path(
                getattr(self, field),
                field=field,
                must_exist=must_exist,
            )
        if sha256_file(Path(self.site_path)) != self.site_sha256:
            raise ControllerError("site profile digest mismatch")
        if sha256_file(Path(self.environment_manifest_path)) != self.environment_manifest_sha256:
            raise ControllerError("environment manifest digest mismatch")
        campaign_path = Path(self.campaign_dir)
        try:
            source_manifest_sha256 = sha256_file(campaign_path / "manifest.jsonl")
            source_meta_sha256 = sha256_file(campaign_path / "manifest.meta.json")
        except OSError as exc:
            raise ControllerError("source campaign manifest files are unavailable") from exc
        if (
            source_manifest_sha256 != self.source_manifest_sha256
            or source_meta_sha256 != self.source_meta_sha256
        ):
            raise ControllerError("source campaign manifest digest mismatch")
        payload = self.to_dict()
        controller_id = payload.pop("controller_id")
        if controller_id != _sha256_payload(payload):
            raise ControllerError("controller config digest mismatch")
        _validate_campaign_binding(self)


def _load_site(config: ControllerConfig) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(Path(config.site_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ControllerError(f"cannot read site profile: {config.site_path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ControllerError("site profile schema_version must equal 1")
    if raw.get("scheduler") != "slurm":
        raise ControllerError("continuation controller requires a Slurm site")
    profiles = raw.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ControllerError("site profiles must be a mapping")
    if config.controller_profile != "h100_dev":
        raise ControllerError("controller_profile must be h100_dev")
    profile = profiles.get(config.controller_profile)
    if not isinstance(profile, Mapping):
        raise ControllerError("controller_profile must resolve to an H100 profile")
    directives = profile.get("directives")
    if (
        str(profile.get("architecture", "")).upper() != "H100"
        or profile.get("accelerators_per_task") != 1
        or profile.get("fixed_walltime") is not True
        or profile.get("max_walltime") != "02:00:00"
        or not isinstance(directives, Mapping)
        or directives.get("nodes") != 1
        or directives.get("ntasks") != 1
        or directives.get("gres") != "gpu:1"
        or directives.get("time") != "02:00:00"
    ):
        raise ControllerError("h100_dev must remain a fixed two-hour mono-H100 profile")
    return raw


def _validate_campaign_binding(config: ControllerConfig) -> tuple[dict[str, Any], list[Any]]:
    campaign_dir = Path(config.campaign_dir)
    meta, tasks = load_manifest(
        campaign_dir / "manifest.jsonl",
        meta_path=campaign_dir / "manifest.meta.json",
        verify_digest=True,
    )
    if not tasks:
        raise ControllerError("source campaign is empty")
    campaign_id = str(meta["campaign_id"])
    if Path(config.result_root).name != campaign_id:
        raise ControllerError("result_root must end with the immutable campaign_id")
    if any(task.method_id not in _MATCH_METHODS for task in tasks):
        raise ControllerError("continuation controller accepts only Match methods")
    site = _load_site(config)
    site_id = str(site.get("site_id"))
    profiles = site["profiles"]
    for task in tasks:
        profile = profiles.get(task.resource_profile)
        if task.assigned_site != site_id:
            raise ControllerError("every task must be assigned to the configured site")
        contract = _MATCH_CONTINUATION_PROFILES.get(task.resource_profile)
        if contract is None or not isinstance(profile, Mapping):
            raise ControllerError(
                "every continuation task must retain a registered Match continuation profile"
            )
        directives = profile.get("directives")
        setup = profile.get("setup")
        planned_segment = int(contract["planned_segment_seconds"])
        walltime = str(contract["walltime"])
        expected_concurrency = int(contract["concurrency"])
        if (
            str(profile.get("architecture", "")).upper() != "H100"
            or profile.get("accelerators_per_task") != 1
            or profile.get("fixed_walltime") is not True
            or profile.get("concurrency") != expected_concurrency
            or profile.get("initial_concurrency") != expected_concurrency
            or profile.get("max_walltime") != walltime
            or not isinstance(setup, list)
            or f"export MODSSC_PLANNED_SEGMENT_SECONDS={planned_segment}" not in setup
            or not isinstance(directives, Mapping)
            or directives.get("nodes") != 1
            or directives.get("ntasks") != 1
            or directives.get("gres") != "gpu:1"
            or directives.get("time") != walltime
            or directives.get("signal") != "B:USR1@300"
        ):
            raise ControllerError(
                "Match continuation profiles must remain registered fixed mono-H100 "
                "allocations with an authenticated planned segment"
            )
    return meta, tasks


class _ControllerLock(AbstractContextManager["_ControllerLock"]):
    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._stream: Any = None

    def __enter__(self) -> _ControllerLock:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._state_dir / "controller.lock"
        self._stream = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._stream.close()
            self._stream = None
            raise ControllerBusy("another continuation controller owns the lock") from exc
        atomic_write_json(
            self._state_dir / "lock-owner.json",
            {
                "schema_version": 1,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "acquired_at": _utc_now(),
            },
        )
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None


class _StateStore:
    """One atomic, authenticated snapshot contains both state and journal."""

    def __init__(self, state_dir: Path, *, controller_id: str) -> None:
        self.state_dir = state_dir
        self.controller_id = controller_id
        self.snapshots_dir = state_dir / "snapshots"
        self.current_path = state_dir / "CURRENT.json"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.sequence = -1
        self.state: dict[str, Any] = {}
        self.journal: list[dict[str, Any]] = []

    def load_or_initialize(self, initial_state: dict[str, Any]) -> None:
        if not self.current_path.is_file():
            if any(self.snapshots_dir.iterdir()):
                raise ControllerError(
                    "controller CURRENT pointer is missing while state snapshots exist"
                )
            self.state = dict(initial_state)
            self.commit("controller_initialized", {})
            return
        try:
            current = json.loads(self.current_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ControllerError("cannot load atomic controller state") from exc
        if not isinstance(current, dict) or current.get("schema_version") != 1:
            raise ControllerError("controller CURRENT pointer has an invalid schema")
        current_sequence = current.get("sequence")
        if (
            current.get("controller_id") != self.controller_id
            or isinstance(current_sequence, bool)
            or not isinstance(current_sequence, int)
            or current_sequence < 0
        ):
            raise ControllerError("controller CURRENT pointer is bound to invalid state")
        raw_snapshot = current.get("snapshot")
        if not isinstance(raw_snapshot, str):
            raise ControllerError("controller CURRENT snapshot path is invalid")
        relative_snapshot = Path(raw_snapshot)
        if (
            relative_snapshot.is_absolute()
            or relative_snapshot.parts[:1] != ("snapshots",)
            or len(relative_snapshot.parts) != 2
        ):
            raise ControllerError("controller CURRENT snapshot path escapes the state directory")
        name_match = _SNAPSHOT_NAME_RE.fullmatch(relative_snapshot.name)
        if name_match is None or int(name_match.group("sequence")) != current_sequence:
            raise ControllerError("controller CURRENT snapshot name is invalid")
        snapshot_path = self.state_dir / relative_snapshot
        expected_file_sha256 = current.get("sha256")
        if (
            not isinstance(expected_file_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256) is None
        ):
            raise ControllerError("controller CURRENT snapshot digest is invalid")
        try:
            if sha256_file(snapshot_path) != expected_file_sha256:
                raise ControllerError("controller state snapshot digest mismatch")
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ControllerError("cannot load atomic controller state") from exc
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("schema_version") != 1
            or snapshot.get("controller_id") != self.controller_id
            or snapshot.get("sequence") != current_sequence
        ):
            raise ControllerError("controller state is bound to a different config")
        if relative_snapshot.name != (f"{current_sequence:06d}-{_sha256_payload(snapshot)}.json"):
            raise ControllerError("controller state snapshot name digest mismatch")
        snapshot_state = snapshot.get("state")
        snapshot_journal = snapshot.get("journal")
        if (
            not isinstance(snapshot_state, dict)
            or not isinstance(snapshot_journal, list)
            or len(snapshot_journal) != current_sequence + 1
            or any(
                not isinstance(item, dict) or item.get("sequence") != sequence
                for sequence, item in enumerate(snapshot_journal)
            )
        ):
            raise ControllerError("controller state snapshot contents are invalid")
        inventory: dict[int, Path] = {}
        for candidate in self.snapshots_dir.iterdir():
            match = _SNAPSHOT_NAME_RE.fullmatch(candidate.name)
            if not candidate.is_file() or match is None:
                raise ControllerError("controller snapshot inventory contains an invalid entry")
            sequence = int(match.group("sequence"))
            if sequence in inventory:
                raise ControllerError("controller snapshot inventory contains a fork")
            inventory[sequence] = candidate
        if (
            set(inventory) != set(range(current_sequence + 1))
            or inventory[current_sequence] != snapshot_path
        ):
            raise ControllerError("controller CURRENT pointer does not name the latest history")
        self.sequence = current_sequence
        self.state = dict(snapshot_state)
        self.journal = list(snapshot_journal)

    def commit(self, event_type: str, details: Mapping[str, Any]) -> None:
        self.sequence += 1
        event = {
            "sequence": self.sequence,
            "event": event_type,
            "created_at": _utc_now(),
            "details": dict(details),
        }
        self.journal.append(event)
        payload = {
            "schema_version": 1,
            "controller_id": self.controller_id,
            "sequence": self.sequence,
            "state": self.state,
            "journal": self.journal,
        }
        snapshot_name = f"{self.sequence:06d}-{_sha256_payload(payload)}.json"
        snapshot_path = self.snapshots_dir / snapshot_name
        atomic_write_json(snapshot_path, payload)
        atomic_write_json(
            self.current_path,
            {
                "schema_version": 1,
                "controller_id": self.controller_id,
                "sequence": self.sequence,
                "snapshot": str(snapshot_path.relative_to(self.state_dir)),
                "sha256": sha256_file(snapshot_path),
            },
        )


class SlurmScheduler:
    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self.environment = dict(environment) if environment is not None else None

    def _run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
                env=self.environment,
            )
        except OSError as exc:
            raise ControllerError(f"cannot execute scheduler command {command[0]}") from exc

    def find(self, job_name: str) -> str | None:
        commands = (
            ("squeue", "--noheader", f"--name={job_name}", "--format=%A"),
            (
                "sacct",
                "--noheader",
                f"--name={job_name}",
                "--starttime",
                (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d"),
                "--format=JobIDRaw",
            ),
        )
        failed_lookups: list[str] = []
        for command in commands:
            completed = self._run(command)
            if completed.returncode != 0:
                failed_lookups.append(command[0])
                continue
            identifiers = {
                match.group(1)
                for line in completed.stdout.splitlines()
                if (match := re.match(r"\s*([1-9][0-9]*)", line))
            }
            if len(identifiers) > 1:
                raise ControllerError(f"multiple Slurm jobs use deterministic name {job_name}")
            if identifiers:
                return next(iter(identifiers))
        if failed_lookups:
            unavailable = ", ".join(failed_lookups)
            raise ControllerError(
                "cannot safely recover a deterministic Slurm job because "
                f"scheduler lookup failed: {unavailable}"
            )
        return None

    def submit(self, command: Sequence[str]) -> str:
        completed = self._run(command)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
            raise ControllerError(f"sbatch failed: {detail}")
        output = completed.stdout.strip()
        match = _JOB_ID_RE.fullmatch(output)
        if match is None:
            raise ControllerError(f"sbatch returned an invalid job id: {output!r}")
        return str(match.group("job_id"))


def _profile_sbatch_options(
    site: Mapping[str, Any],
    *,
    profile_id: str,
    walltime: str,
) -> list[str]:
    profile = site["profiles"][profile_id]
    directives = profile.get("directives")
    if not isinstance(directives, Mapping):
        raise ControllerError(f"site profile {profile_id} has no Slurm directives")
    options: list[str] = []
    for key in _SBATCH_DIRECTIVES:
        if key in directives:
            value = str(directives[key])
            if any(character in value for character in ("\n", "\r", "\0")):
                raise ControllerError(f"unsafe site directive {key}")
            options.append(f"--{key}={value}")
    options.append(f"--time={walltime}")
    return options


def _export_option(values: Mapping[str, str]) -> str:
    pairs: list[str] = []
    for key, value in values.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ControllerError(f"invalid exported environment key: {key}")
        if any(character in value for character in (",", "\n", "\r", "\0")):
            raise ControllerError(f"unsafe Slurm export value for {key}")
        pairs.append(f"{key}={value}")
    return "--export=ALL," + ",".join(pairs)


def _job_name(controller_id: str, segment_index: int, kind: str) -> str:
    return f"msc-{controller_id[:12]}-s{segment_index:03d}-{kind}"


def _submit_once(
    *,
    store: _StateStore,
    scheduler: SlurmScheduler,
    submission: dict[str, Any],
    field: str,
    job_name: str,
    command: list[str],
) -> str:
    record = submission.setdefault(field, {})
    recorded_id = record.get("job_id")
    if recorded_id is not None:
        return str(recorded_id)
    if record.get("job_name") != job_name:
        record.clear()
        record["job_name"] = job_name
        store.commit("submission_intent", {"field": field, "job_name": job_name})
    job_id = scheduler.find(job_name)
    recovered = job_id is not None
    if job_id is None:
        job_id = scheduler.submit(command)
    record["job_id"] = job_id
    record["recovered"] = recovered
    store.commit(
        "submission_recorded",
        {
            "field": field,
            "job_name": job_name,
            "job_id": job_id,
            "recovered": recovered,
        },
    )
    return job_id


def _initial_state(config: ControllerConfig, meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "controller_id": config.controller_id,
        "campaign_id": str(meta["campaign_id"]),
        "source_manifest_sha256": str(meta["manifest_sha256"]),
        "max_segments": config.max_segments,
        "status": "initialized",
        "last_observed_segment": 0,
        "reconcile_attempts": {},
        "segments": {},
        "bootstrap": {},
        "active_submission": None,
        "updated_at": _utc_now(),
    }


def _validate_compute_node() -> None:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_node = os.environ.get("SLURMD_NODENAME")
    if not slurm_job_id or not slurm_node:
        raise ControllerError("continuation controller must run inside a Slurm allocation")
    if socket.gethostname().split(".", maxsplit=1)[0] != slurm_node:
        raise ControllerError("continuation controller refuses a login node")


def _load_reconcile_artifacts(
    config: ControllerConfig,
    *,
    report_path: Path,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], Path | None]:
    reconciliations_root = (Path(config.state_dir) / "reconciliations").resolve()
    resolved_report = report_path.resolve()
    if (
        not resolved_report.is_relative_to(reconciliations_root)
        or resolved_report.name != "reconcile.json"
    ):
        raise ControllerError("reconcile report path escapes the controller state")
    if expected_sha256 is not None and (
        re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or sha256_file(resolved_report) != expected_sha256
    ):
        raise ControllerError("reconcile report digest mismatch")
    try:
        payload = json.loads(resolved_report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControllerError("cannot read authenticated reconcile report") from exc
    if not isinstance(payload, dict):
        raise ControllerError("authenticated reconcile report must be an object")
    raw_continuation = payload.get("continuation_campaign_path")
    if raw_continuation is None:
        return payload, None
    if not isinstance(raw_continuation, str):
        raise ControllerError("reconcile continuation path is invalid")
    if raw_continuation == "bundle://continuation-campaign":
        continuation = resolved_report.parent / "continuation-campaign"
    elif raw_continuation.startswith("bundle://"):
        raise ControllerError("reconcile continuation reference is invalid")
    else:
        # Compatibility with already-sealed controller state created before
        # reconciliation bundles switched to portable logical references.
        continuation = Path(raw_continuation).resolve()
    if continuation != resolved_report.parent / "continuation-campaign":
        raise ControllerError("reconcile continuation path escapes its output directory")
    return payload, continuation


def _reconcile_segment(
    *,
    config: ControllerConfig,
    store: _StateStore,
    segment_index: int,
) -> tuple[dict[str, Any], Path | None]:
    attempts = store.state["reconcile_attempts"]
    key = str(segment_index)
    attempt = int(attempts.get(key, 0))
    segment_record = store.state["segments"].get(key)
    if isinstance(segment_record, Mapping):
        report_path = Path(str(segment_record.get("report_path", "")))
        if not report_path.is_file():
            raise ControllerError("stored reconcile report is missing")
        expected_sha256 = segment_record.get("report_sha256")
        if not isinstance(expected_sha256, str):
            raise ControllerError("stored reconcile report digest is invalid")
        return _load_reconcile_artifacts(
            config,
            report_path=report_path,
            expected_sha256=expected_sha256,
        )

    attempt += 1
    attempts[key] = attempt
    output_dir = (
        Path(config.state_dir)
        / "reconciliations"
        / f"segment-{segment_index:03d}-attempt-{attempt:03d}"
    )
    store.commit(
        "reconcile_started",
        {"segment_index": segment_index, "attempt": attempt, "output_dir": str(output_dir)},
    )
    report = reconcile_campaign(
        Path(config.campaign_dir) / "manifest.jsonl",
        meta_path=Path(config.campaign_dir) / "manifest.meta.json",
        result_roots=[Path(config.result_root)],
        output_dir=output_dir,
        emit_retry=False,
    )
    report_path = Path(report.report_path)
    payload, continuation_path = _load_reconcile_artifacts(
        config,
        report_path=report_path,
        expected_sha256=None,
    )
    if (
        str(continuation_path) if continuation_path is not None else None
    ) != report.continuation_campaign_path:
        raise ControllerError("reconcile API and report continuation paths differ")
    store.state["segments"][key] = {
        "report_path": report.report_path,
        "report_sha256": sha256_file(report_path),
        "counts": report.counts,
        "status": report.status,
        "tasks": [
            {
                "task_id": str(item.get("task_id")),
                "status": str(item.get("status")),
                "continuation_attempt_count": int(item.get("continuation_attempt_count", 0)),
            }
            for item in payload.get("tasks", [])
        ],
        "continuation_campaign_path": (
            str(continuation_path) if continuation_path is not None else None
        ),
    }
    store.state["last_observed_segment"] = segment_index
    store.state["updated_at"] = _utc_now()
    store.commit(
        "reconcile_completed",
        {
            "segment_index": segment_index,
            "counts": report.counts,
            "status": report.status,
        },
    )
    return payload, continuation_path


def _validate_continuation_campaign(
    config: ControllerConfig,
    *,
    continuation_dir: Path,
    pending_task_ids: set[str],
    wrapper_paths: Sequence[Path] | None = None,
) -> tuple[Path, str, str]:
    _, source_tasks = _validate_campaign_binding(config)
    source_by_id = {task.task_id: task.to_dict() for task in source_tasks}
    meta, continuation_tasks = load_manifest(
        continuation_dir / "manifest.jsonl",
        meta_path=continuation_dir / "manifest.meta.json",
        verify_digest=True,
    )
    if {task.task_id for task in continuation_tasks} != pending_task_ids:
        raise ControllerError("continuation manifest does not equal the pending task set")
    for task in continuation_tasks:
        if task.to_dict() != source_by_id.get(task.task_id):
            raise ControllerError("continuation changed a task identity, seed, or manifest row")

    resources_path = continuation_dir / "profiles" / "resources.json"
    try:
        resources = json.loads(resources_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControllerError("cannot read continuation resource catalogue") from exc
    resource_rows = resources.get("resources")
    if (
        resources.get("schema_version") != 1
        or resources.get("manifest_sha256") != meta.get("manifest_sha256")
        or not isinstance(resource_rows, list)
        or not resource_rows
        or any(not isinstance(row, dict) for row in resource_rows)
    ):
        raise ControllerError("continuation resource catalogue is empty")
    expected_profiles = {task.resource_profile for task in continuation_tasks}
    if {str(row.get("profile_id")) for row in resource_rows} != expected_profiles:
        raise ControllerError("continuation resource catalogue changed the task profiles")
    for row in resource_rows:
        profile_id = str(row["profile_id"])
        contract = _MATCH_CONTINUATION_PROFILES[profile_id]
        expected_concurrency = int(contract["concurrency"])
        walltime_seconds = int(contract["walltime_seconds"])
        if (
            str(row.get("architecture", "")).upper() != "H100"
            or row.get("accelerators_per_task") != 1
            or row.get("fixed_walltime") is not True
            or row.get("configured_walltime_seconds") != walltime_seconds
            or row.get("max_walltime_seconds") != walltime_seconds
            or row.get("initial_concurrency") != expected_concurrency
        ):
            raise ControllerError(
                "continuation resource catalogue is not a registered fixed mono-H100 "
                "Match allocation"
            )
    index_rows = resources.get("array_indices")
    if (
        not isinstance(index_rows, list)
        or len(index_rows) != 1
        or not isinstance(index_rows[0], dict)
    ):
        raise ControllerError("continuation must contain exactly one array index record")
    index_record = index_rows[0]
    index_relpath = index_record.get("path")
    if not isinstance(index_relpath, str):
        raise ControllerError("continuation array index path is invalid")
    relative_index = Path(index_relpath)
    if (
        relative_index.is_absolute()
        or relative_index.parts[:1] != ("profiles",)
        or len(relative_index.parts) != 2
    ):
        raise ControllerError("continuation array index escapes its campaign")
    index_path = continuation_dir / relative_index
    expected_index_text = "".join(f"{task.task_index}\n" for task in continuation_tasks)
    try:
        actual_index_text = index_path.read_text(encoding="utf-8")
        index_sha256 = sha256_file(index_path)
    except (OSError, UnicodeError) as exc:
        raise ControllerError("cannot read continuation array index") from exc
    wrappers = (
        sorted(path.resolve() for path in wrapper_paths)
        if wrapper_paths is not None
        else sorted((continuation_dir / "submit").glob("*/*.slurm"))
    )
    if len(wrappers) != 1:
        raise ControllerError("Match continuation requires exactly one generated H100 wrapper")
    wrapper = wrappers[0].resolve()
    if len(expected_profiles) != 1:
        raise ControllerError("Match continuation requires exactly one long resource profile")
    profile_id = next(iter(expected_profiles))
    site = _load_site(config)
    site_id = str(site["site_id"])
    if (
        index_record.get("site_id") != site_id
        or index_record.get("profile_id") != profile_id
        or index_record.get("task_count") != len(continuation_tasks)
        or index_record.get("sha256") != index_sha256
        or actual_index_text != expected_index_text
    ):
        raise ControllerError("continuation array index changed its immutable task mapping")
    if wrapper.parent.name != site_id or wrapper.name != f"{profile_id}.slurm":
        raise ControllerError("continuation wrapper path changed its resource profile")
    architecture = str(resource_rows[0]["architecture"])
    expected_wrapper = render_slurm_wrapper(
        campaign_id=str(meta["campaign_id"]),
        site=site,
        profile_id=profile_id,
        profile=site["profiles"][profile_id],
        task_count=len(continuation_tasks),
        index_filename=relative_index.name,
        index_sha256=index_sha256,
        manifest_sha256=str(meta["manifest_sha256"]),
        resource_profile_id=profile_id,
        architecture=architecture,
    )
    try:
        wrapper_text = wrapper.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ControllerError("cannot read continuation wrapper") from exc
    if wrapper_text != expected_wrapper:
        raise ControllerError("continuation wrapper changed its fixed Match Slurm contract")
    manifest_digest = str(meta["manifest_sha256"])
    return wrapper, manifest_digest, str(continuation_tasks[0].environment_lock_sha256)


def _preflight_command(
    config: ControllerConfig,
    *,
    next_segment: int,
    continuation_dir: Path,
    preflight_path: Path,
    job_name: str,
) -> list[str]:
    site = _load_site(config)
    log_path = continuation_dir / "logs" / f"preflight-H100-s{next_segment:03d}-%j.out"
    return [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        *_profile_sbatch_options(
            site,
            profile_id=config.controller_profile,
            walltime="00:30:00",
        ),
        f"--output={log_path}",
        _export_option({"MODSSC_ROOT": config.repo_root}),
        str(Path(config.repo_root) / "tools/hpc/slurm/run-operation.sh"),
        "preflight",
        "--manifest",
        str(continuation_dir / "manifest.jsonl"),
        "--allocation",
        config.allocation_path,
        "--site",
        config.site_path,
        "--repo-root",
        config.repo_root,
        "--output",
        str(preflight_path),
        "--environment-manifest",
        config.environment_manifest_path,
        "--require-architecture",
        "H100",
    ]


def _array_command(
    config: ControllerConfig,
    *,
    continuation_dir: Path,
    wrapper: Path,
    preflight_path: Path,
    preflight_job_id: str,
    environment_lock_sha256: str,
    job_name: str,
) -> list[str]:
    return [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--dependency=afterok:{preflight_job_id}",
        f"--chdir={continuation_dir}",
        _export_option(
            {
                "MODSSC_ROOT": config.repo_root,
                "MODSSC_CAMPAIGN_DIR": str(continuation_dir),
                "MODSSC_CAMPAIGN_RESULTS": str(Path(config.result_root).parent),
                "MODSSC_CAMPAIGN_CHECKPOINTS": config.checkpoint_base,
                "MODSSC_PREFLIGHT_REPORT": str(preflight_path),
                "MODSSC_PREFLIGHT_EXPIRY_POLICY": "generated_by_dependency",
                "MODSSC_PREFLIGHT_JOB_ID": preflight_job_id,
                "MODSSC_ENVIRONMENT_MANIFEST": config.environment_manifest_path,
                "MODSSC_ENVIRONMENT_LOCK_SHA256": environment_lock_sha256,
            }
        ),
        str(wrapper),
    ]


def _controller_command(
    config: ControllerConfig,
    *,
    segment_index: int,
    dependency_job_id: str,
    job_name: str,
) -> list[str]:
    site = _load_site(config)
    logs_dir = Path(config.state_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--dependency=afterany:{dependency_job_id}",
        *_profile_sbatch_options(
            site,
            profile_id=config.controller_profile,
            walltime="00:20:00",
        ),
        f"--output={logs_dir}/controller-s{segment_index:03d}-%j.out",
        _export_option(
            {
                "MODSSC_ROOT": config.repo_root,
                "MODSSC_MATCH_CONTROLLER_CONFIG": str(
                    Path(config.state_dir) / "controller-config.json"
                ),
                "MODSSC_MATCH_CONTROLLER_SEGMENT": str(segment_index),
            }
        ),
        str(Path(config.repo_root) / "tools/hpc/slurm/run-operation.sh"),
        "continuation",
        "--config",
        str(Path(config.state_dir) / "controller-config.json"),
        "--segment-index",
        str(segment_index),
    ]


def _advance_controller(
    *,
    config: ControllerConfig,
    store: _StateStore,
    scheduler: SlurmScheduler,
    segment_index: int,
) -> dict[str, Any]:
    last_segment = int(store.state["last_observed_segment"])
    if segment_index < last_segment:
        return {"status": "already_observed", "segment_index": segment_index}
    if segment_index > last_segment + 1:
        raise ControllerError("controller segment sequence contains a gap")
    if segment_index == last_segment and store.state["status"] in _TERMINAL_STATUSES:
        return {
            "status": str(store.state["status"]),
            "segment_index": segment_index,
        }

    report, continuation_dir = _reconcile_segment(
        config=config,
        store=store,
        segment_index=segment_index,
    )
    task_states = report.get("tasks")
    if not isinstance(task_states, list):
        raise ControllerError("reconcile report has no task states")
    statuses = {str(item.get("status")) for item in task_states}
    if statuses == {"success"}:
        store.state["status"] = "complete"
        store.state["active_submission"] = None
        store.state["updated_at"] = _utc_now()
        store.commit("campaign_complete", {"segment_index": segment_index})
        return {"status": "complete", "segment_index": segment_index}
    if not statuses.issubset({"success", "continuation_pending"}):
        store.state["status"] = "blocked"
        store.state["active_submission"] = None
        store.state["updated_at"] = _utc_now()
        store.commit(
            "campaign_blocked",
            {"segment_index": segment_index, "statuses": sorted(statuses)},
        )
        return {
            "status": "blocked",
            "segment_index": segment_index,
            "statuses": sorted(statuses),
        }
    if segment_index > 1:
        previous_record = store.state["segments"].get(str(segment_index - 1), {})
        previous_tasks = {
            str(item.get("task_id")): int(item.get("continuation_attempt_count", 0))
            for item in previous_record.get("tasks", [])
        }
        stalled = sorted(
            str(item.get("task_id"))
            for item in task_states
            if item.get("status") == "continuation_pending"
            and int(item.get("continuation_attempt_count", 0))
            <= previous_tasks.get(str(item.get("task_id")), -1)
        )
        if stalled:
            store.state["status"] = "blocked"
            store.state["active_submission"] = None
            store.state["updated_at"] = _utc_now()
            store.commit(
                "segment_made_no_progress",
                {"segment_index": segment_index, "task_ids": stalled},
            )
            return {
                "status": "blocked",
                "segment_index": segment_index,
                "reason": "segment_made_no_progress",
                "task_ids": stalled,
            }
    if segment_index >= config.max_segments:
        store.state["status"] = "max_segments_exceeded"
        store.state["active_submission"] = None
        store.state["updated_at"] = _utc_now()
        store.commit(
            "max_segments_exceeded",
            {"segment_index": segment_index, "max_segments": config.max_segments},
        )
        return {
            "status": "max_segments_exceeded",
            "segment_index": segment_index,
        }
    if continuation_dir is None:
        raise ControllerError("pending tasks have no rendered continuation campaign")

    pending_ids = {
        str(item["task_id"]) for item in task_states if item.get("status") == "continuation_pending"
    }
    wrapper_paths = render_slurm_sites(
        site_paths=[Path(config.site_path)],
        campaign_dir=continuation_dir,
        submission_dir=(
            Path(config.state_dir) / "submissions" / f"segment-{segment_index + 1:03d}"
        ),
    )
    wrapper, manifest_digest, environment_digest = _validate_continuation_campaign(
        config,
        continuation_dir=continuation_dir,
        pending_task_ids=pending_ids,
        wrapper_paths=wrapper_paths,
    )
    next_segment = segment_index + 1
    active = store.state.get("active_submission")
    if not isinstance(active, dict) or active.get("segment_index") != next_segment:
        active = {
            "segment_index": next_segment,
            "continuation_campaign_path": str(continuation_dir),
            "continuation_manifest_sha256": manifest_digest,
            "pending_task_ids": sorted(pending_ids),
        }
        store.state["active_submission"] = active
        store.state["status"] = "submitting"
        store.state["updated_at"] = _utc_now()
        store.commit(
            "continuation_prepared",
            {
                "segment_index": next_segment,
                "pending_task_ids": sorted(pending_ids),
                "manifest_sha256": manifest_digest,
            },
        )

    preflight_path = continuation_dir / f"preflight-H100-s{next_segment:03d}.json"
    preflight_name = _job_name(config.controller_id, next_segment, "preflight")
    preflight_id = _submit_once(
        store=store,
        scheduler=scheduler,
        submission=active,
        field="preflight",
        job_name=preflight_name,
        command=_preflight_command(
            config,
            next_segment=next_segment,
            continuation_dir=continuation_dir,
            preflight_path=preflight_path,
            job_name=preflight_name,
        ),
    )
    array_name = _job_name(config.controller_id, next_segment, "array")
    array_id = _submit_once(
        store=store,
        scheduler=scheduler,
        submission=active,
        field="array",
        job_name=array_name,
        command=_array_command(
            config,
            continuation_dir=continuation_dir,
            wrapper=wrapper,
            preflight_path=preflight_path,
            preflight_job_id=preflight_id,
            environment_lock_sha256=environment_digest,
            job_name=array_name,
        ),
    )
    controller_name = _job_name(config.controller_id, next_segment, "controller")
    controller_id = _submit_once(
        store=store,
        scheduler=scheduler,
        submission=active,
        field="controller",
        job_name=controller_name,
        command=_controller_command(
            config,
            segment_index=next_segment,
            dependency_job_id=array_id,
            job_name=controller_name,
        ),
    )
    store.state["status"] = "scheduled"
    store.state["updated_at"] = _utc_now()
    store.commit(
        "continuation_scheduled",
        {
            "segment_index": next_segment,
            "preflight_job_id": preflight_id,
            "array_job_id": array_id,
            "controller_job_id": controller_id,
        },
    )
    return {
        "status": "scheduled",
        "segment_index": next_segment,
        "preflight_job_id": preflight_id,
        "array_job_id": array_id,
        "controller_job_id": controller_id,
    }


def run_controller(
    config_path: Path,
    *,
    segment_index: int,
    scheduler: SlurmScheduler | None = None,
    require_slurm: bool = True,
) -> dict[str, Any]:
    if isinstance(segment_index, bool) or segment_index <= 0:
        raise ControllerError("segment_index must be a positive integer")
    if require_slurm:
        _validate_compute_node()
    config = ControllerConfig.load(config_path)
    meta, _ = _validate_campaign_binding(config)
    state_dir = Path(config.state_dir)
    with _ControllerLock(state_dir):
        store = _StateStore(state_dir, controller_id=config.controller_id)
        store.load_or_initialize(_initial_state(config, meta))
        return _advance_controller(
            config=config,
            store=store,
            scheduler=scheduler or SlurmScheduler(),
            segment_index=segment_index,
        )


def bootstrap_controller(
    *,
    config: ControllerConfig,
    after_job_id: str,
    scheduler: SlurmScheduler | None = None,
) -> dict[str, Any]:
    if _SAFE_JOB_ID_RE.fullmatch(after_job_id) is None:
        raise ControllerError("after_job_id must be a positive Slurm job id")
    state_dir = Path(config.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    config_path = state_dir / "controller-config.json"
    if config_path.is_file():
        existing = ControllerConfig.load(config_path)
        if existing != config:
            raise ControllerError("state directory already contains a different config")
    else:
        atomic_write_json(config_path, config.to_dict())

    meta, _ = _validate_campaign_binding(config)
    with _ControllerLock(state_dir):
        store = _StateStore(state_dir, controller_id=config.controller_id)
        store.load_or_initialize(_initial_state(config, meta))
        submission = store.state["bootstrap"]
        previous_dependency = submission.get("after_job_id")
        if previous_dependency is not None and previous_dependency != after_job_id:
            raise ControllerError("bootstrap is already bound to a different initial job")
        if previous_dependency is None:
            submission["after_job_id"] = after_job_id
            store.commit("bootstrap_intent", {"after_job_id": after_job_id})
        job_name = _job_name(config.controller_id, 1, "controller")
        job_id = _submit_once(
            store=store,
            scheduler=scheduler or SlurmScheduler(),
            submission=submission,
            field="controller",
            job_name=job_name,
            command=_controller_command(
                config,
                segment_index=1,
                dependency_job_id=after_job_id,
                job_name=job_name,
            ),
        )
        store.state["status"] = "waiting_for_segment"
        store.state["updated_at"] = _utc_now()
        store.commit(
            "bootstrap_scheduled",
            {"after_job_id": after_job_id, "controller_job_id": job_id},
        )
    return {
        "status": "scheduled",
        "segment_index": 1,
        "controller_job_id": job_id,
        "config_path": str(config_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Idempotent configured Slurm site Match continuation controller"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="reconcile one completed segment and advance it")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--segment-index", type=int, required=True)

    bootstrap = commands.add_parser(
        "bootstrap",
        help="write the immutable controller config and schedule the first controller job",
    )
    bootstrap.add_argument("--repo-root", type=Path, required=True)
    bootstrap.add_argument("--campaign-dir", type=Path, required=True)
    bootstrap.add_argument("--result-root", type=Path, required=True)
    bootstrap.add_argument("--state-dir", type=Path, required=True)
    bootstrap.add_argument("--site", type=Path, required=True)
    bootstrap.add_argument("--allocation", type=Path, required=True)
    bootstrap.add_argument("--environment-manifest", type=Path, required=True)
    bootstrap.add_argument("--checkpoint-base", type=Path, required=True)
    bootstrap.add_argument("--max-segments", type=int, required=True)
    bootstrap.add_argument("--after-job-id", required=True)
    bootstrap.add_argument("--controller-profile", default="h100_dev")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run_controller(
                args.config,
                segment_index=args.segment_index,
            )
        else:
            config = ControllerConfig.build(
                repo_root=args.repo_root,
                campaign_dir=args.campaign_dir,
                result_root=args.result_root,
                state_dir=args.state_dir,
                site_path=args.site,
                allocation_path=args.allocation,
                environment_manifest_path=args.environment_manifest,
                checkpoint_base=args.checkpoint_base,
                max_segments=args.max_segments,
                controller_profile=args.controller_profile,
            )
            result = bootstrap_controller(
                config=config,
                after_job_id=args.after_job_id,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] not in {"blocked", "max_segments_exceeded"} else 2
    except ControllerBusy as exc:
        print(json.dumps({"status": "already_running", "detail": str(exc)}, sort_keys=True))
        return 0
    except (ControllerError, CampaignError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
