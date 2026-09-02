"""Native reconciliation of independently executed seed evaluations.

This module has no knowledge of schedulers, YAML cards, or research articles.
It validates that observed run reports form a partial function over an explicit
seed set, keeps every terminal category distinct, and aggregates metrics from
successful runs only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, cast

from modssc.runtime.execution import RunIdentity
from modssc.runtime.protocol import effective_config_sha256, protocol_sha256
from modssc.runtime.software import software_sha256

from .aggregation import aggregate_metric_records, summarize_numeric

SeedRunStatus = Literal["success", "failed", "not_evaluable"]
SeedReconciliationStatus = Literal["success", "partial_failure", "not_evaluable", "failed"]

_RUN_STATUSES = frozenset({"success", "failed", "not_evaluable"})
_HASH_FIELDS = (
    "config_hash",
    "effective_config_hash",
    "protocol_sha256",
    "software_sha256",
)


class SeedReconciliationError(ValueError):
    """Raised when seed reports cannot be reconciled without ambiguity."""


@dataclass(frozen=True)
class SeedReconciliation:
    """Validated partition and aggregate for one requested seed set."""

    requested_seeds: tuple[int, ...]
    success_seeds: tuple[int, ...]
    failed_seeds: tuple[int, ...]
    not_evaluable_seeds: tuple[int, ...]
    missing_seeds: tuple[int, ...]
    metrics: Mapping[str, Any]
    run_info: Mapping[str, Any]
    runs: tuple[Mapping[str, Any], ...]
    execution_identity_complete: bool

    @property
    def status(self) -> SeedReconciliationStatus:
        """Return the terminal status of the complete requested seed set."""

        if not (self.failed_seeds or self.not_evaluable_seeds or self.missing_seeds):
            return "success"
        if self.success_seeds:
            return "partial_failure"
        if self.not_evaluable_seeds and not (self.failed_seeds or self.missing_seeds):
            return "not_evaluable"
        return "failed"

    @property
    def certifiable(self) -> bool:
        """Whether every requested seed completed successfully."""

        return self.status == "success" and self.execution_identity_complete

    def categories(self) -> dict[str, list[int]]:
        """Return the four disjoint seed categories as JSON-compatible lists."""

        return {
            "success": list(self.success_seeds),
            "failed": list(self.failed_seeds),
            "not_evaluable": list(self.not_evaluable_seeds),
            "missing": list(self.missing_seeds),
        }

    def summary(self) -> dict[str, Any]:
        """Return stable counts and categories for an aggregate report."""

        completed = len(self.success_seeds) + len(self.failed_seeds) + len(self.not_evaluable_seeds)
        return {
            "requested_seeds": list(self.requested_seeds),
            "requested_run_count": len(self.requested_seeds),
            "completed_run_count": completed,
            "successful_run_count": len(self.success_seeds),
            "failed_run_count": len(self.failed_seeds),
            "not_evaluable_run_count": len(self.not_evaluable_seeds),
            "missing_run_count": len(self.missing_seeds),
            "success_seeds": list(self.success_seeds),
            "failed_seeds": list(self.failed_seeds),
            "not_evaluable_seeds": list(self.not_evaluable_seeds),
            "missing_seeds": list(self.missing_seeds),
            "categories": self.categories(),
            "execution_identity_complete": self.execution_identity_complete,
            "status": self.status,
            "certifiable": self.certifiable,
        }


def _seed(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SeedReconciliationError(f"{field} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise SeedReconciliationError(f"{field} must be a non-negative integer")
    return normalized


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SeedReconciliationError(f"{field} must be a mapping")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise SeedReconciliationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _expected_seed_hashes(
    value: Mapping[int, str] | None,
    *,
    requested: tuple[int, ...],
    field: str,
) -> dict[int, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SeedReconciliationError(f"{field} must be a mapping")
    normalized: dict[int, str] = {}
    for raw_seed, raw_hash in value.items():
        seed = _seed(raw_seed, field=f"{field} seed")
        normalized[seed] = _sha256(raw_hash, field=f"{field}[{seed}]")
    if set(normalized) != set(requested):
        raise SeedReconciliationError(f"{field} keys must exactly match requested_seeds")
    return normalized


def _run_entry(
    report: Mapping[str, Any],
    *,
    source_path: str | Path | None,
) -> tuple[int, SeedRunStatus, dict[str, Any]]:
    run = _mapping(report.get("run"), field="report.run")
    seed = _seed(run.get("seed"), field="report.run.seed")
    status_value = run.get("status")
    if status_value not in _RUN_STATUSES:
        raise SeedReconciliationError("report.run.status must be success, failed, or not_evaluable")
    status = cast(SeedRunStatus, status_value)

    metrics_value = report.get("metrics")
    if metrics_value is not None and not isinstance(metrics_value, Mapping):
        raise SeedReconciliationError("report.metrics must be a mapping or null")
    if status == "success" and metrics_value is None:
        raise SeedReconciliationError("a successful report must contain metrics")
    metrics = None if metrics_value is None else dict(metrics_value)

    source = None if source_path is None else Path(source_path)
    entry = {
        "seed": seed,
        "name": run.get("name"),
        "run_id": run.get("run_id"),
        "status": status,
        "run_dir": None if source is None else str(source.parent),
        "run_json": None if source is None else str(source),
        "error_code": run.get("error_code"),
        "hashes": report.get("hashes"),
        "error": report.get("error"),
        "run_info": report.get("run_info"),
        "task_info": report.get("task_info"),
        "graph_info": report.get("graph_info"),
        "metrics": metrics,
    }
    return seed, status, entry


def _aggregate_run_info(runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    runtime_values: list[float] = []
    gpu_devices: set[str] = set()
    hardware_mismatch_count = 0
    for run in runs:
        run_info = run.get("run_info")
        if not isinstance(run_info, Mapping):
            continue
        runtime = run_info.get("run_time_seconds")
        if isinstance(runtime, Real) and not isinstance(runtime, bool):
            runtime_values.append(float(runtime))
        gpu = run_info.get("gpu_device")
        if gpu is not None and str(gpu):
            gpu_devices.add(str(gpu))
        if run_info.get("hardware_mismatch") is True:
            hardware_mismatch_count += 1

    ordered_gpus = sorted(gpu_devices)
    summary: dict[str, Any] = {
        "gpu_devices": ordered_gpus,
        "gpu_device": (
            ordered_gpus[0] if len(ordered_gpus) == 1 else ("Mixed" if ordered_gpus else None)
        ),
        "hardware_mismatch": hardware_mismatch_count > 0,
        "hardware_mismatch_count": hardware_mismatch_count,
    }
    if runtime_values:
        summary["run_time_seconds"] = summarize_numeric(runtime_values)
    return summary


def reconcile_seed_reports(
    *,
    requested_seeds: Iterable[int],
    reports: Iterable[Mapping[str, Any]],
    source_paths: Iterable[str | Path | None] | None = None,
    expected_config_hashes: Mapping[int, str] | None = None,
    expected_protocol_hashes: Mapping[int, str] | None = None,
    require_execution_identity: bool = True,
) -> SeedReconciliation:
    """Validate and aggregate reports for an explicit requested seed set.

    Duplicate requested or observed seeds and reports for seeds outside the
    requested set are invalid. Every report must carry complete configuration,
    protocol, and software hashes. Software identity is cohort-wide; protocol
    identity may differ only when validated against ``expected_protocol_hashes``.
    Portable execution identity is required by default; callers reading reports
    predating it must opt in explicitly with ``require_execution_identity=False``.
    Consequently ``success``, ``failed``, ``not_evaluable``, and ``missing`` are
    an exact disjoint partition.
    """

    requested = tuple(_seed(value, field="requested seed") for value in requested_seeds)
    if not requested:
        raise SeedReconciliationError("requested_seeds must not be empty")
    if len(set(requested)) != len(requested):
        raise SeedReconciliationError("requested_seeds contains duplicates")

    expected_hashes = _expected_seed_hashes(
        expected_config_hashes,
        requested=requested,
        field="expected_config_hashes",
    )
    expected_protocols = _expected_seed_hashes(
        expected_protocol_hashes,
        requested=requested,
        field="expected_protocol_hashes",
    )

    report_values = tuple(reports)
    sources = (None,) * len(report_values) if source_paths is None else tuple(source_paths)
    if len(sources) != len(report_values):
        raise SeedReconciliationError("source_paths must contain one value per report")

    requested_set = set(requested)
    observed: dict[int, tuple[SeedRunStatus, dict[str, Any]]] = {}
    protocol_hashes: set[str] = set()
    software_hashes: set[str] = set()
    legacy_identity_observed = False
    for report_index, (report, source) in enumerate(zip(report_values, sources, strict=True)):
        payload = _mapping(report, field=f"reports[{report_index}]")
        seed, status, entry = _run_entry(payload, source_path=source)
        if seed not in requested_set:
            raise SeedReconciliationError(f"unexpected observed seed: {seed}")
        if seed in observed:
            raise SeedReconciliationError(f"duplicate observed seed: {seed}")
        hashes = _mapping(
            payload.get("hashes"),
            field=f"reports[{report_index}].hashes",
        )
        identity_hashes = {
            field: _sha256(
                hashes.get(field),
                field=f"reports[{report_index}].hashes.{field}",
            )
            for field in _HASH_FIELDS
        }
        execution_payload = payload.get("execution_identity")
        execution_digest = hashes.get("execution_identity_sha256")
        if execution_payload is None and execution_digest is None:
            if require_execution_identity:
                raise SeedReconciliationError(
                    f"report for seed {seed} is missing required execution identity"
                )
            legacy_identity_observed = True
        else:
            if not isinstance(execution_payload, Mapping) or execution_digest is None:
                raise SeedReconciliationError(
                    f"report for seed {seed} has an incomplete execution identity"
                )
            try:
                execution_identity = RunIdentity.from_dict(execution_payload)
            except (TypeError, ValueError) as exc:
                raise SeedReconciliationError(
                    f"report execution identity is invalid for seed {seed}: {exc}"
                ) from exc
            observed_execution_digest = _sha256(
                execution_digest,
                field=f"reports[{report_index}].hashes.execution_identity_sha256",
            )
            if observed_execution_digest != execution_identity.sha256:
                raise SeedReconciliationError(
                    f"execution identity hash does not match for seed {seed}"
                )
            if execution_identity.seed != seed:
                raise SeedReconciliationError(
                    f"execution identity seed does not match for seed {seed}"
                )
            if execution_identity.config_sha256 != identity_hashes["protocol_sha256"]:
                raise SeedReconciliationError(
                    f"execution identity protocol does not match for seed {seed}"
                )
            if execution_identity.code_sha256 != identity_hashes["software_sha256"]:
                raise SeedReconciliationError(
                    f"execution identity software does not match for seed {seed}"
                )
            if entry["run_id"] != execution_identity.short_id:
                raise SeedReconciliationError(
                    f"run_id does not match portable execution identity for seed {seed}"
                )
            entry["execution_identity"] = execution_identity.to_dict()
            entry["execution_identity_sha256"] = execution_identity.sha256
        report_config = _mapping(
            payload.get("config"),
            field=f"reports[{report_index}].config",
        )
        report_versions = _mapping(
            payload.get("versions"),
            field=f"reports[{report_index}].versions",
        )
        try:
            recomputed_effective_config = effective_config_sha256(report_config)
            recomputed_protocol = protocol_sha256(report_config)
        except (TypeError, ValueError) as exc:
            raise SeedReconciliationError(
                f"cannot hash report config for seed {seed}: {exc}"
            ) from exc
        if identity_hashes["effective_config_hash"] != recomputed_effective_config:
            raise SeedReconciliationError(
                f"effective config hash does not match report config for seed {seed}"
            )
        if identity_hashes["protocol_sha256"] != recomputed_protocol:
            raise SeedReconciliationError(
                f"protocol hash does not match report config for seed {seed}"
            )
        try:
            recomputed_software = software_sha256(report_versions)
        except (TypeError, ValueError) as exc:
            raise SeedReconciliationError(
                f"cannot hash report versions for seed {seed}: {exc}"
            ) from exc
        if identity_hashes["software_sha256"] != recomputed_software:
            raise SeedReconciliationError(
                f"software hash does not match report versions for seed {seed}"
            )
        if expected_hashes is not None and identity_hashes["config_hash"] != expected_hashes[seed]:
            raise SeedReconciliationError(f"config hash mismatch for seed {seed}")
        if (
            expected_protocols is not None
            and identity_hashes["protocol_sha256"] != expected_protocols[seed]
        ):
            raise SeedReconciliationError(f"protocol hash mismatch for seed {seed}")
        protocol_hashes.add(identity_hashes["protocol_sha256"])
        software_hashes.add(identity_hashes["software_sha256"])
        observed[seed] = (status, entry)

    if len(software_hashes) > 1:
        raise SeedReconciliationError("software hash differs between seed reports")
    if expected_protocols is None and expected_hashes is None and len(protocol_hashes) > 1:
        raise SeedReconciliationError(
            "protocol hash differs between seed reports without expected identities"
        )

    def seeds_with(status: SeedRunStatus) -> tuple[int, ...]:
        return tuple(seed for seed in requested if seed in observed and observed[seed][0] == status)

    success_seeds = seeds_with("success")
    failed_seeds = seeds_with("failed")
    not_evaluable_seeds = seeds_with("not_evaluable")
    missing_seeds = tuple(seed for seed in requested if seed not in observed)
    ordered_runs = tuple(observed[seed][1] for seed in requested if seed in observed)

    try:
        metrics = aggregate_metric_records(observed[seed][1]["metrics"] for seed in success_seeds)
    except ValueError as exc:
        raise SeedReconciliationError(f"successful metric records are incompatible: {exc}") from exc

    return SeedReconciliation(
        requested_seeds=requested,
        success_seeds=success_seeds,
        failed_seeds=failed_seeds,
        not_evaluable_seeds=not_evaluable_seeds,
        missing_seeds=missing_seeds,
        metrics=metrics,
        run_info=_aggregate_run_info(ordered_runs),
        runs=ordered_runs,
        execution_identity_complete=not missing_seeds and not legacy_identity_observed,
    )


__all__ = [
    "SeedReconciliation",
    "SeedReconciliationError",
    "SeedReconciliationStatus",
    "SeedRunStatus",
    "reconcile_seed_reports",
]
