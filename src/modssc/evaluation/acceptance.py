"""Generic, declarative acceptance of reconciled repeated evaluations.

The evaluator deliberately has no knowledge of benchmark cards, files, papers,
or particular methods.  Its inputs are an in-memory acceptance specification
and already authenticated/reconciled ``run.json`` payloads.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from numbers import Integral, Real
from typing import Any, Literal, cast

from .aggregation import summarize_numeric

AssessmentStatus = Literal["passed", "failed", "not_evaluable"]
FidelityStatus = Literal["paper_matched", "paper_approx", "not_claimable"]
ConformityStatus = Literal["passed", "pending", "failed", "not_assessed"]
TargetTransform = Literal["identity", "one_minus"]
DiagnosticOperator = Literal[
    "eq",
    "gt",
    "ge",
    "lt",
    "le",
    "between",
    "present",
    "truthy",
    "nonempty",
]

_FIDELITY_STATUSES = frozenset({"paper_matched", "paper_approx", "not_claimable"})
_CONFORMITY_STATUSES = frozenset({"passed", "pending", "failed", "not_assessed"})
_TRANSFORMS = frozenset({"identity", "one_minus"})
_DIAGNOSTIC_OPERATORS = frozenset(
    {"eq", "gt", "ge", "lt", "le", "between", "present", "truthy", "nonempty"}
)
_VALUELESS_DIAGNOSTIC_OPERATORS = frozenset({"present", "truthy", "nonempty"})
_RUN_STATUSES = frozenset({"success", "failed", "not_evaluable"})
_ACCEPTANCE_KEYS = frozenset(
    {
        "protocol_id",
        "method_id",
        "repetitions",
        "fidelity_ceiling",
        "conformity",
        "target",
        "secondary_targets",
        "informational_targets",
        "diagnostic_targets",
        "required_diagnostics",
        "deviations",
        "equivalences",
        "unknowns",
    }
)
_TARGET_KEYS = frozenset(
    {
        "id",
        "path",
        "split",
        "metric",
        "transform",
        "published_mean",
        "published_std",
        "published_std_ddof",
        "margin_absolute",
        "offset",
    }
)


class AcceptanceSpecError(ValueError):
    """Raised when an acceptance specification or run cohort is ambiguous."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptanceSpecError(f"{field} must be a mapping")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    field: str,
) -> None:
    unknown = sorted(repr(key) for key in value if key not in allowed)
    if unknown:
        raise AcceptanceSpecError(f"{field} contains unknown keys: {unknown}")


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcceptanceSpecError(f"{field} must be a non-empty string")
    return value


def _string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise AcceptanceSpecError(f"{field} must be a sequence of non-empty strings")
    normalized = tuple(_nonempty_string(item, field=field) for item in value)
    if len(normalized) != len(set(normalized)):
        raise AcceptanceSpecError(f"{field} must not contain duplicates")
    return tuple(sorted(normalized))


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AcceptanceSpecError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise AcceptanceSpecError(f"{field} must be a finite number")
    return normalized


def _optional_finite(value: Any, *, field: str) -> float | None:
    return None if value is None else _finite(value, field=field)


def _json_canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AcceptanceSpecError("acceptance data must be strict JSON data") from exc


@dataclass(frozen=True)
class ConformityReview:
    """Identity and timestamp of a scientific conformity review."""

    reviewed_by: str
    reviewed_at: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, field: str) -> ConformityReview:
        _reject_unknown_keys(
            value,
            allowed=frozenset({"reviewed_by", "reviewed_at"}),
            field=field,
        )
        reviewed_by = _nonempty_string(value.get("reviewed_by"), field=f"{field}.reviewed_by")
        reviewed_at = _nonempty_string(value.get("reviewed_at"), field=f"{field}.reviewed_at")
        try:
            timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AcceptanceSpecError(f"{field}.reviewed_at must be ISO-8601") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise AcceptanceSpecError(f"{field}.reviewed_at must include a timezone")
        return cls(reviewed_by=reviewed_by, reviewed_at=reviewed_at)

    def to_dict(self) -> dict[str, str]:
        return {"reviewed_by": self.reviewed_by, "reviewed_at": self.reviewed_at}


@dataclass(frozen=True)
class ConformitySpec:
    """Auditable conformity gate attached to an acceptance protocol."""

    status: ConformityStatus
    basis: str | None = None
    evidence: tuple[str, ...] = ()
    review: ConformityReview | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, field: str = "conformity") -> ConformitySpec:
        _reject_unknown_keys(
            value,
            allowed=frozenset({"status", "basis", "evidence", "review"}),
            field=field,
        )
        status_value = value.get("status")
        if status_value not in _CONFORMITY_STATUSES:
            raise AcceptanceSpecError(
                f"{field}.status must be passed, pending, failed, or not_assessed"
            )
        basis_value = value.get("basis")
        basis = (
            None if basis_value is None else _nonempty_string(basis_value, field=f"{field}.basis")
        )
        evidence = _string_tuple(value.get("evidence", []), field=f"{field}.evidence")
        review_value = value.get("review")
        review = (
            None
            if review_value is None
            else ConformityReview.from_mapping(
                _mapping(review_value, field=f"{field}.review"),
                field=f"{field}.review",
            )
        )
        if status_value in {"passed", "failed"} and (
            basis is None or not evidence or review is None
        ):
            raise AcceptanceSpecError(
                f"{field} with status={status_value} requires basis, evidence, and review"
            )
        return cls(
            status=cast(ConformityStatus, status_value),
            basis=basis,
            evidence=evidence,
            review=review,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "basis": self.basis,
            "evidence": list(self.evidence),
            "review": None if self.review is None else self.review.to_dict(),
        }


@dataclass(frozen=True)
class TargetSpec:
    """A published numeric target evaluated over all successful repetitions."""

    id: str
    path: str
    transform: TargetTransform
    published_mean: float
    margin_absolute: float
    published_std: float | None = None
    published_std_ddof: int = 1
    offset: float = 0.0

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        field: str,
        default_id: str | None = None,
        diagnostic: bool = False,
    ) -> TargetSpec:
        _reject_unknown_keys(value, allowed=_TARGET_KEYS, field=field)
        raw_id = value.get("id", default_id)
        target_id = _nonempty_string(raw_id, field=f"{field}.id")
        if "path" not in value:
            if diagnostic:
                raise AcceptanceSpecError(f"{field}.path is required")
            split = _nonempty_string(value.get("split"), field=f"{field}.split")
            metric = _nonempty_string(value.get("metric"), field=f"{field}.metric")
            path = f"metrics.{split}.{metric}"
        else:
            if "split" in value or "metric" in value:
                raise AcceptanceSpecError(f"{field} must use either path or split/metric")
            path = _nonempty_string(value.get("path"), field=f"{field}.path")
        transform_value = value.get("transform", "identity")
        if transform_value not in _TRANSFORMS:
            raise AcceptanceSpecError(f"{field}.transform must be identity or one_minus")
        published_mean = _finite(value.get("published_mean"), field=f"{field}.published_mean")
        margin_absolute = _finite(value.get("margin_absolute"), field=f"{field}.margin_absolute")
        if margin_absolute < 0:
            raise AcceptanceSpecError(f"{field}.margin_absolute must be >= 0")
        published_std = _optional_finite(value.get("published_std"), field=f"{field}.published_std")
        if published_std is not None and published_std < 0:
            raise AcceptanceSpecError(f"{field}.published_std must be >= 0")
        ddof = value.get("published_std_ddof", 1)
        if isinstance(ddof, bool) or not isinstance(ddof, Integral) or int(ddof) not in {0, 1}:
            raise AcceptanceSpecError(f"{field}.published_std_ddof must be 0 or 1")
        offset = _finite(value.get("offset", 0.0), field=f"{field}.offset")
        return cls(
            id=target_id,
            path=path,
            transform=cast(TargetTransform, transform_value),
            published_mean=published_mean,
            margin_absolute=margin_absolute,
            published_std=published_std,
            published_std_ddof=int(ddof),
            offset=offset,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "transform": self.transform,
            "published_mean": self.published_mean,
            "published_std": self.published_std,
            "published_std_ddof": self.published_std_ddof,
            "margin_absolute": self.margin_absolute,
            "offset": self.offset,
        }


@dataclass(frozen=True)
class DiagnosticRule:
    """A per-run declarative diagnostic predicate."""

    path: str
    op: DiagnosticOperator
    value: Any = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, field: str) -> DiagnosticRule:
        _reject_unknown_keys(
            value,
            allowed=frozenset({"path", "op", "value"}),
            field=field,
        )
        path = _nonempty_string(value.get("path"), field=f"{field}.path")
        op_value = value.get("op")
        if op_value not in _DIAGNOSTIC_OPERATORS:
            raise AcceptanceSpecError(f"{field}.op is invalid")
        has_value = "value" in value
        expected = value.get("value")
        if op_value in {"eq", "gt", "ge", "lt", "le", "between"} and not has_value:
            raise AcceptanceSpecError(f"{field}.value is required for {op_value}")
        if op_value in _VALUELESS_DIAGNOSTIC_OPERATORS and has_value:
            raise AcceptanceSpecError(f"{field}.value is forbidden for {op_value}")
        if op_value in {"gt", "ge", "lt", "le"}:
            expected = _finite(expected, field=f"{field}.value")
        elif op_value == "between":
            if (
                not isinstance(expected, Sequence)
                or isinstance(expected, str | bytes)
                or len(expected) != 2
            ):
                raise AcceptanceSpecError(f"{field}.value must contain two finite bounds")
            lower = _finite(expected[0], field=f"{field}.value[0]")
            upper = _finite(expected[1], field=f"{field}.value[1]")
            if lower > upper:
                raise AcceptanceSpecError(f"{field}.value bounds must be ordered")
            expected = (lower, upper)
        elif op_value == "eq":
            _json_canonical(expected)
        return cls(path=path, op=cast(DiagnosticOperator, op_value), value=expected)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "op": self.op,
        }
        if self.op not in _VALUELESS_DIAGNOSTIC_OPERATORS:
            payload["value"] = list(self.value) if isinstance(self.value, tuple) else self.value
        return payload


def _parse_targets(
    value: Any,
    *,
    field: str,
    diagnostic: bool = False,
) -> tuple[TargetSpec, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise AcceptanceSpecError(f"{field} must be a sequence")
    targets = tuple(
        TargetSpec.from_mapping(
            _mapping(item, field=f"{field}[{index}]"),
            field=f"{field}[{index}]",
            diagnostic=diagnostic,
        )
        for index, item in enumerate(value)
    )
    ids = [target.id for target in targets]
    if len(ids) != len(set(ids)):
        raise AcceptanceSpecError(f"{field} target ids must be unique")
    return tuple(sorted(targets, key=lambda target: target.id))


@dataclass(frozen=True)
class AcceptanceSpec:
    """Complete, method-agnostic scientific acceptance specification."""

    protocol_id: str
    method_id: str
    repetitions: int
    fidelity_ceiling: FidelityStatus
    conformity: ConformitySpec
    target: TargetSpec
    secondary_targets: tuple[TargetSpec, ...] = ()
    informational_targets: tuple[TargetSpec, ...] = ()
    diagnostic_targets: tuple[TargetSpec, ...] = ()
    required_diagnostics: tuple[DiagnosticRule, ...] = ()
    deviations: tuple[str, ...] = ()
    equivalences: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AcceptanceSpec:
        _reject_unknown_keys(value, allowed=_ACCEPTANCE_KEYS, field="acceptance spec")
        protocol_id = _nonempty_string(value.get("protocol_id"), field="protocol_id")
        method_id = _nonempty_string(value.get("method_id"), field="method_id")
        repetitions = value.get("repetitions")
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, Integral)
            or int(repetitions) < 2
        ):
            raise AcceptanceSpecError("repetitions must be an integer >= 2")
        ceiling_value = value.get("fidelity_ceiling")
        if ceiling_value not in _FIDELITY_STATUSES:
            raise AcceptanceSpecError(
                "fidelity_ceiling must be paper_matched, paper_approx, or not_claimable"
            )
        conformity = ConformitySpec.from_mapping(
            _mapping(value.get("conformity"), field="conformity")
        )
        target = TargetSpec.from_mapping(
            _mapping(value.get("target"), field="target"),
            field="target",
            default_id="primary",
        )
        secondary = _parse_targets(value.get("secondary_targets", []), field="secondary_targets")
        informational = _parse_targets(
            value.get("informational_targets", []), field="informational_targets"
        )
        diagnostic_targets = _parse_targets(
            value.get("diagnostic_targets", []),
            field="diagnostic_targets",
            diagnostic=True,
        )
        all_ids = [
            target.id,
            *(item.id for item in secondary),
            *(item.id for item in informational),
            *(item.id for item in diagnostic_targets),
        ]
        if len(all_ids) != len(set(all_ids)):
            raise AcceptanceSpecError("target ids must be unique across all target groups")
        raw_rules = value.get("required_diagnostics", [])
        if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, str | bytes):
            raise AcceptanceSpecError("required_diagnostics must be a sequence")
        rules = tuple(
            DiagnosticRule.from_mapping(
                _mapping(item, field=f"required_diagnostics[{index}]"),
                field=f"required_diagnostics[{index}]",
            )
            for index, item in enumerate(raw_rules)
        )
        rule_keys = [(rule.path, rule.op, _json_canonical(rule.value)) for rule in rules]
        if len(rule_keys) != len(set(rule_keys)):
            raise AcceptanceSpecError("required_diagnostics must not contain duplicates")
        rules = tuple(
            rule
            for _key, rule in sorted(
                zip(rule_keys, rules, strict=True),
                key=lambda item: item[0],
            )
        )
        return cls(
            protocol_id=protocol_id,
            method_id=method_id,
            repetitions=int(repetitions),
            fidelity_ceiling=cast(FidelityStatus, ceiling_value),
            conformity=conformity,
            target=target,
            secondary_targets=secondary,
            informational_targets=informational,
            diagnostic_targets=diagnostic_targets,
            required_diagnostics=rules,
            deviations=_string_tuple(value.get("deviations", []), field="deviations"),
            equivalences=_string_tuple(value.get("equivalences", []), field="equivalences"),
            unknowns=_string_tuple(value.get("unknowns", []), field="unknowns"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "method_id": self.method_id,
            "repetitions": self.repetitions,
            "fidelity_ceiling": self.fidelity_ceiling,
            "conformity": self.conformity.to_dict(),
            "target": self.target.to_dict(),
            "secondary_targets": [target.to_dict() for target in self.secondary_targets],
            "informational_targets": [target.to_dict() for target in self.informational_targets],
            "diagnostic_targets": [target.to_dict() for target in self.diagnostic_targets],
            "required_diagnostics": [rule.to_dict() for rule in self.required_diagnostics],
            "deviations": list(self.deviations),
            "equivalences": list(self.equivalences),
            "unknowns": list(self.unknowns),
        }


def parse_acceptance_spec(value: Mapping[str, Any]) -> AcceptanceSpec:
    """Parse and validate a declarative acceptance mapping."""

    return AcceptanceSpec.from_mapping(_mapping(value, field="acceptance spec"))


@dataclass(frozen=True)
class TargetAssessment:
    """A target summary and its two independent numeric gates."""

    id: str
    path: str
    transform: TargetTransform
    expected_count: int
    observed_count: int
    available: bool
    summary: Mapping[str, Any] | None
    published_mean: float
    published_std: float | None
    published_std_ddof: int
    std_absolute_difference: float | None
    absolute_difference: float | None
    margin_absolute: float
    target_in_ci95: bool
    within_margin: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "transform": self.transform,
            "expected_count": self.expected_count,
            "observed_count": self.observed_count,
            "available": self.available,
            "summary": None if self.summary is None else dict(self.summary),
            "published_mean": self.published_mean,
            "published_std": self.published_std,
            "published_std_ddof": self.published_std_ddof,
            "std_absolute_difference": self.std_absolute_difference,
            "absolute_difference": self.absolute_difference,
            "margin_absolute": self.margin_absolute,
            "target_in_ci95": self.target_in_ci95,
            "within_margin": self.within_margin,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class DiagnosticFailure:
    seed: int
    path: str
    op: DiagnosticOperator
    expected: Any
    actual: Any
    present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "path": self.path,
            "op": self.op,
            "expected": list(self.expected) if isinstance(self.expected, tuple) else self.expected,
            "actual": self.actual,
            "present": self.present,
        }


@dataclass(frozen=True)
class DiagnosticAssessment:
    path: str
    op: DiagnosticOperator
    expected: Any
    checked_run_count: int
    passed_run_count: int
    failures: tuple[DiagnosticFailure, ...]

    @property
    def passed(self) -> bool:
        return not self.failures and self.checked_run_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "op": self.op,
            "expected": list(self.expected) if isinstance(self.expected, tuple) else self.expected,
            "checked_run_count": self.checked_run_count,
            "passed_run_count": self.passed_run_count,
            "failure_count": len(self.failures),
            "passed": self.passed,
            "failures": [failure.to_dict() for failure in self.failures],
        }


@dataclass(frozen=True)
class AcceptanceReport:
    """Canonical three-state assessment, including its own payload digest."""

    protocol_id: str
    method_id: str
    assessment_status: AssessmentStatus
    fidelity_status: FidelityStatus
    fidelity_ceiling: FidelityStatus
    repetitions_expected: int
    runs: tuple[Mapping[str, Any], ...]
    conformity: ConformitySpec
    primary_target: TargetAssessment
    secondary_targets: tuple[TargetAssessment, ...]
    informational_targets: tuple[TargetAssessment, ...]
    diagnostic_targets: tuple[TargetAssessment, ...]
    required_diagnostics: tuple[DiagnosticAssessment, ...]
    deviations: tuple[str, ...]
    equivalences: tuple[str, ...]
    unknowns: tuple[str, ...]
    reasons: tuple[str, ...]
    acceptance_sha256: str

    def _payload_dict(self) -> dict[str, Any]:
        diagnostic_failures = sorted(
            (
                failure.to_dict()
                for assessment in self.required_diagnostics
                for failure in assessment.failures
            ),
            key=lambda item: (
                item["seed"],
                item["path"],
                item["op"],
                _json_canonical(item["expected"]),
            ),
        )
        return {
            "schema_version": 1,
            "protocol_id": self.protocol_id,
            "method_id": self.method_id,
            "assessment_status": self.assessment_status,
            "fidelity_status": self.fidelity_status,
            "fidelity_ceiling": self.fidelity_ceiling,
            "repetitions_expected": self.repetitions_expected,
            "runs": [dict(run) for run in self.runs],
            "conformity": self.conformity.to_dict(),
            "primary_target": self.primary_target.to_dict(),
            "secondary_targets": [target.to_dict() for target in self.secondary_targets],
            "informational_targets": [target.to_dict() for target in self.informational_targets],
            "diagnostic_targets": [target.to_dict() for target in self.diagnostic_targets],
            "required_diagnostics": [
                assessment.to_dict() for assessment in self.required_diagnostics
            ],
            "diagnostic_failures": diagnostic_failures,
            "deviations": list(self.deviations),
            "equivalences": list(self.equivalences),
            "unknowns": list(self.unknowns),
            "reasons": list(self.reasons),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["acceptance_sha256"] = self.acceptance_sha256
        return payload


@dataclass(frozen=True)
class _ObservedRun:
    seed: int
    status: str
    run_id: str | None
    payload: Mapping[str, Any]

    def report_entry(self) -> dict[str, Any]:
        return {"seed": self.seed, "status": self.status, "run_id": self.run_id}


def _observed_runs(
    runs: Iterable[Mapping[str, Any]],
    *,
    expected_method_id: str,
) -> tuple[_ObservedRun, ...]:
    observed: list[_ObservedRun] = []
    for index, raw in enumerate(runs):
        payload = _mapping(raw, field=f"runs[{index}]")
        config = _mapping(payload.get("config"), field=f"runs[{index}].config")
        method = _mapping(config.get("method"), field=f"runs[{index}].config.method")
        method_id = _nonempty_string(
            method.get("id"),
            field=f"runs[{index}].config.method.id",
        )
        if method_id != expected_method_id:
            raise AcceptanceSpecError(
                f"runs[{index}].config.method.id differs from acceptance spec method_id"
            )
        run_value = payload.get("run")
        run = _mapping(run_value, field=f"runs[{index}].run") if run_value is not None else payload
        seed_value = run.get("seed")
        if isinstance(seed_value, bool) or not isinstance(seed_value, Integral) or seed_value < 0:
            raise AcceptanceSpecError(f"runs[{index}].run.seed must be a non-negative integer")
        status_value = run.get("status")
        if status_value not in _RUN_STATUSES:
            raise AcceptanceSpecError(
                f"runs[{index}].run.status must be success, failed, or not_evaluable"
            )
        run_id_value = run.get("run_id")
        run_id = None if run_id_value is None else str(run_id_value)
        observed.append(
            _ObservedRun(
                seed=int(seed_value),
                status=str(status_value),
                run_id=run_id,
                payload=payload,
            )
        )
    observed.sort(key=lambda run: run.seed)
    seeds = [run.seed for run in observed]
    if len(seeds) != len(set(seeds)):
        raise AcceptanceSpecError("runs contain duplicate seeds")
    return tuple(observed)


def _nested(payload: Mapping[str, Any], dotted_path: str) -> tuple[bool, Any]:
    current: Any = payload
    for part in dotted_path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif (
            isinstance(current, Sequence)
            and not isinstance(current, str | bytes)
            and part.isdigit()
            and int(part) < len(current)
        ):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def _target_value(payload: Mapping[str, Any], target: TargetSpec) -> float | None:
    exists, value = _nested(payload, target.path)
    if not exists or isinstance(value, bool) or not isinstance(value, Real):
        return None
    normalized = float(value)
    if not math.isfinite(normalized):
        return None
    if target.transform == "one_minus":
        normalized = 1.0 - normalized
    return normalized + target.offset


def _target_assessment(
    runs: tuple[_ObservedRun, ...],
    *,
    target: TargetSpec,
    expected_count: int,
    require_ci: bool,
) -> TargetAssessment:
    values = [
        value
        for run in runs
        if run.status == "success" and (value := _target_value(run.payload, target)) is not None
    ]
    summary = summarize_numeric(values) if values else None
    available = len(values) == expected_count
    mean = None if summary is None else float(summary["mean"])
    absolute_difference = None if mean is None else abs(mean - target.published_mean)
    ci_low = None if summary is None else summary["ci95_low"]
    ci_high = None if summary is None else summary["ci95_high"]
    target_in_ci95 = bool(
        available
        and ci_low is not None
        and ci_high is not None
        and float(ci_low) <= target.published_mean <= float(ci_high)
    )
    within_margin = bool(
        available
        and absolute_difference is not None
        and absolute_difference <= target.margin_absolute
    )
    if summary is None or target.published_std is None:
        std_absolute_difference = None
    else:
        observed_std = (
            float(summary["std"])
            if target.published_std_ddof == 1
            else float(summary["population_std"])
        )
        std_absolute_difference = abs(observed_std - target.published_std)
    return TargetAssessment(
        id=target.id,
        path=target.path,
        transform=target.transform,
        expected_count=expected_count,
        observed_count=len(values),
        available=available,
        summary=summary,
        published_mean=target.published_mean,
        published_std=target.published_std,
        published_std_ddof=target.published_std_ddof,
        std_absolute_difference=std_absolute_difference,
        absolute_difference=absolute_difference,
        margin_absolute=target.margin_absolute,
        target_in_ci95=target_in_ci95,
        within_margin=within_margin,
        passed=available and within_margin and (target_in_ci95 or not require_ci),
    )


def _numeric_rule(actual: Any, expected: Any, op: DiagnosticOperator) -> bool:
    if isinstance(actual, bool) or not isinstance(actual, Real):
        return False
    number = float(actual)
    if not math.isfinite(number):
        return False
    if op == "between":
        lower, upper = cast(tuple[float, float], expected)
        return lower <= number <= upper
    target = float(expected)
    if op == "gt":
        return number > target
    if op == "ge":
        return number >= target
    if op == "lt":
        return number < target
    return number <= target


def _rule_passes(payload: Mapping[str, Any], rule: DiagnosticRule) -> tuple[bool, bool, Any]:
    present, actual = _nested(payload, rule.path)
    if rule.op == "present":
        return present, present, actual
    if not present:
        return False, False, None
    if rule.op == "truthy":
        return bool(actual), True, actual
    if rule.op == "nonempty":
        try:
            return len(actual) > 0, True, actual
        except TypeError:
            return False, True, actual
    if rule.op == "eq":
        if (
            isinstance(actual, Real)
            and not isinstance(actual, bool)
            and isinstance(rule.value, Real)
            and not isinstance(rule.value, bool)
        ):
            return (
                math.isclose(float(actual), float(rule.value), rel_tol=1e-9, abs_tol=1e-12),
                True,
                actual,
            )
        return actual == rule.value, True, actual
    return _numeric_rule(actual, rule.value, rule.op), True, actual


def _diagnostic_assessments(
    runs: tuple[_ObservedRun, ...], rules: tuple[DiagnosticRule, ...]
) -> tuple[DiagnosticAssessment, ...]:
    successful = tuple(run for run in runs if run.status == "success")
    assessments: list[DiagnosticAssessment] = []
    for rule in rules:
        failures: list[DiagnosticFailure] = []
        passed_count = 0
        for run in successful:
            passed, present, actual = _rule_passes(run.payload, rule)
            if passed:
                passed_count += 1
            else:
                failures.append(
                    DiagnosticFailure(
                        seed=run.seed,
                        path=rule.path,
                        op=rule.op,
                        expected=rule.value,
                        actual=actual,
                        present=present,
                    )
                )
        assessments.append(
            DiagnosticAssessment(
                path=rule.path,
                op=rule.op,
                expected=rule.value,
                checked_run_count=len(successful),
                passed_run_count=passed_count,
                failures=tuple(failures),
            )
        )
    return tuple(assessments)


def _report_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_canonical(payload).encode("utf-8")).hexdigest()


def evaluate_acceptance(
    spec: AcceptanceSpec | Mapping[str, Any],
    runs: Iterable[Mapping[str, Any]],
) -> AcceptanceReport:
    """Evaluate authenticated, reconciled run payloads against ``spec``.

    Authentication and reconciliation stay upstream.  This function performs
    no filesystem access and never dispatches on ``method_id``.
    """

    parsed = (
        parse_acceptance_spec(spec.to_dict())
        if isinstance(spec, AcceptanceSpec)
        else parse_acceptance_spec(spec)
    )
    observed = _observed_runs(runs, expected_method_id=parsed.method_id)
    successful_count = sum(run.status == "success" for run in observed)
    complete = len(observed) == parsed.repetitions and successful_count == parsed.repetitions

    primary = _target_assessment(
        observed,
        target=parsed.target,
        expected_count=parsed.repetitions,
        require_ci=True,
    )
    secondary = tuple(
        _target_assessment(
            observed,
            target=target,
            expected_count=parsed.repetitions,
            require_ci=True,
        )
        for target in parsed.secondary_targets
    )
    informational = tuple(
        _target_assessment(
            observed,
            target=target,
            expected_count=parsed.repetitions,
            require_ci=True,
        )
        for target in parsed.informational_targets
    )
    diagnostic_targets = tuple(
        _target_assessment(
            observed,
            target=target,
            expected_count=parsed.repetitions,
            require_ci=False,
        )
        for target in parsed.diagnostic_targets
    )
    diagnostics = _diagnostic_assessments(observed, parsed.required_diagnostics)

    missing_gating_target = not primary.available or any(
        not target.available for target in (*secondary, *diagnostic_targets)
    )
    numeric_gates_passed = (
        primary.passed
        and all(target.passed for target in secondary)
        and all(target.passed for target in diagnostic_targets)
    )
    diagnostics_passed = all(
        assessment.passed and assessment.checked_run_count == parsed.repetitions
        for assessment in diagnostics
    )
    if not parsed.required_diagnostics:
        diagnostics_passed = True

    reasons: list[str] = []
    if not complete:
        reasons.append("repetitions_incomplete_or_non_success")
    if complete and not primary.available:
        reasons.append("primary_target_missing")
    if any(not target.available for target in secondary):
        reasons.append("secondary_target_missing")
    if any(not target.available for target in diagnostic_targets):
        reasons.append("diagnostic_target_missing")
    if primary.available and not primary.target_in_ci95:
        reasons.append("primary_target_outside_ci95")
    if primary.available and not primary.within_margin:
        reasons.append("primary_target_margin_exceeded")
    if any(target.available and not target.target_in_ci95 for target in secondary):
        reasons.append("secondary_target_outside_ci95")
    if any(target.available and not target.within_margin for target in secondary):
        reasons.append("secondary_target_margin_exceeded")
    if any(target.available and not target.within_margin for target in diagnostic_targets):
        reasons.append("diagnostic_target_margin_exceeded")
    if not diagnostics_passed:
        reasons.append("required_diagnostics_failed")
    if parsed.conformity.status != "passed":
        reasons.append(f"conformity={parsed.conformity.status}")
    if parsed.deviations:
        reasons.append("documented_deviations")
    if parsed.unknowns:
        reasons.append("critical_unknowns")
    if parsed.fidelity_ceiling != "paper_matched":
        reasons.append(f"fidelity_ceiling={parsed.fidelity_ceiling}")

    conformity_unresolved = parsed.conformity.status in {"pending", "not_assessed"}
    if not complete or (complete and not primary.available) or conformity_unresolved:
        assessment_status: AssessmentStatus = "not_evaluable"
    elif (
        missing_gating_target
        or not numeric_gates_passed
        or not diagnostics_passed
        or parsed.conformity.status != "passed"
    ):
        assessment_status = "failed"
    else:
        assessment_status = "passed"

    if (
        assessment_status == "not_evaluable"
        or parsed.fidelity_ceiling == "not_claimable"
        or parsed.unknowns
    ):
        fidelity_status: FidelityStatus = "not_claimable"
    elif (
        assessment_status == "passed"
        and parsed.fidelity_ceiling == "paper_matched"
        and not parsed.deviations
    ):
        fidelity_status = "paper_matched"
    else:
        fidelity_status = "paper_approx"

    report_without_hash = AcceptanceReport(
        protocol_id=parsed.protocol_id,
        method_id=parsed.method_id,
        assessment_status=assessment_status,
        fidelity_status=fidelity_status,
        fidelity_ceiling=parsed.fidelity_ceiling,
        repetitions_expected=parsed.repetitions,
        runs=tuple(run.report_entry() for run in observed),
        conformity=parsed.conformity,
        primary_target=primary,
        secondary_targets=secondary,
        informational_targets=informational,
        diagnostic_targets=diagnostic_targets,
        required_diagnostics=diagnostics,
        deviations=parsed.deviations,
        equivalences=parsed.equivalences,
        unknowns=parsed.unknowns,
        reasons=tuple(reasons),
        acceptance_sha256="",
    )
    digest = _report_digest(report_without_hash._payload_dict())
    return replace(report_without_hash, acceptance_sha256=digest)


__all__ = [
    "AcceptanceReport",
    "AcceptanceSpec",
    "AcceptanceSpecError",
    "AssessmentStatus",
    "ConformityReview",
    "ConformitySpec",
    "ConformityStatus",
    "DiagnosticAssessment",
    "DiagnosticFailure",
    "DiagnosticOperator",
    "DiagnosticRule",
    "FidelityStatus",
    "TargetAssessment",
    "TargetSpec",
    "TargetTransform",
    "evaluate_acceptance",
    "parse_acceptance_spec",
]
