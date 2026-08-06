from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bench.utils.io import atomic_write_json

from .aggregate import _critical_95
from .errors import CampaignError
from .executor import validate_result_directory
from .manifest import load_manifest, sha256_file
from .models import CampaignTask
from .reconcile import materialize_reconcile_paths
from .scientific_gates import evaluate_gate, load_gate_registry

_FIDELITY_STATUSES = {"paper_matched", "paper_approx", "not_claimable"}
_PROTOCOL_CONFORMITY_STATUSES = {"pending", "passed", "failed", "not_assessed"}
_CONTROL_MODES = {"learner_0", "learner_1", "learner_2", "combining_only"}


@dataclass(frozen=True)
class PaperAcceptanceReport:
    campaign_id: str
    status_counts: dict[str, int]
    protocol_count: int
    report_path: str
    matrix_path: str
    secondary_matrix_path: str
    informational_matrix_path: str


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"cannot read {label}: {path}") from exc
    if not isinstance(raw, dict):
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{label} must be a mapping")
    return raw


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field} must be a list[str]")
    return [str(item) for item in value]


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field} must be finite")
    return result


def _validate_target(target: Any, *, field: str, require_id: bool = False) -> None:
    if not isinstance(target, Mapping):
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field} must be a mapping")
    if require_id and (not isinstance(target.get("id"), str) or not target["id"]):
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.id is required")
    if target.get("transform", "identity") not in {"identity", "one_minus"}:
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.transform is invalid")
    metric_path = target.get("path")
    if metric_path is not None:
        if not isinstance(metric_path, str) or not metric_path:
            raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.path must be non-empty")
    else:
        for key in ("split", "metric"):
            if not isinstance(target.get(key), str) or not target[key]:
                raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.{key} is required")
    _finite_number(target.get("published_mean"), field=f"{field}.published_mean")
    if target.get("published_std") is not None:
        published_std = _finite_number(target.get("published_std"), field=f"{field}.published_std")
        if published_std < 0:
            raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.published_std must be >= 0")
    published_std_ddof = target.get("published_std_ddof", 1)
    if (
        isinstance(published_std_ddof, bool)
        or not isinstance(published_std_ddof, int)
        or published_std_ddof not in {0, 1}
    ):
        raise CampaignError(
            "E_PAPER_ACCEPTANCE_SCHEMA",
            f"{field}.published_std_ddof must equal 0 or 1",
        )
    margin = _finite_number(target.get("margin_absolute"), field=f"{field}.margin_absolute")
    if margin < 0:
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.margin_absolute must be >= 0")


def _validate_diagnostic_target(target: Any, *, field: str) -> None:
    if not isinstance(target, Mapping):
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field} must be a mapping")
    for key in ("id", "path"):
        if not isinstance(target.get(key), str) or not target[key]:
            raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.{key} is required")
    _finite_number(target.get("published_mean"), field=f"{field}.published_mean")
    margin = _finite_number(target.get("margin_absolute"), field=f"{field}.margin_absolute")
    if margin < 0:
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.margin_absolute must be >= 0")
    if target.get("offset") is not None:
        _finite_number(target.get("offset"), field=f"{field}.offset")


def _validate_control_target(target: Any, *, field: str) -> None:
    _validate_target(target, field=field, require_id=True)
    if target.get("control_mode") not in _CONTROL_MODES:
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.control_mode is invalid")


def _validate_confidence_candidate(candidate: Any, *, field: str) -> None:
    if not isinstance(candidate, Mapping):
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field} must be a mapping")
    if not isinstance(candidate.get("id"), str) or not candidate["id"]:
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.id is required")
    if candidate.get("estimator") not in {"training_accuracy", "kfold_oof"}:
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.estimator is invalid")
    if candidate.get("interval") not in {"wald", "wilson", "clopper_pearson"}:
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.interval is invalid")
    folds = candidate.get("folds")
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.folds must be >= 2")
    if candidate.get("role") not in {"v1_control", "primary_reconstruction", "conditional"}:
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", f"{field}.role is invalid")
    if candidate.get("protocol_conformity") != "pending":
        raise CampaignError(
            "E_PAPER_ACCEPTANCE_SCHEMA",
            f"{field}.protocol_conformity must remain pending before diagnostic review",
        )
    if candidate.get("test_information_used") is not False:
        raise CampaignError(
            "E_PAPER_ACCEPTANCE_SCHEMA",
            f"{field}.test_information_used must be false",
        )


def _load_acceptance_cards(path: Path) -> dict[str, dict[str, Any]]:
    raw = _load_mapping(path, label="paper acceptance registry")
    if raw.get("schema_version") != 1:
        raise CampaignError(
            "E_PAPER_ACCEPTANCE_SCHEMA", "paper acceptance registry must use schema_version=1"
        )
    protocols = raw.get("protocols")
    if not isinstance(protocols, Mapping) or not protocols:
        raise CampaignError("E_PAPER_ACCEPTANCE_SCHEMA", "protocols must be a non-empty mapping")
    cards: dict[str, dict[str, Any]] = {}
    for protocol_id, value in protocols.items():
        if not isinstance(protocol_id, str) or not protocol_id or not isinstance(value, Mapping):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA", "each protocol must be a named mapping"
            )
        card = dict(value)
        method_id = card.get("method_id")
        repetitions = card.get("repetitions")
        target = card.get("target")
        if not isinstance(method_id, str) or not method_id:
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA", f"protocols.{protocol_id}.method_id is required"
            )
        if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.repetitions must be greater than zero",
            )
        _validate_target(target, field=f"protocols.{protocol_id}.target")
        secondary_targets = card.get("secondary_targets", [])
        if not isinstance(secondary_targets, list):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.secondary_targets must be a list",
            )
        for index, secondary in enumerate(secondary_targets):
            _validate_target(
                secondary,
                field=f"protocols.{protocol_id}.secondary_targets[{index}]",
                require_id=True,
            )
        secondary_ids = [str(secondary["id"]) for secondary in secondary_targets]
        if len(secondary_ids) != len(set(secondary_ids)):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.secondary_targets ids must be unique",
            )
        informational_targets = card.get("informational_targets", [])
        if not isinstance(informational_targets, list):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.informational_targets must be a list",
            )
        for index, informational in enumerate(informational_targets):
            _validate_target(
                informational,
                field=f"protocols.{protocol_id}.informational_targets[{index}]",
                require_id=True,
            )
        informational_ids = [str(informational["id"]) for informational in informational_targets]
        if len(informational_ids) != len(set(informational_ids)):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.informational_targets ids must be unique",
            )
        diagnostic_targets = card.get("diagnostic_targets", [])
        if not isinstance(diagnostic_targets, list):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.diagnostic_targets must be a list",
            )
        for index, diagnostic in enumerate(diagnostic_targets):
            _validate_diagnostic_target(
                diagnostic,
                field=f"protocols.{protocol_id}.diagnostic_targets[{index}]",
            )
        diagnostic_ids = [str(diagnostic["id"]) for diagnostic in diagnostic_targets]
        if len(diagnostic_ids) != len(set(diagnostic_ids)):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.diagnostic_targets ids must be unique",
            )
        control_targets = card.get("control_targets", [])
        if not isinstance(control_targets, list):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.control_targets must be a list",
            )
        for index, control in enumerate(control_targets):
            _validate_control_target(
                control,
                field=f"protocols.{protocol_id}.control_targets[{index}]",
            )
        control_ids = [str(control["id"]) for control in control_targets]
        control_modes = [str(control["control_mode"]) for control in control_targets]
        if len(control_ids) != len(set(control_ids)) or len(control_modes) != len(
            set(control_modes)
        ):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.control_targets ids and control modes must be unique",
            )
        confidence_candidates = card.get("confidence_candidates", [])
        if not isinstance(confidence_candidates, list):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.confidence_candidates must be a list",
            )
        for index, candidate in enumerate(confidence_candidates):
            _validate_confidence_candidate(
                candidate,
                field=f"protocols.{protocol_id}.confidence_candidates[{index}]",
            )
        candidate_ids = [str(candidate["id"]) for candidate in confidence_candidates]
        candidate_settings = [
            (str(candidate["estimator"]), str(candidate["interval"]))
            for candidate in confidence_candidates
        ]
        if len(candidate_ids) != len(set(candidate_ids)) or len(candidate_settings) != len(
            set(candidate_settings)
        ):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.confidence_candidates must be unique",
            )
        protocol_conformity = card.get("protocol_conformity")
        if (
            protocol_conformity is not None
            and protocol_conformity not in _PROTOCOL_CONFORMITY_STATUSES
        ):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.protocol_conformity is invalid",
            )
        rules = card.get("required_diagnostics", [])
        if not isinstance(rules, list) or any(not isinstance(rule, Mapping) for rule in rules):
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_SCHEMA",
                f"protocols.{protocol_id}.required_diagnostics must be a list[mapping]",
            )
        for index, rule in enumerate(rules):
            if not isinstance(rule.get("path"), str) or not rule["path"]:
                raise CampaignError(
                    "E_PAPER_ACCEPTANCE_SCHEMA",
                    f"{protocol_id}.required_diagnostics[{index}].path is required",
                )
            if rule.get("op") not in {
                "present",
                "eq",
                "gt",
                "ge",
                "lt",
                "le",
                "between",
                "truthy",
                "nonempty",
            }:
                raise CampaignError(
                    "E_PAPER_ACCEPTANCE_SCHEMA",
                    f"{protocol_id}.required_diagnostics[{index}].op is invalid",
                )
        _string_list(card.get("known_deviations", []), field=f"{protocol_id}.known_deviations")
        _string_list(
            card.get("documented_equivalences", []),
            field=f"{protocol_id}.documented_equivalences",
        )
        _string_list(card.get("critical_unknowns", []), field=f"{protocol_id}.critical_unknowns")
        _string_list(
            card.get("environment_differences", []),
            field=f"{protocol_id}.environment_differences",
        )
        cards[protocol_id] = card
    return cards


def _nested(payload: Mapping[str, Any], dotted_path: str) -> tuple[bool, Any]:
    current: Any = payload
    for part in dotted_path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
            continue
        return False, None
    return True, current


def _numeric_comparison(actual: Any, expected: Any, op: str) -> bool:
    if isinstance(actual, bool) or not isinstance(actual, int | float):
        return False
    number = float(actual)
    if not math.isfinite(number):
        return False
    if op == "between":
        if (
            not isinstance(expected, list)
            or len(expected) != 2
            or any(isinstance(item, bool) or not isinstance(item, int | float) for item in expected)
        ):
            return False
        return float(expected[0]) <= number <= float(expected[1])
    if isinstance(expected, bool) or not isinstance(expected, int | float):
        return False
    target = float(expected)
    if op == "gt":
        return number > target
    if op == "ge":
        return number >= target
    if op == "lt":
        return number < target
    if op == "le":
        return number <= target
    return False


def _rule_passes(payload: Mapping[str, Any], rule: Mapping[str, Any]) -> tuple[bool, Any]:
    exists, actual = _nested(payload, str(rule["path"]))
    op = str(rule["op"])
    if op == "present":
        return exists, actual
    if not exists:
        return False, actual
    if op == "truthy":
        return bool(actual), actual
    if op == "nonempty":
        try:
            return len(actual) > 0, actual
        except TypeError:
            return False, actual
    if op == "eq":
        expected = rule.get("value")
        if (
            not isinstance(actual, bool)
            and not isinstance(expected, bool)
            and isinstance(actual, int | float)
            and isinstance(expected, int | float)
        ):
            return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12), actual
        return actual == expected, actual
    return _numeric_comparison(actual, rule.get("value"), op), actual


def _metric_value(payload: Mapping[str, Any], target: Mapping[str, Any]) -> float | None:
    if target.get("path") is not None:
        exists, value = _nested(payload, str(target["path"]))
        if not exists:
            return None
    else:
        metrics = payload.get("metrics")
        split = metrics.get(target["split"]) if isinstance(metrics, Mapping) else None
        value = split.get(target["metric"]) if isinstance(split, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    if target.get("transform", "identity") == "one_minus":
        result = 1.0 - result
    return result


def _diagnostic_value(payload: Mapping[str, Any], target: Mapping[str, Any]) -> float | None:
    exists, value = _nested(payload, str(target["path"]))
    if (
        not exists
        or isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value) + float(target.get("offset", 0.0))


def _summarize_values(
    values: list[float],
    *,
    target: Mapping[str, Any],
    expected: int,
    metric: str,
) -> dict[str, Any]:
    available = len(values) == expected
    mean = std = ci_low = ci_high = absolute_difference = None
    target_in_ci95 = within_margin = False
    published_mean = float(target["published_mean"])
    published_std = None if target.get("published_std") is None else float(target["published_std"])
    published_std_ddof = int(target.get("published_std_ddof", 1))
    margin = float(target["margin_absolute"])
    if available:
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        sample_std = float(arr.std(ddof=1)) if len(values) > 1 else 0.0
        std = float(arr.std(ddof=published_std_ddof)) if len(values) > published_std_ddof else 0.0
        half_width = _critical_95(len(values)) * sample_std / math.sqrt(len(values))
        ci_low = mean - half_width
        ci_high = mean + half_width
        absolute_difference = abs(mean - published_mean)
        target_in_ci95 = ci_low <= published_mean <= ci_high
        within_margin = absolute_difference <= margin
    return {
        "id": target.get("id", "primary"),
        "metric": metric,
        "available": available,
        "n": len(values),
        "replication_mean": mean,
        "replication_std": std,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "published_mean": published_mean,
        "published_std": published_std,
        "published_std_ddof": published_std_ddof,
        "std_absolute_difference": (
            None if std is None or published_std is None else abs(std - published_std)
        ),
        "absolute_difference": absolute_difference,
        "margin_absolute": margin,
        "target_in_ci95": target_in_ci95,
        "within_margin": within_margin,
    }


def _summarize_target(
    successful: list[tuple[CampaignTask, dict[str, Any]]],
    *,
    target: Mapping[str, Any],
    expected: int,
) -> dict[str, Any]:
    values = [
        value for _, payload in successful if (value := _metric_value(payload, target)) is not None
    ]
    metric = (
        f"path:{target['path']}:{target.get('transform', 'identity')}"
        if target.get("path") is not None
        else f"{target['split']}.{target['metric']}:{target.get('transform', 'identity')}"
    )
    return _summarize_values(
        values,
        target=target,
        expected=expected,
        metric=metric,
    )


def _summarize_diagnostic_target(
    successful: list[tuple[CampaignTask, dict[str, Any]]],
    *,
    target: Mapping[str, Any],
    expected: int,
) -> dict[str, Any]:
    values = [
        value
        for _, payload in successful
        if (value := _diagnostic_value(payload, target)) is not None
    ]
    return _summarize_values(
        values,
        target=target,
        expected=expected,
        metric=f"diagnostic:{target['path']}",
    )


def _result_status(
    *,
    complete: bool,
    metric_available: bool,
    target_in_ci95: bool,
    within_margin: bool,
) -> str:
    if not complete:
        return "incomplete"
    if not metric_available:
        return "target_missing"
    if not within_margin:
        return "failed_margin"
    if not target_in_ci95:
        return "failed_ci95"
    return "matched"


def _write_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "protocol_id",
        "method_id",
        "status",
        "protocol_status",
        "result_status",
        "equation_conformity",
        "protocol_conformity",
        "n_expected",
        "n_success",
        "metric",
        "replication_mean",
        "replication_std",
        "ci95_low",
        "ci95_high",
        "published_mean",
        "published_std",
        "published_std_ddof",
        "std_absolute_difference",
        "absolute_difference",
        "margin_absolute",
        "target_in_ci95",
        "within_margin",
        "secondary_targets_ok",
        "diagnostic_targets_ok",
        "diagnostics_ok",
        "algorithmic_conformity",
        "scientific_gate_allowed",
        "scientific_gate_blockers",
        "reasons",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_secondary_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "protocol_id",
        "method_id",
        "protocol_status",
        "id",
        "metric",
        "available",
        "n",
        "replication_mean",
        "replication_std",
        "ci95_low",
        "ci95_high",
        "published_mean",
        "published_std",
        "published_std_ddof",
        "std_absolute_difference",
        "absolute_difference",
        "margin_absolute",
        "target_in_ci95",
        "within_margin",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _status_for_protocol(
    *,
    complete: bool,
    metric_available: bool,
    manifest_ceiling: str,
    critical_unknowns: list[str],
    known_deviations: list[str],
    target_in_ci95: bool,
    within_margin: bool,
    secondary_targets_ok: bool,
    diagnostic_targets_ok: bool,
    diagnostics_ok: bool,
    conformity_status: str,
    protocol_conformity: str,
) -> str:
    if (
        not complete
        or not metric_available
        or manifest_ceiling == "not_claimable"
        or critical_unknowns
    ):
        return "not_claimable"
    if (
        manifest_ceiling == "paper_matched"
        and not known_deviations
        and target_in_ci95
        and within_margin
        and secondary_targets_ok
        and diagnostic_targets_ok
        and diagnostics_ok
        and conformity_status == "passed"
        and protocol_conformity == "passed"
    ):
        return "paper_matched"
    return "paper_approx"


def evaluate_paper_campaign(
    manifest_path: Path,
    *,
    reconcile_path: Path,
    acceptance_path: Path,
    gate_registry_path: Path,
    output_dir: Path,
    meta_path: Path | None = None,
) -> PaperAcceptanceReport:
    meta, tasks = load_manifest(manifest_path, meta_path=meta_path, verify_digest=True)
    if any(task.track != "paper" for task in tasks):
        raise CampaignError("E_PAPER_ACCEPTANCE_INPUT", "manifest must contain only paper tasks")
    diagnostic_profiles = sorted(
        {
            task.method_profile
            for task in tasks
            if task.method_profile.endswith(":diagnostic-dev")
            or task.method_profile == "paper:zhou2004-vote-diagnostic-v2"
        }
    )
    if diagnostic_profiles:
        raise CampaignError(
            "E_PAPER_ACCEPTANCE_DIAGNOSTIC",
            "diagnostic profile metrics are protocol-localization evidence only and "
            f"cannot enter paper acceptance: {diagnostic_profiles}",
        )
    reconcile = materialize_reconcile_paths(
        reconcile_path,
        _load_mapping(reconcile_path, label="reconcile report"),
    )
    if reconcile.get("campaign_id") != meta.get("campaign_id"):
        raise CampaignError("E_PAPER_ACCEPTANCE_INPUT", "reconcile campaign_id differs")
    if reconcile.get("manifest_sha256") != meta.get("manifest_sha256"):
        raise CampaignError("E_PAPER_ACCEPTANCE_INPUT", "reconcile manifest digest differs")
    raw_states = reconcile.get("tasks")
    if not isinstance(raw_states, list) or any(
        not isinstance(item, Mapping) for item in raw_states
    ):
        raise CampaignError("E_PAPER_ACCEPTANCE_INPUT", "reconcile tasks are missing")
    state_ids = [str(item.get("task_id")) for item in raw_states]
    if len(state_ids) != len(set(state_ids)):
        raise CampaignError("E_PAPER_ACCEPTANCE_INPUT", "reconcile contains duplicate task rows")
    states = dict(zip(state_ids, raw_states, strict=True))
    if set(states) != {task.task_id for task in tasks}:
        raise CampaignError("E_PAPER_ACCEPTANCE_INPUT", "reconcile task set differs")
    if reconcile.get("status") == "invalid":
        raise CampaignError("E_PAPER_ACCEPTANCE_INPUT", "invalid reconcile report is not claimable")

    cards = _load_acceptance_cards(acceptance_path)
    gate_registry = load_gate_registry(gate_registry_path)
    grouped: dict[str, list[CampaignTask]] = defaultdict(list)
    for task in tasks:
        if task.protocol_id is None:
            raise CampaignError("E_PAPER_ACCEPTANCE_INPUT", "paper task has no protocol_id")
        grouped[task.protocol_id].append(task)
    missing_cards = sorted(set(grouped) - set(cards))
    if missing_cards:
        raise CampaignError(
            "E_PAPER_ACCEPTANCE_SCHEMA", f"acceptance cards missing: {missing_cards}"
        )

    rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []
    informational_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    status_counts: dict[str, int] = defaultdict(int)
    for protocol_id, protocol_tasks in sorted(grouped.items()):
        card = cards[protocol_id]
        method_id = str(card["method_id"])
        if {task.method_id for task in protocol_tasks} != {method_id}:
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_INPUT", f"method differs for protocol {protocol_id}"
            )
        expected = int(card["repetitions"])
        declared_statuses = {task.fidelity_status for task in protocol_tasks}
        if len(declared_statuses) != 1 or next(iter(declared_statuses)) not in _FIDELITY_STATUSES:
            raise CampaignError(
                "E_PAPER_ACCEPTANCE_INPUT", f"fidelity status differs for {protocol_id}"
            )
        manifest_ceiling = str(next(iter(declared_statuses)))
        successful: list[tuple[CampaignTask, dict[str, Any]]] = []
        for task in sorted(protocol_tasks, key=lambda item: item.seed):
            state = states[task.task_id]
            result_dirs = state.get("result_dirs")
            paths = state.get("run_json_paths")
            digests = state.get("run_json_sha256")
            if state.get("status") != "success":
                continue
            if (
                not isinstance(result_dirs, list)
                or len(result_dirs) != 1
                or not isinstance(paths, list)
                or len(paths) != 1
                or not isinstance(digests, list)
                or len(digests) != 1
            ):
                raise CampaignError(
                    "E_PAPER_ACCEPTANCE_INPUT",
                    f"successful reconcile row is incomplete for {task.task_id}",
                )
            validated_path, payload, digest = validate_result_directory(
                Path(str(result_dirs[0])), task
            )
            if validated_path.resolve() != Path(str(paths[0])).resolve() or digest != digests[0]:
                raise CampaignError(
                    "E_PAPER_ACCEPTANCE_INPUT",
                    f"reconcile result identity differs for {task.task_id}",
                )
            successful.append((task, payload))
        complete = (
            len(protocol_tasks) == expected
            and len(successful) == expected
            and len({task.seed for task, _ in successful}) == expected
        )

        target = card["target"]
        primary = _summarize_target(successful, target=target, expected=expected)
        metric_available = bool(primary["available"])
        target_in_ci95 = bool(primary["target_in_ci95"])
        within_margin = bool(primary["within_margin"])
        secondary_target_summaries = [
            _summarize_target(successful, target=secondary, expected=expected)
            for secondary in card.get("secondary_targets", [])
        ]
        secondary_targets_ok = all(
            summary["available"] and summary["target_in_ci95"] and summary["within_margin"]
            for summary in secondary_target_summaries
        )
        informational_target_summaries = [
            _summarize_target(successful, target=informational, expected=expected)
            for informational in card.get("informational_targets", [])
        ]
        diagnostic_target_summaries = [
            _summarize_diagnostic_target(successful, target=diagnostic, expected=expected)
            for diagnostic in card.get("diagnostic_targets", [])
        ]
        diagnostic_targets_ok = all(
            summary["available"] and summary["within_margin"]
            for summary in diagnostic_target_summaries
        )

        diagnostic_failures: list[dict[str, Any]] = []
        for task, payload in successful:
            for rule in card.get("required_diagnostics", []):
                passed, actual = _rule_passes(payload, rule)
                if not passed:
                    diagnostic_failures.append(
                        {
                            "task_id": task.task_id,
                            "seed": task.seed,
                            "path": rule["path"],
                            "op": rule["op"],
                            "expected": rule.get("value"),
                            "actual": actual,
                        }
                    )
        diagnostics_ok = complete and not diagnostic_failures
        known_deviations = _string_list(
            card.get("known_deviations", []), field=f"{protocol_id}.known_deviations"
        )
        documented_equivalences = _string_list(
            card.get("documented_equivalences", []),
            field=f"{protocol_id}.documented_equivalences",
        )
        critical_unknowns = _string_list(
            card.get("critical_unknowns", []), field=f"{protocol_id}.critical_unknowns"
        )
        environment_differences = _string_list(
            card.get("environment_differences", []),
            field=f"{protocol_id}.environment_differences",
        )
        declared_protocol_conformity = card.get("protocol_conformity")
        protocol_conformity = (
            str(declared_protocol_conformity)
            if declared_protocol_conformity is not None
            else ("pending" if known_deviations or critical_unknowns else "passed")
        )
        gate_decision = evaluate_gate(
            gate_registry,
            campaign_id=str(meta["campaign_id"]),
            track="paper",
            method_id=method_id,
        )
        conformity_status = gate_registry.status(method_id)
        status = _status_for_protocol(
            complete=complete,
            metric_available=metric_available,
            manifest_ceiling=manifest_ceiling,
            critical_unknowns=critical_unknowns,
            known_deviations=known_deviations,
            target_in_ci95=target_in_ci95,
            within_margin=within_margin,
            secondary_targets_ok=secondary_targets_ok,
            diagnostic_targets_ok=diagnostic_targets_ok,
            diagnostics_ok=diagnostics_ok,
            conformity_status="passed" if gate_decision.allowed else "blocked",
            protocol_conformity=protocol_conformity,
        )
        result_status = _result_status(
            complete=complete,
            metric_available=metric_available,
            target_in_ci95=target_in_ci95,
            within_margin=within_margin,
        )
        reasons: list[str] = []
        if not complete:
            reasons.append("repetitions_incomplete")
        if not metric_available:
            reasons.append("target_metric_missing")
        if critical_unknowns:
            reasons.append("critical_protocol_unknowns")
        if manifest_ceiling != "paper_matched":
            reasons.append(f"manifest_ceiling={manifest_ceiling}")
        if known_deviations:
            reasons.append("known_protocol_deviations")
        if metric_available and not target_in_ci95:
            reasons.append("published_target_outside_replication_ci95")
        if metric_available and not within_margin:
            reasons.append("absolute_margin_exceeded")
        if any(not summary["available"] for summary in secondary_target_summaries):
            reasons.append("secondary_target_missing")
        if any(
            summary["available"] and not summary["target_in_ci95"]
            for summary in secondary_target_summaries
        ):
            reasons.append("secondary_target_outside_replication_ci95")
        if any(
            summary["available"] and not summary["within_margin"]
            for summary in secondary_target_summaries
        ):
            reasons.append("secondary_target_margin_exceeded")
        if any(not summary["available"] for summary in diagnostic_target_summaries):
            reasons.append("protocol_diagnostic_target_missing")
        if any(
            summary["available"] and not summary["within_margin"]
            for summary in diagnostic_target_summaries
        ):
            reasons.append("protocol_diagnostic_target_margin_exceeded")
        if not diagnostics_ok:
            reasons.append("secondary_diagnostics_failed")
        if protocol_conformity != "passed":
            reasons.append(f"protocol_conformity={protocol_conformity}")
        if not gate_decision.allowed:
            reasons.extend(f"scientific_gate={blocker}" for blocker in gate_decision.blockers)
        status_counts[status] += 1
        row = {
            "protocol_id": protocol_id,
            "method_id": method_id,
            "status": status,
            "protocol_status": status,
            "result_status": result_status,
            "equation_conformity": conformity_status,
            "protocol_conformity": protocol_conformity,
            "n_expected": expected,
            "n_success": len(successful),
            "metric": primary["metric"],
            "replication_mean": primary["replication_mean"],
            "replication_std": primary["replication_std"],
            "ci95_low": primary["ci95_low"],
            "ci95_high": primary["ci95_high"],
            "published_mean": primary["published_mean"],
            "published_std": primary["published_std"],
            "published_std_ddof": primary["published_std_ddof"],
            "std_absolute_difference": primary["std_absolute_difference"],
            "absolute_difference": primary["absolute_difference"],
            "margin_absolute": primary["margin_absolute"],
            "target_in_ci95": target_in_ci95,
            "within_margin": within_margin,
            "secondary_targets_ok": secondary_targets_ok,
            "diagnostic_targets_ok": diagnostic_targets_ok,
            "diagnostics_ok": diagnostics_ok,
            "algorithmic_conformity": conformity_status,
            "scientific_gate_allowed": gate_decision.allowed,
            "scientific_gate_blockers": ";".join(gate_decision.blockers),
            "reasons": ";".join(reasons),
        }
        rows.append(row)
        secondary_rows.extend(
            {
                "protocol_id": protocol_id,
                "method_id": method_id,
                "protocol_status": status,
                **summary,
            }
            for summary in secondary_target_summaries
        )
        informational_rows.extend(
            {
                "protocol_id": protocol_id,
                "method_id": method_id,
                "protocol_status": status,
                **summary,
            }
            for summary in informational_target_summaries
        )
        details.append(
            {
                **row,
                "manifest_fidelity_ceiling": manifest_ceiling,
                "known_deviations": known_deviations,
                "documented_equivalences": documented_equivalences,
                "critical_unknowns": critical_unknowns,
                "environment_differences": environment_differences,
                "diagnostic_failures": diagnostic_failures,
                "secondary_targets": secondary_target_summaries,
                "informational_targets": informational_target_summaries,
                "diagnostic_targets": diagnostic_target_summaries,
                "control_targets": card.get("control_targets", []),
                "confidence_candidates": card.get("confidence_candidates", []),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "paper-acceptance.csv"
    _write_matrix(matrix_path, rows)
    secondary_matrix_path = output_dir / "paper-secondary-targets.csv"
    _write_secondary_matrix(secondary_matrix_path, secondary_rows)
    informational_matrix_path = output_dir / "paper-informational-targets.csv"
    _write_secondary_matrix(informational_matrix_path, informational_rows)
    report_path = output_dir / "paper-acceptance.json"
    atomic_write_json(
        report_path,
        {
            "schema_version": 1,
            "campaign_id": meta["campaign_id"],
            "manifest_sha256": meta["manifest_sha256"],
            "evaluated_at": datetime.now(UTC).isoformat(),
            "acceptance_registry": str(acceptance_path.resolve()),
            "acceptance_registry_sha256": sha256_file(acceptance_path),
            "scientific_gate_registry": str(gate_registry_path.resolve()),
            "scientific_gate_registry_sha256": sha256_file(gate_registry_path),
            "status_counts": dict(sorted(status_counts.items())),
            "informational_targets_path": str(informational_matrix_path),
            "protocols": details,
        },
    )
    return PaperAcceptanceReport(
        campaign_id=str(meta["campaign_id"]),
        status_counts=dict(sorted(status_counts.items())),
        protocol_count=len(details),
        report_path=str(report_path),
        matrix_path=str(matrix_path),
        secondary_matrix_path=str(secondary_matrix_path),
        informational_matrix_path=str(informational_matrix_path),
    )
