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

from bench.utils.io import atomic_write_json

from .dcl_partition_lock import (
    DCL_DATASET_ID,
    DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS,
    DCL_DIAGNOSTIC_CONTROL_PROTOCOLS,
    DCL_DIAGNOSTIC_METHOD_PROFILE,
    DCL_METHOD_ID,
    DCL_PAPER_PROTOCOL_ID,
)
from .errors import CampaignError
from .executor import validate_result_directory
from .manifest import load_manifest, sha256_file
from .models import CampaignTask
from .paper_acceptance import (
    _diagnostic_value,
    _load_acceptance_cards,
    _nested,
    _summarize_values,
)
from .reconcile import materialize_reconcile_paths

_EXPECTED_REPETITIONS = 20
_PRIMARY_CONFIDENCE_PROTOCOLS = {
    protocol_id
    for protocol_id, settings in DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS.items()
    if settings in {("training_accuracy", "wald"), ("kfold_oof", "wald")}
}
_CONDITIONAL_CONFIDENCE_PROTOCOLS = (
    set(DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS) - _PRIMARY_CONFIDENCE_PROTOCOLS
)
_ROUND_TRACE_FIELDS = {"round", "majority_eligible_count", "learners"}
_LEARNER_TRACE_FIELDS = {
    "learner_index",
    "classifier_id",
    "original_interval",
    "weight",
    "evolving_interval",
    "training_size_before",
    "training_size_after",
    "disagreement_count",
    "proposal_count",
    "error_estimate_before",
    "proposal_error",
    "error_estimate_after",
    "q",
    "q_prime",
    "accepted",
    "added_count",
}
_INTERVAL_FIELDS = {"lower", "upper"}
_EXPECTED_LEARNER_IDS = ("gaussian_nb", "decision_tree", "knn")
_EXPECTED_CONTROL_MODES = ("learner_0", "learner_1", "learner_2", "combining_only")
_EXPECTED_CONTROL_CONFIDENCE_PROTOCOL = {
    "estimator": "training_accuracy",
    "interval": "wald",
    "folds": 10,
    "seed": 0,
}
_INITIAL_LABELED_COUNT = 40
_UNLABELED_COUNT = 200


@dataclass(frozen=True)
class DCLDiagnosticReport:
    campaign_id: str
    diagnostic_kind: str
    status: str
    gate_statuses: dict[str, str]
    report_path: str
    matrix_path: str


def _invalid(message: str) -> CampaignError:
    return CampaignError("E_DCL_DIAGNOSTIC_INPUT", message)


def _load_reconcile(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _invalid(f"cannot read reconcile report: {path}") from exc
    if not isinstance(payload, dict):
        raise _invalid("reconcile report must be a mapping")
    return materialize_reconcile_paths(path, payload)


def _validate_manifest_identity(tasks: list[CampaignTask]) -> tuple[str, set[str]]:
    if not tasks:
        raise _invalid("diagnostic manifest is empty")
    if any(
        task.track != "paper"
        or task.method_id != DCL_METHOD_ID
        or task.method_profile != DCL_DIAGNOSTIC_METHOD_PROFILE
        or task.dataset_id != DCL_DATASET_ID
        or task.fidelity_status != "not_claimable"
        or task.partition_selection is None
        for task in tasks
    ):
        raise _invalid("manifest contains a non-DCL-v2 diagnostic task")
    protocols = {str(task.protocol_id) for task in tasks}
    if protocols == set(DCL_DIAGNOSTIC_CONTROL_PROTOCOLS):
        return "controls", protocols
    if protocols == _PRIMARY_CONFIDENCE_PROTOCOLS:
        return "confidence_primary", protocols
    if protocols == _CONDITIONAL_CONFIDENCE_PROTOCOLS:
        return "confidence_conditional", protocols
    raise _invalid(
        "manifest must contain exactly the four controls, the two primary "
        "confidence candidates, or the two conditional candidates"
    )


def _collect_results(
    tasks: list[CampaignTask],
    *,
    reconcile: Mapping[str, Any],
    campaign_id: str,
    manifest_sha256: str,
) -> dict[str, list[tuple[CampaignTask, dict[str, Any]]]]:
    if reconcile.get("campaign_id") != campaign_id:
        raise _invalid("reconcile campaign_id differs")
    if reconcile.get("manifest_sha256") != manifest_sha256:
        raise _invalid("reconcile manifest digest differs")
    raw_states = reconcile.get("tasks")
    if not isinstance(raw_states, list) or any(
        not isinstance(state, Mapping) for state in raw_states
    ):
        raise _invalid("reconcile tasks are missing")
    state_ids = [str(state.get("task_id")) for state in raw_states]
    if len(state_ids) != len(set(state_ids)):
        raise _invalid("reconcile contains duplicate task rows")
    state_by_id = dict(zip(state_ids, raw_states, strict=True))
    if set(state_by_id) != {task.task_id for task in tasks}:
        raise _invalid("reconcile task set differs from the manifest")
    if reconcile.get("status") == "invalid":
        raise _invalid("invalid reconcile report cannot enter DCL diagnostics")

    grouped_tasks: dict[str, list[CampaignTask]] = defaultdict(list)
    for task in tasks:
        assert task.protocol_id is not None
        grouped_tasks[task.protocol_id].append(task)
    results: dict[str, list[tuple[CampaignTask, dict[str, Any]]]] = {}
    for protocol_id, protocol_tasks in grouped_tasks.items():
        seeds = [task.seed for task in protocol_tasks]
        ranks = [
            int(task.partition_selection["selection_rank"])
            for task in protocol_tasks
            if task.partition_selection is not None
        ]
        if (
            len(protocol_tasks) != _EXPECTED_REPETITIONS
            or len(seeds) != len(set(seeds))
            or set(seeds) != set(range(1, _EXPECTED_REPETITIONS + 1))
            or set(ranks) != set(range(1, _EXPECTED_REPETITIONS + 1))
        ):
            raise _invalid(f"protocol {protocol_id} does not contain the exact 20 locked cells")
        successful: list[tuple[CampaignTask, dict[str, Any]]] = []
        for task in sorted(protocol_tasks, key=lambda value: value.seed):
            state = state_by_id[task.task_id]
            if state.get("status") != "success":
                continue
            result_dirs = state.get("result_dirs")
            run_paths = state.get("run_json_paths")
            run_digests = state.get("run_json_sha256")
            if (
                not isinstance(result_dirs, list)
                or len(result_dirs) != 1
                or not isinstance(run_paths, list)
                or len(run_paths) != 1
                or not isinstance(run_digests, list)
                or len(run_digests) != 1
            ):
                raise _invalid(f"successful reconcile row is incomplete for {task.task_id}")
            run_path, payload, digest = validate_result_directory(Path(result_dirs[0]), task)
            if run_path.resolve() != Path(run_paths[0]).resolve() or digest != run_digests[0]:
                raise _invalid(f"reconcile result identity differs for {task.task_id}")
            successful.append((task, payload))
        results[protocol_id] = successful
    return results


def _cell_completeness(successful: list[tuple[CampaignTask, dict[str, Any]]]) -> bool:
    return (
        len(successful) == _EXPECTED_REPETITIONS
        and len({task.seed for task, _ in successful}) == _EXPECTED_REPETITIONS
    )


def _gate_status(statuses: list[str]) -> str:
    if any(status == "incomplete" for status in statuses):
        return "incomplete"
    if any(status == "failed" for status in statuses):
        return "failed"
    return "passed"


def _trace_int(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return int(value)


def _trace_float(value: Any, *, minimum: float = 0.0) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        return None
    return number


def _trace_interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Mapping) or set(value) != _INTERVAL_FIELDS:
        return None
    lower = _trace_float(value.get("lower"))
    upper = _trace_float(value.get("upper"))
    if lower is None or upper is None or upper > 1.0 or lower > upper:
        return None
    return lower, upper


def _isolated_control_failures(
    payload: Mapping[str, Any],
    *,
    expected_mode: str,
) -> list[str]:
    exists, diagnostics = _nested(payload, "artifacts.method.diagnostics")
    if not exists or not isinstance(diagnostics, Mapping):
        return ["diagnostics must be a mapping"]
    failures: list[str] = []
    additions = diagnostics.get("pseudo_labels_added_per_learner")
    if (
        _trace_int(diagnostics.get("n_iter")) != 0
        or _trace_int(diagnostics.get("changed_rounds")) != 0
        or diagnostics.get("converged") is not True
        or not isinstance(additions, list)
        or len(additions) != 3
        or any(_trace_int(value) != 0 for value in additions)
        or _trace_int(diagnostics.get("pseudo_labels_added_total")) != 0
        or diagnostics.get("round_trace") != []
    ):
        failures.append("isolated control executed or reported a DCL update")

    control = diagnostics.get("control")
    if (
        not isinstance(control, Mapping)
        or control.get("mode") != expected_mode
        or control.get("available_modes") != list(_EXPECTED_CONTROL_MODES)
        or control.get("learner_ids") != list(_EXPECTED_LEARNER_IDS)
    ):
        failures.append("control learner metadata differ from the locked protocol")
    if diagnostics.get("confidence_protocol") != _EXPECTED_CONTROL_CONFIDENCE_PROTOCOL:
        failures.append("control confidence protocol differs from the locked protocol")
    return failures


def _round_trace_failures(payload: Mapping[str, Any]) -> list[str]:
    exists, diagnostics = _nested(payload, "artifacts.method.diagnostics")
    if not exists or not isinstance(diagnostics, Mapping):
        return ["diagnostics must be a mapping"]
    n_iter = _trace_int(diagnostics.get("n_iter"), minimum=1)
    changed_rounds = _trace_int(diagnostics.get("changed_rounds"))
    converged = diagnostics.get("converged")
    trace = diagnostics.get("round_trace")
    additions = diagnostics.get("pseudo_labels_added_per_learner")
    total_added = _trace_int(diagnostics.get("pseudo_labels_added_total"))
    if (
        n_iter is None
        or changed_rounds is None
        or converged is not True
        or not isinstance(trace, list)
        or len(trace) != n_iter
        or not isinstance(additions, list)
        or len(additions) != 3
        or any(_trace_int(value) is None or int(value) > _UNLABELED_COUNT for value in additions)
        or total_added is None
        or total_added != sum(int(value) for value in additions)
    ):
        return ["top-level trace diagnostics are inconsistent"]

    failures: list[str] = []
    previous_sizes = dict.fromkeys(range(3), _INITIAL_LABELED_COUNT)
    previous_errors = {0: 0.0, 1: 0.0, 2: 0.0}
    learner_ids: dict[int, str] = {}
    traced_additions = {0: 0, 1: 0, 2: 0}
    traced_changed_rounds = 0
    for expected_round, raw_round in enumerate(trace, start=1):
        prefix = f"round_trace[{expected_round - 1}]"
        if not isinstance(raw_round, Mapping) or set(raw_round) != _ROUND_TRACE_FIELDS:
            failures.append(f"{prefix}: fields differ")
            continue
        round_number = _trace_int(raw_round.get("round"), minimum=1)
        majority_count = _trace_int(raw_round.get("majority_eligible_count"))
        learners = raw_round.get("learners")
        if round_number != expected_round:
            failures.append(f"{prefix}: round is not sequential")
        if majority_count is None or majority_count > _UNLABELED_COUNT:
            failures.append(f"{prefix}: majority_eligible_count is invalid")
        if not isinstance(learners, list) or len(learners) != 3:
            failures.append(f"{prefix}: exactly three learners are required")
            continue
        raw_indices = [
            learner.get("learner_index") if isinstance(learner, Mapping) else None
            for learner in learners
        ]
        if raw_indices != [0, 1, 2]:
            failures.append(f"{prefix}: learner indices must be ordered uniquely 0..2")
            continue

        round_changed = False
        current_ids: set[str] = set()
        for raw_learner in learners:
            assert isinstance(raw_learner, Mapping)
            learner_index = int(raw_learner["learner_index"])
            learner_prefix = f"{prefix}.learners[{learner_index}]"
            if set(raw_learner) != _LEARNER_TRACE_FIELDS:
                failures.append(f"{learner_prefix}: fields differ")
                continue
            classifier_id = raw_learner.get("classifier_id")
            if not isinstance(classifier_id, str) or not classifier_id:
                failures.append(f"{learner_prefix}: classifier_id is invalid")
                continue
            if classifier_id != _EXPECTED_LEARNER_IDS[learner_index]:
                failures.append(
                    f"{learner_prefix}: classifier_id does not match the locked learner"
                )
            current_ids.add(classifier_id)
            prior_id = learner_ids.setdefault(learner_index, classifier_id)
            if classifier_id != prior_id:
                failures.append(f"{learner_prefix}: classifier_id changed between rounds")

            original_interval = _trace_interval(raw_learner.get("original_interval"))
            evolving_interval = _trace_interval(raw_learner.get("evolving_interval"))
            weight = _trace_float(raw_learner.get("weight"))
            if original_interval is None or evolving_interval is None or weight is None:
                failures.append(f"{learner_prefix}: confidence interval or weight is invalid")
            elif not math.isclose(
                weight,
                (original_interval[0] + original_interval[1]) / 2.0,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                failures.append(f"{learner_prefix}: weight is not the interval midpoint")

            size_before = _trace_int(raw_learner.get("training_size_before"), minimum=1)
            size_after = _trace_int(raw_learner.get("training_size_after"), minimum=1)
            disagreement_count = _trace_int(raw_learner.get("disagreement_count"))
            proposal_count = _trace_int(raw_learner.get("proposal_count"))
            added_count = _trace_int(raw_learner.get("added_count"))
            accepted = raw_learner.get("accepted")
            error_before = _trace_float(raw_learner.get("error_estimate_before"))
            proposal_error = _trace_float(raw_learner.get("proposal_error"))
            error_after = _trace_float(raw_learner.get("error_estimate_after"))
            q_value = _trace_float(raw_learner.get("q"))
            q_prime = _trace_float(raw_learner.get("q_prime"))
            numeric_values = (
                size_before,
                size_after,
                disagreement_count,
                proposal_count,
                added_count,
                error_before,
                proposal_error,
                error_after,
                q_value,
                q_prime,
                majority_count,
            )
            if any(value is None for value in numeric_values) or not isinstance(accepted, bool):
                failures.append(f"{learner_prefix}: numeric or boolean field is invalid")
                continue
            assert (
                size_before is not None
                and size_after is not None
                and disagreement_count is not None
                and proposal_count is not None
                and added_count is not None
                and error_before is not None
                and proposal_error is not None
                and error_after is not None
                and q_value is not None
                and q_prime is not None
                and majority_count is not None
            )
            if size_before != previous_sizes[learner_index] or not math.isclose(
                error_before,
                previous_errors[learner_index],
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                failures.append(f"{learner_prefix}: size or error continuity failed")
            if (
                disagreement_count < proposal_count
                or disagreement_count > _UNLABELED_COUNT
                or proposal_count > majority_count
                or proposal_error > proposal_count
                or error_before > size_before
                or size_after > _INITIAL_LABELED_COUNT + _UNLABELED_COUNT
            ):
                failures.append(f"{learner_prefix}: proposal/error bounds failed")

            expected_q = size_before * (1.0 - 2.0 * error_before / size_before) ** 2
            candidate_size = size_before + proposal_count
            expected_q_prime = (
                candidate_size * (1.0 - 2.0 * (error_before + proposal_error) / candidate_size) ** 2
            )
            if not math.isclose(q_value, expected_q, rel_tol=1e-9, abs_tol=1e-9):
                failures.append(f"{learner_prefix}: q is inconsistent")
            if not math.isclose(
                q_prime,
                expected_q_prime,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                failures.append(f"{learner_prefix}: q_prime is inconsistent")

            expected_accepted = proposal_count > 0 and q_prime > q_value
            expected_added = proposal_count if expected_accepted else 0
            expected_error_after = (
                error_before + proposal_error if expected_accepted else error_before
            )
            if accepted != expected_accepted:
                failures.append(f"{learner_prefix}: accepted is inconsistent with q/q_prime")
            if (
                added_count != expected_added
                or size_after - size_before != added_count
                or not math.isclose(
                    error_after,
                    expected_error_after,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or error_after > size_after
            ):
                failures.append(f"{learner_prefix}: added_count/size/error-after failed")
            previous_sizes[learner_index] = size_after
            previous_errors[learner_index] = error_after
            traced_additions[learner_index] += added_count
            round_changed = round_changed or accepted
        if len(current_ids) != 3:
            failures.append(f"{prefix}: classifier ids must be unique")
        if round_changed:
            traced_changed_rounds += 1
        if expected_round < n_iter and not round_changed:
            failures.append(f"{prefix}: convergence occurred before the terminal pass")
        if expected_round == n_iter and round_changed:
            failures.append(f"{prefix}: terminal convergence pass changed a learner")

    if changed_rounds != traced_changed_rounds or changed_rounds != n_iter - 1:
        failures.append("changed_rounds does not match the traced terminal-pass convention")
    if [traced_additions[index] for index in range(3)] != [int(value) for value in additions]:
        failures.append("traced additions differ from pseudo_labels_added_per_learner")
    if [previous_sizes[index] for index in range(3)] != [
        _INITIAL_LABELED_COUNT + int(value) for value in additions
    ]:
        failures.append("final training sizes differ from the traced additions")
    return failures


def _write_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "protocol_id",
        "diagnostic_kind",
        "candidate",
        "cell_status",
        "integrity_status",
        "numerical_equivalence_status",
        "complete",
        "target_id",
        "metric",
        "n",
        "replication_mean",
        "replication_std",
        "published_mean",
        "absolute_difference",
        "margin_absolute",
        "within_margin",
        "nb_receives_most",
        "test_information_used",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _control_evaluation(
    results: Mapping[str, list[tuple[CampaignTask, dict[str, Any]]]],
    *,
    control_targets: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    target_by_mode = {str(target["control_mode"]): target for target in control_targets}
    protocols: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    integrity_statuses: list[str] = []
    numerical_equivalence_statuses: list[str] = []
    for protocol_id, control_mode in sorted(DCL_DIAGNOSTIC_CONTROL_PROTOCOLS.items()):
        successful = results[protocol_id]
        complete = _cell_completeness(successful)
        target = target_by_mode[control_mode]
        values: list[float] = []
        protocol_failures: list[dict[str, Any]] = []
        for task, payload in successful:
            exists, actual_mode = _nested(payload, "artifacts.method.diagnostics.control.mode")
            control_failures = _isolated_control_failures(
                payload,
                expected_mode=control_mode,
            )
            if not exists or actual_mode != control_mode or control_failures:
                protocol_failures.append(
                    {
                        "task_id": task.task_id,
                        "seed": task.seed,
                        "expected_control_mode": control_mode,
                        "actual_control_mode": actual_mode,
                        "control_diagnostic_failures": control_failures,
                    }
                )
            metrics = payload.get("metrics")
            test_metrics = metrics.get("test") if isinstance(metrics, Mapping) else None
            value = test_metrics.get("accuracy") if isinstance(test_metrics, Mapping) else None
            if (
                not isinstance(value, bool)
                and isinstance(value, int | float)
                and math.isfinite(float(value))
            ):
                values.append(float(value))
        summary = _summarize_values(
            values,
            target=target,
            expected=_EXPECTED_REPETITIONS,
            metric="test.accuracy:identity",
        )
        if not complete:
            integrity_status = "incomplete"
        elif protocol_failures:
            integrity_status = "failed"
        else:
            integrity_status = "passed"
        if not complete or not summary["available"]:
            numerical_equivalence_status = "incomplete"
        elif summary["within_margin"]:
            numerical_equivalence_status = "passed"
        else:
            numerical_equivalence_status = "failed"
        if integrity_status == "passed" and numerical_equivalence_status == "passed":
            cell_status = "passed"
        elif "incomplete" in {
            integrity_status,
            numerical_equivalence_status,
        }:
            cell_status = "incomplete"
        else:
            cell_status = "failed"
        integrity_statuses.append(integrity_status)
        numerical_equivalence_statuses.append(numerical_equivalence_status)
        protocols.append(
            {
                "protocol_id": protocol_id,
                "control_mode": control_mode,
                "complete": complete,
                "n_success": len(successful),
                "status": cell_status,
                "integrity_status": integrity_status,
                "numerical_equivalence_status": numerical_equivalence_status,
                "target": summary,
                "protocol_failures": protocol_failures,
                "paper_claim_allowed": False,
            }
        )
        matrix_rows.append(
            {
                "protocol_id": protocol_id,
                "diagnostic_kind": "controls",
                "candidate": control_mode,
                "cell_status": cell_status,
                "integrity_status": integrity_status,
                "numerical_equivalence_status": numerical_equivalence_status,
                "complete": complete,
                "target_id": summary["id"],
                **summary,
                "test_information_used": True,
            }
        )
    integrity_status = _gate_status(integrity_statuses)
    numerical_equivalence_status = _gate_status(numerical_equivalence_statuses)
    gates = {
        "control_integrity": {
            "status": integrity_status,
            "required_protocols": sorted(DCL_DIAGNOSTIC_CONTROL_PROTOCOLS),
            "failed_protocols": [
                protocol["protocol_id"]
                for protocol in protocols
                if protocol["integrity_status"] == "failed"
            ],
            "incomplete_protocols": [
                protocol["protocol_id"]
                for protocol in protocols
                if protocol["integrity_status"] == "incomplete"
            ],
        },
        "numerical_equivalence": {
            "status": numerical_equivalence_status,
            "required_protocols": sorted(DCL_DIAGNOSTIC_CONTROL_PROTOCOLS),
            "failed_protocols": [
                protocol["protocol_id"]
                for protocol in protocols
                if protocol["numerical_equivalence_status"] == "failed"
            ],
            "incomplete_protocols": [
                protocol["protocol_id"]
                for protocol in protocols
                if protocol["numerical_equivalence_status"] == "incomplete"
            ],
        },
        "confidence": {"status": "not_applicable"},
        "dynamics": {"status": "not_applicable"},
    }
    return protocols, matrix_rows, gates


def _confidence_evaluation(
    results: Mapping[str, list[tuple[CampaignTask, dict[str, Any]]]],
    *,
    diagnostic_targets: list[Mapping[str, Any]],
    confidence_candidates: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidate_by_settings = {
        (str(candidate["estimator"]), str(candidate["interval"])): candidate
        for candidate in confidence_candidates
    }
    protocols: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    confidence_statuses: list[str] = []
    dynamics_statuses: list[str] = []
    eligible_protocols: list[str] = []
    for protocol_id in sorted(results):
        expected_estimator, expected_interval = DCL_DIAGNOSTIC_CONFIDENCE_PROTOCOLS[protocol_id]
        candidate = candidate_by_settings[(expected_estimator, expected_interval)]
        successful = results[protocol_id]
        complete = _cell_completeness(successful)
        protocol_failures: list[dict[str, Any]] = []
        for task, payload in successful:
            metrics = payload.get("metrics")
            if isinstance(metrics, Mapping) and any(
                str(split) == "test" or str(split).startswith("test_") for split in metrics
            ):
                raise CampaignError(
                    "E_DCL_DIAGNOSTIC_TEST_LEAK",
                    f"confidence task {task.task_id} contains held-out test metrics",
                )
            exists, actual_protocol = _nested(
                payload,
                "artifacts.method.diagnostics.confidence_protocol",
            )
            expected_protocol = {
                "estimator": expected_estimator,
                "interval": expected_interval,
                "folds": 10,
                "seed": 0,
            }
            mode_exists, actual_mode = _nested(
                payload,
                "artifacts.method.diagnostics.control.mode",
            )
            trace_failures = _round_trace_failures(payload)
            if (
                not exists
                or actual_protocol != expected_protocol
                or not mode_exists
                or actual_mode != "dcl"
                or trace_failures
            ):
                protocol_failures.append(
                    {
                        "task_id": task.task_id,
                        "seed": task.seed,
                        "expected_confidence_protocol": expected_protocol,
                        "actual_confidence_protocol": actual_protocol,
                        "actual_control_mode": actual_mode,
                        "round_trace_failures": trace_failures,
                    }
                )
        target_summaries: list[dict[str, Any]] = []
        for target in diagnostic_targets:
            values = [
                value
                for _, payload in successful
                if (value := _diagnostic_value(payload, target)) is not None
            ]
            target_summaries.append(
                _summarize_values(
                    values,
                    target=target,
                    expected=_EXPECTED_REPETITIONS,
                    metric=f"diagnostic:{target['path']}",
                )
            )
        additions: list[list[float]] = [[], [], []]
        for _, payload in successful:
            for learner_index in range(3):
                exists, value = _nested(
                    payload,
                    (
                        "artifacts.method.diagnostics."
                        f"pseudo_labels_added_per_learner.{learner_index}"
                    ),
                )
                if (
                    exists
                    and not isinstance(value, bool)
                    and isinstance(value, int | float)
                    and math.isfinite(float(value))
                ):
                    additions[learner_index].append(float(value))
        additions_means = [
            sum(values) / len(values) if len(values) == _EXPECTED_REPETITIONS else None
            for values in additions
        ]
        nb_receives_most = bool(
            all(value is not None for value in additions_means)
            and additions_means[0] > additions_means[1]
            and additions_means[0] > additions_means[2]
        )
        if not complete:
            confidence_status = dynamics_status = "incomplete"
        else:
            confidence_status = "failed" if protocol_failures else "passed"
            if protocol_failures:
                dynamics_status = "failed"
            elif any(not summary["available"] for summary in target_summaries):
                dynamics_status = "incomplete"
            elif (
                any(not summary["within_margin"] for summary in target_summaries)
                or not nb_receives_most
            ):
                dynamics_status = "failed"
            else:
                dynamics_status = "passed"
                eligible_protocols.append(protocol_id)
        confidence_statuses.append(confidence_status)
        dynamics_statuses.append(dynamics_status)
        cell_status = (
            "passed"
            if confidence_status == "passed" and dynamics_status == "passed"
            else (
                "incomplete" if "incomplete" in {confidence_status, dynamics_status} else "failed"
            )
        )
        protocols.append(
            {
                "protocol_id": protocol_id,
                "candidate_id": candidate["id"],
                "candidate_role": candidate["role"],
                "protocol_conformity": "pending",
                "complete": complete,
                "n_success": len(successful),
                "status": cell_status,
                "integrity_status": "not_applicable",
                "numerical_equivalence_status": "not_applicable",
                "confidence_status": confidence_status,
                "dynamics_status": dynamics_status,
                "targets": target_summaries,
                "pseudo_labels_added_mean_per_learner": additions_means,
                "nb_receives_most": nb_receives_most,
                "test_information_used": False,
                "protocol_failures": protocol_failures,
                "paper_claim_allowed": False,
            }
        )
        for summary in target_summaries:
            matrix_rows.append(
                {
                    "protocol_id": protocol_id,
                    "diagnostic_kind": "confidence",
                    "candidate": candidate["id"],
                    "cell_status": cell_status,
                    "integrity_status": "not_applicable",
                    "numerical_equivalence_status": "not_applicable",
                    "complete": complete,
                    "target_id": summary["id"],
                    **summary,
                    "nb_receives_most": nb_receives_most,
                    "test_information_used": False,
                }
            )
    confidence_gate_status = _gate_status(confidence_statuses)
    if any(status == "incomplete" for status in dynamics_statuses):
        dynamics_gate_status = "incomplete"
    elif eligible_protocols:
        dynamics_gate_status = "passed"
    else:
        dynamics_gate_status = "failed"
    gates = {
        "control_integrity": {"status": "not_applicable"},
        "numerical_equivalence": {"status": "not_applicable"},
        "confidence": {
            "status": confidence_gate_status,
            "failed_protocols": [
                protocol["protocol_id"]
                for protocol in protocols
                if protocol["confidence_status"] != "passed"
            ],
            "test_information_used": False,
        },
        "dynamics": {
            "status": dynamics_gate_status,
            "eligible_protocols": eligible_protocols,
            "selection_basis": "table2_trajectory_only",
            "requires_nb_most_pseudo_labels": True,
            "test_information_used": False,
        },
    }
    return protocols, matrix_rows, gates


def evaluate_dcl_diagnostics(
    manifest_path: Path,
    *,
    reconcile_path: Path,
    acceptance_path: Path,
    output_dir: Path,
    meta_path: Path | None = None,
) -> DCLDiagnosticReport:
    meta, tasks = load_manifest(manifest_path, meta_path=meta_path, verify_digest=True)
    diagnostic_kind, _protocol_ids = _validate_manifest_identity(tasks)
    reconcile = _load_reconcile(reconcile_path)
    results = _collect_results(
        tasks,
        reconcile=reconcile,
        campaign_id=str(meta["campaign_id"]),
        manifest_sha256=str(meta["manifest_sha256"]),
    )
    cards = _load_acceptance_cards(acceptance_path)
    card = cards[DCL_PAPER_PROTOCOL_ID]
    if diagnostic_kind == "controls":
        protocols, matrix_rows, gates = _control_evaluation(
            results,
            control_targets=card["control_targets"],
        )
        required_gates = ("control_integrity", "numerical_equivalence")
    else:
        protocols, matrix_rows, gates = _confidence_evaluation(
            results,
            diagnostic_targets=card["diagnostic_targets"],
            confidence_candidates=card["confidence_candidates"],
        )
        required_gates = ("confidence", "dynamics")
    status = (
        "passed" if all(gates[gate]["status"] == "passed" for gate in required_gates) else "blocked"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "dcl-diagnostic-targets.csv"
    _write_matrix(matrix_path, matrix_rows)
    report_path = output_dir / "dcl-diagnostic-gates.json"
    atomic_write_json(
        report_path,
        {
            "schema_version": 2,
            "campaign_id": meta["campaign_id"],
            "diagnostic_kind": diagnostic_kind,
            "status": status,
            "paper_claim_allowed": False,
            "protocol_conformity": card["protocol_conformity"],
            "manifest_sha256": meta["manifest_sha256"],
            "acceptance_registry": str(acceptance_path.resolve()),
            "acceptance_registry_sha256": sha256_file(acceptance_path),
            "evaluated_at": datetime.now(UTC).isoformat(),
            "gates": gates,
            "protocols": protocols,
        },
    )
    return DCLDiagnosticReport(
        campaign_id=str(meta["campaign_id"]),
        diagnostic_kind=diagnostic_kind,
        status=status,
        gate_statuses={key: str(value["status"]) for key, value in gates.items()},
        report_path=str(report_path),
        matrix_path=str(matrix_path),
    )
