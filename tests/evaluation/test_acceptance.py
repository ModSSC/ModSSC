from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

import pytest

from modssc.evaluation import (
    AcceptanceSpec,
    AcceptanceSpecError,
    evaluate_acceptance,
    parse_acceptance_spec,
)


def _spec(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "protocol_id": "protocol-arbitrary-v1",
        "method_id": "method-with-no-runtime-branch",
        "repetitions": 2,
        "fidelity_ceiling": "paper_matched",
        "conformity": {
            "status": "passed",
            "basis": "independent-equation-oracle",
            "evidence": ["evidence://oracle", "tests/test_oracle.py"],
            "review": {
                "reviewed_by": "reviewer:scientific",
                "reviewed_at": "2026-08-29T12:00:00+02:00",
            },
        },
        "target": {
            "split": "test",
            "metric": "error",
            "transform": "one_minus",
            "published_mean": 0.5,
            "published_std": 0.1,
            "published_std_ddof": 0,
            "margin_absolute": 0.01,
        },
        "secondary_targets": [],
        "informational_targets": [],
        "diagnostic_targets": [],
        "required_diagnostics": [],
        "deviations": [],
        "equivalences": ["same decision rule"],
        "unknowns": [],
    }
    value.update(updates)
    return value


def _run(
    seed: int,
    error: float,
    *,
    status: str = "success",
    diagnostics: dict[str, Any] | None = None,
    method_id: str = "method-with-no-runtime-branch",
) -> dict[str, Any]:
    return {
        "run": {"seed": seed, "status": status, "run_id": f"run-{seed}"},
        "config": {"method": {"id": method_id}},
        "metrics": {"test": {"error": error}},
        "diagnostics": diagnostics or {},
    }


def test_parse_acceptance_spec_normalizes_declarative_collections() -> None:
    raw = _spec(
        deviations=["z", "a"],
        required_diagnostics=[
            {"path": "diagnostics.ready", "op": "truthy"},
            {"path": "diagnostics.count", "op": "ge", "value": 2},
        ],
        secondary_targets=[
            {
                "id": "secondary-z",
                "path": "metrics.test.error",
                "published_mean": 0.5,
                "margin_absolute": 0.1,
            },
            {
                "id": "secondary-a",
                "path": "metrics.test.error",
                "published_mean": 0.5,
                "margin_absolute": 0.1,
            },
        ],
    )

    parsed = parse_acceptance_spec(raw)

    assert isinstance(parsed, AcceptanceSpec)
    assert parsed.repetitions == 2
    assert parsed.target.id == "primary"
    assert parsed.target.path == "metrics.test.error"
    assert parsed.deviations == ("a", "z")
    assert [target.id for target in parsed.secondary_targets] == [
        "secondary-a",
        "secondary-z",
    ]
    assert [rule.path for rule in parsed.required_diagnostics] == [
        "diagnostics.count",
        "diagnostics.ready",
    ]
    assert parsed.to_dict()["conformity"]["review"]["reviewed_by"] == ("reviewer:scientific")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(repetitions=1), "integer >= 2"),
        (lambda value: value.update(fidelity_ceiling="matched"), "fidelity_ceiling"),
        (lambda value: value["target"].update(transform="invert"), "transform"),
        (lambda value: value["target"].update(margin_absolute=-1), "must be >= 0"),
        (lambda value: value["conformity"].pop("review"), "requires basis, evidence"),
        (
            lambda value: value.update(
                required_diagnostics=[{"path": "x", "op": "between", "value": [2, 1]}]
            ),
            "bounds must be ordered",
        ),
        (
            lambda value: value.update(required_diagnostics=[{"path": "x", "op": "ge"}]),
            "value is required",
        ),
    ],
)
def test_parse_acceptance_spec_rejects_ambiguous_scientific_contracts(
    mutation: Any,
    message: str,
) -> None:
    raw = _spec()
    mutation(raw)

    with pytest.raises(AcceptanceSpecError, match=message):
        parse_acceptance_spec(raw)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(unexpected=True),
        lambda value: value["conformity"].update(unexpected=True),
        lambda value: value["conformity"]["review"].update(unexpected=True),
        lambda value: value["target"].update(unexpected=True),
        lambda value: value.update(
            secondary_targets=[
                {
                    "id": "secondary",
                    "path": "metrics.test.error",
                    "published_mean": 0.5,
                    "margin_absolute": 0.1,
                    "unexpected": True,
                }
            ]
        ),
        lambda value: value.update(
            diagnostic_targets=[
                {
                    "id": "diagnostic",
                    "path": "diagnostics.value",
                    "published_mean": 0.5,
                    "margin_absolute": 0.1,
                    "unexpected": True,
                }
            ]
        ),
        lambda value: value.update(
            required_diagnostics=[{"path": "diagnostics.ready", "op": "truthy", "unexpected": True}]
        ),
    ],
)
def test_acceptance_spec_rejects_unknown_keys_at_every_declarative_level(
    mutation: Any,
) -> None:
    raw = _spec()
    mutation(raw)

    with pytest.raises(AcceptanceSpecError, match="contains unknown keys"):
        parse_acceptance_spec(raw)


@pytest.mark.parametrize("op", ["present", "truthy", "nonempty"])
def test_valueless_diagnostic_operators_reject_value(op: str) -> None:
    raw = _spec(required_diagnostics=[{"path": "diagnostics.ready", "op": op, "value": None}])

    with pytest.raises(AcceptanceSpecError, match=f"value is forbidden for {op}"):
        parse_acceptance_spec(raw)


def test_conformity_review_requires_an_iso_timestamp_with_timezone() -> None:
    raw = _spec()
    raw["conformity"]["review"]["reviewed_at"] = "2026-08-29T12:00:00"

    with pytest.raises(AcceptanceSpecError, match="must include a timezone"):
        parse_acceptance_spec(raw)


def test_failed_conformity_requires_auditable_basis_evidence_and_review() -> None:
    invalid = _spec(conformity={"status": "failed"})
    with pytest.raises(AcceptanceSpecError, match="status=failed requires basis, evidence"):
        parse_acceptance_spec(invalid)

    failed_conformity = deepcopy(_spec()["conformity"])
    failed_conformity["status"] = "failed"
    report = evaluate_acceptance(
        _spec(conformity=failed_conformity),
        [_run(0, 0.4), _run(1, 0.6)],
    )

    assert report.assessment_status == "failed"
    assert report.fidelity_status == "paper_approx"
    assert "conformity=failed" in report.reasons


@pytest.mark.parametrize("status", ["pending", "not_assessed"])
def test_unresolved_conformity_is_not_evaluable(status: str) -> None:
    report = evaluate_acceptance(
        _spec(conformity={"status": status, "basis": None, "evidence": []}),
        [_run(0, 0.4), _run(1, 0.6)],
    )

    assert report.assessment_status == "not_evaluable"
    assert report.fidelity_status == "not_claimable"
    assert f"conformity={status}" in report.reasons


def test_acceptance_applies_transform_student_ci95_and_absolute_margin() -> None:
    report = evaluate_acceptance(_spec(), [_run(4, 0.4), _run(2, 0.6)])

    assert report.assessment_status == "passed"
    assert report.fidelity_status == "paper_matched"
    assert report.primary_target.available is True
    assert report.primary_target.summary == {
        "count": 2,
        "mean": 0.5,
        "std": pytest.approx(2**0.5 / 10),
        "std_ddof": 1,
        "population_std": pytest.approx(0.1),
        "min": pytest.approx(0.4),
        "max": pytest.approx(0.6),
        "ci95_low": pytest.approx(-0.7706204736432095),
        "ci95_high": pytest.approx(1.7706204736432095),
        "values": pytest.approx([0.4, 0.6]),
    }
    assert report.primary_target.std_absolute_difference == pytest.approx(0.0)
    assert report.primary_target.target_in_ci95 is True
    assert report.primary_target.within_margin is True
    assert [run["seed"] for run in report.runs] == [2, 4]


def test_required_diagnostics_support_every_declarative_operator() -> None:
    diagnostics = {
        "exact": "expected",
        "score": 3.0,
        "lower": 1.0,
        "bounded": 0.5,
        "ready": True,
        "items": [1],
        "present_even_when_null": None,
    }
    rules = [
        {"path": "diagnostics.exact", "op": "eq", "value": "expected"},
        {"path": "diagnostics.score", "op": "gt", "value": 2},
        {"path": "diagnostics.score", "op": "ge", "value": 3},
        {"path": "diagnostics.lower", "op": "lt", "value": 2},
        {"path": "diagnostics.lower", "op": "le", "value": 1},
        {"path": "diagnostics.bounded", "op": "between", "value": [0, 1]},
        {"path": "diagnostics.present_even_when_null", "op": "present"},
        {"path": "diagnostics.ready", "op": "truthy"},
        {"path": "diagnostics.items", "op": "nonempty"},
    ]

    report = evaluate_acceptance(
        _spec(required_diagnostics=rules),
        [_run(0, 0.4, diagnostics=diagnostics), _run(1, 0.6, diagnostics=diagnostics)],
    )

    assert report.assessment_status == "passed"
    assert len(report.required_diagnostics) == 9
    assert all(item.passed for item in report.required_diagnostics)
    assert all(
        item.checked_run_count == item.passed_run_count == 2 for item in report.required_diagnostics
    )
    assert report.to_dict()["diagnostic_failures"] == []


def test_numeric_and_required_diagnostic_gates_are_distinct_from_information() -> None:
    secondary = {
        "id": "secondary",
        "path": "metrics.test.error",
        "transform": "identity",
        "published_mean": 0.9,
        "margin_absolute": 0.01,
    }
    informational = {**secondary, "id": "informational"}
    diagnostic = {
        "id": "diagnostic",
        "path": "diagnostics.rate",
        "published_mean": 0.0,
        "margin_absolute": 0.01,
    }
    runs = [
        _run(0, 0.4, diagnostics={"rate": 1.0, "ready": False}),
        _run(1, 0.6, diagnostics={"rate": 1.0, "ready": True}),
    ]

    informational_only = evaluate_acceptance(
        _spec(informational_targets=[informational]),
        runs,
    )
    assert informational_only.assessment_status == "passed"
    assert informational_only.informational_targets[0].passed is False

    failed = evaluate_acceptance(
        _spec(
            secondary_targets=[secondary],
            diagnostic_targets=[diagnostic],
            required_diagnostics=[{"path": "diagnostics.ready", "op": "truthy"}],
        ),
        runs,
    )
    assert failed.assessment_status == "failed"
    assert failed.fidelity_status == "paper_approx"
    assert failed.secondary_targets[0].passed is False
    assert failed.diagnostic_targets[0].target_in_ci95 is False
    assert failed.diagnostic_targets[0].within_margin is False
    assert failed.required_diagnostics[0].failures[0].seed == 0
    assert {
        "secondary_target_margin_exceeded",
        "diagnostic_target_margin_exceeded",
        "required_diagnostics_failed",
    }.issubset(failed.reasons)


@pytest.mark.parametrize(
    "runs", [[], [_run(0, 0.4)], [_run(0, 0.4, status="failed"), _run(1, 0.6)]]
)
def test_incomplete_or_non_success_cohorts_fail_closed(runs: list[dict[str, Any]]) -> None:
    report = evaluate_acceptance(_spec(), runs)

    assert report.assessment_status == "not_evaluable"
    assert report.fidelity_status == "not_claimable"
    assert "repetitions_incomplete_or_non_success" in report.reasons


@pytest.mark.parametrize(
    ("updates", "assessment", "fidelity", "reason"),
    [
        ({"deviations": ["optimizer differs"]}, "passed", "paper_approx", "documented_deviations"),
        ({"unknowns": ["paper split unknown"]}, "passed", "not_claimable", "critical_unknowns"),
        (
            {"fidelity_ceiling": "paper_approx"},
            "passed",
            "paper_approx",
            "fidelity_ceiling=paper_approx",
        ),
        (
            {"fidelity_ceiling": "not_claimable"},
            "passed",
            "not_claimable",
            "fidelity_ceiling=not_claimable",
        ),
        (
            {"conformity": {"status": "pending", "evidence": [], "basis": None}},
            "not_evaluable",
            "not_claimable",
            "conformity=pending",
        ),
    ],
)
def test_fidelity_ceiling_and_protocol_knowledge_cap_claims(
    updates: dict[str, Any],
    assessment: str,
    fidelity: str,
    reason: str,
) -> None:
    report = evaluate_acceptance(_spec(**updates), [_run(0, 0.4), _run(1, 0.6)])

    assert report.assessment_status == assessment
    assert report.fidelity_status == fidelity
    assert reason in report.reasons


def test_report_hash_is_canonical_across_run_and_diagnostic_order() -> None:
    first_spec = _spec(
        required_diagnostics=[
            {"path": "diagnostics.ready", "op": "truthy"},
            {"path": "diagnostics.value", "op": "eq", "value": 1},
        ]
    )
    second_spec = deepcopy(first_spec)
    second_spec["required_diagnostics"].reverse()
    runs = [
        _run(9, 0.4, diagnostics={"ready": False, "value": 1}),
        _run(3, 0.6, diagnostics={"ready": True, "value": 1}),
    ]

    first = evaluate_acceptance(first_spec, runs)
    second = evaluate_acceptance(second_spec, reversed(runs))

    assert first.to_dict() == second.to_dict()
    payload = first.to_dict()
    digest = payload.pop("acceptance_sha256")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert digest == hashlib.sha256(canonical).hexdigest()
    assert len(digest) == 64


def test_duplicate_run_seeds_are_rejected_as_ambiguous() -> None:
    with pytest.raises(AcceptanceSpecError, match="duplicate seeds"):
        evaluate_acceptance(_spec(), [_run(1, 0.4), _run(1, 0.6)])


def test_each_authenticated_run_must_match_the_spec_method_id() -> None:
    with pytest.raises(AcceptanceSpecError, match="differs from acceptance spec method_id"):
        evaluate_acceptance(
            _spec(),
            [_run(0, 0.4), _run(1, 0.6, method_id="another-method")],
        )

    missing = _run(0, 0.4)
    missing["config"]["method"].pop("id")
    with pytest.raises(AcceptanceSpecError, match=r"config\.method\.id must be"):
        evaluate_acceptance(_spec(), [missing, _run(1, 0.6)])


def test_acceptance_spec_and_nested_objects_must_be_mappings() -> None:
    invalid: Any = []

    with pytest.raises(AcceptanceSpecError, match="acceptance spec must be a mapping"):
        parse_acceptance_spec(invalid)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-a-sequence", "must be a sequence of non-empty strings"),
        (["duplicate", "duplicate"], "must not contain duplicates"),
    ],
)
def test_acceptance_string_collections_are_unambiguous(value: Any, message: str) -> None:
    with pytest.raises(AcceptanceSpecError, match=message):
        parse_acceptance_spec(_spec(deviations=value))


@pytest.mark.parametrize("published_mean", [True, float("inf")])
def test_acceptance_numeric_targets_must_be_finite_numbers(published_mean: Any) -> None:
    raw = _spec()
    raw["target"]["published_mean"] = published_mean

    with pytest.raises(AcceptanceSpecError, match="published_mean must be a finite number"):
        parse_acceptance_spec(raw)


def test_acceptance_eq_diagnostics_must_contain_strict_json_data() -> None:
    raw = _spec(required_diagnostics=[{"path": "diagnostics.value", "op": "eq", "value": object()}])

    with pytest.raises(AcceptanceSpecError, match="must be strict JSON data"):
        parse_acceptance_spec(raw)


def test_conformity_review_rejects_non_iso_timestamp() -> None:
    raw = _spec()
    raw["conformity"]["review"]["reviewed_at"] = "not-a-date"

    with pytest.raises(AcceptanceSpecError, match="reviewed_at must be ISO-8601"):
        parse_acceptance_spec(raw)


def test_conformity_rejects_unknown_status() -> None:
    raw = _spec()
    raw["conformity"]["status"] = "approved"

    with pytest.raises(AcceptanceSpecError, match="status must be passed, pending, failed"):
        parse_acceptance_spec(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda raw: raw.update(
                diagnostic_targets=[
                    {
                        "id": "diagnostic",
                        "published_mean": 0.5,
                        "margin_absolute": 0.1,
                    }
                ]
            ),
            r"diagnostic_targets\[0\]\.path is required",
        ),
        (
            lambda raw: raw["target"].update(path="metrics.test.error"),
            "must use either path or split/metric",
        ),
        (
            lambda raw: raw["target"].update(published_std=-0.1),
            "published_std must be >= 0",
        ),
        (
            lambda raw: raw["target"].update(published_std_ddof=2),
            "published_std_ddof must be 0 or 1",
        ),
    ],
)
def test_target_spec_rejects_ambiguous_or_invalid_fields(mutation: Any, message: str) -> None:
    raw = _spec()
    mutation(raw)

    with pytest.raises(AcceptanceSpecError, match=message):
        parse_acceptance_spec(raw)


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"path": "diagnostics.value", "op": "approximately", "value": 1}, "op is invalid"),
        (
            {"path": "diagnostics.value", "op": "between", "value": [0]},
            "must contain two finite bounds",
        ),
    ],
)
def test_diagnostic_rule_rejects_invalid_operator_or_bounds(
    rule: dict[str, Any], message: str
) -> None:
    with pytest.raises(AcceptanceSpecError, match=message):
        parse_acceptance_spec(_spec(required_diagnostics=[rule]))


def test_target_and_diagnostic_collections_reject_invalid_sequences_and_duplicates() -> None:
    invalid_secondary = _spec(secondary_targets={})
    with pytest.raises(AcceptanceSpecError, match="secondary_targets must be a sequence"):
        parse_acceptance_spec(invalid_secondary)

    target = {
        "id": "duplicate",
        "path": "metrics.test.error",
        "published_mean": 0.5,
        "margin_absolute": 0.1,
    }
    with pytest.raises(AcceptanceSpecError, match="target ids must be unique"):
        parse_acceptance_spec(_spec(secondary_targets=[target, target]))

    cross_group_duplicate = {**target, "id": "primary"}
    with pytest.raises(AcceptanceSpecError, match="unique across all target groups"):
        parse_acceptance_spec(_spec(secondary_targets=[cross_group_duplicate]))

    invalid_rules = _spec(required_diagnostics={})
    with pytest.raises(AcceptanceSpecError, match="required_diagnostics must be a sequence"):
        parse_acceptance_spec(invalid_rules)

    rule = {"path": "diagnostics.ready", "op": "truthy"}
    with pytest.raises(AcceptanceSpecError, match="must not contain duplicates"):
        parse_acceptance_spec(_spec(required_diagnostics=[rule, rule]))


@pytest.mark.parametrize(
    ("run_field", "value", "message"),
    [
        ("seed", -1, "seed must be a non-negative integer"),
        ("status", "complete", "status must be success, failed, or not_evaluable"),
    ],
)
def test_run_identity_fields_are_strict(run_field: str, value: Any, message: str) -> None:
    invalid = _run(0, 0.4)
    invalid["run"][run_field] = value

    with pytest.raises(AcceptanceSpecError, match=message):
        evaluate_acceptance(_spec(), [invalid, _run(1, 0.6)])


def test_nested_sequence_paths_are_supported_by_numeric_targets() -> None:
    target = {
        "id": "first-rate",
        "path": "diagnostics.rates.0",
        "published_mean": 0.5,
        "margin_absolute": 0.0,
    }
    runs = [
        _run(0, 0.4, diagnostics={"rates": [0.5]}),
        _run(1, 0.6, diagnostics={"rates": [0.5]}),
    ]

    report = evaluate_acceptance(_spec(diagnostic_targets=[target]), runs)

    assert report.assessment_status == "passed"
    assert report.diagnostic_targets[0].observed_count == 2


@pytest.mark.parametrize("missing_kind", ["absent", "non_numeric", "non_finite"])
def test_missing_or_invalid_primary_values_are_not_evaluable(missing_kind: str) -> None:
    runs = [_run(0, 0.4), _run(1, 0.6)]
    for run in runs:
        if missing_kind == "absent":
            run["metrics"]["test"].pop("error")
        elif missing_kind == "non_numeric":
            run["metrics"]["test"]["error"] = True
        else:
            run["metrics"]["test"]["error"] = float("nan")

    report = evaluate_acceptance(_spec(), runs)

    assert report.assessment_status == "not_evaluable"
    assert "primary_target_missing" in report.reasons


def test_numeric_diagnostics_fail_closed_for_boolean_and_non_finite_values() -> None:
    rule = {"path": "diagnostics.score", "op": "ge", "value": 1}
    boolean_report = evaluate_acceptance(
        _spec(required_diagnostics=[rule]),
        [
            _run(0, 0.4, diagnostics={"score": True}),
            _run(1, 0.6, diagnostics={"score": True}),
        ],
    )

    assert boolean_report.assessment_status == "failed"
    assert boolean_report.required_diagnostics[0].passed is False

    with pytest.raises(AcceptanceSpecError, match="must be strict JSON data"):
        evaluate_acceptance(
            _spec(required_diagnostics=[rule]),
            [
                _run(0, 0.4, diagnostics={"score": float("nan")}),
                _run(1, 0.6, diagnostics={"score": float("nan")}),
            ],
        )


def test_missing_and_non_sized_required_diagnostics_are_explicit_failures() -> None:
    rules = [
        {"path": "diagnostics.missing", "op": "truthy"},
        {"path": "diagnostics.scalar", "op": "nonempty"},
    ]
    runs = [
        _run(0, 0.4, diagnostics={"scalar": 1}),
        _run(1, 0.6, diagnostics={"scalar": 1}),
    ]

    report = evaluate_acceptance(_spec(required_diagnostics=rules), runs)

    assert report.assessment_status == "failed"
    assert all(not item.passed for item in report.required_diagnostics)
    assert all(failure.present is False for failure in report.required_diagnostics[0].failures)
    assert all(failure.present is True for failure in report.required_diagnostics[1].failures)


def test_missing_secondary_and_diagnostic_targets_are_reported_separately() -> None:
    secondary = {
        "id": "secondary-missing",
        "path": "metrics.test.secondary_missing",
        "published_mean": 0.5,
        "margin_absolute": 0.1,
    }
    diagnostic = {
        "id": "diagnostic-missing",
        "path": "diagnostics.missing",
        "published_mean": 0.5,
        "margin_absolute": 0.1,
    }

    report = evaluate_acceptance(
        _spec(secondary_targets=[secondary], diagnostic_targets=[diagnostic]),
        [_run(0, 0.4), _run(1, 0.6)],
    )

    assert report.assessment_status == "failed"
    assert "secondary_target_missing" in report.reasons
    assert "diagnostic_target_missing" in report.reasons


def test_primary_target_outside_ci_and_margin_reports_both_numeric_failures() -> None:
    report = evaluate_acceptance(_spec(), [_run(0, 0.0), _run(1, 0.0)])

    assert report.assessment_status == "failed"
    assert "primary_target_outside_ci95" in report.reasons
    assert "primary_target_margin_exceeded" in report.reasons


def test_secondary_target_outside_ci_is_reported_even_within_margin() -> None:
    secondary = {
        "id": "secondary",
        "path": "metrics.test.error",
        "published_mean": 0.9,
        "margin_absolute": 0.5,
    }

    report = evaluate_acceptance(
        _spec(secondary_targets=[secondary]),
        [_run(0, 0.5), _run(1, 0.5)],
    )

    assert report.assessment_status == "failed"
    assert report.secondary_targets[0].within_margin is True
    assert report.secondary_targets[0].target_in_ci95 is False
    assert "secondary_target_outside_ci95" in report.reasons
