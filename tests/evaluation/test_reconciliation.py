from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from modssc.evaluation import (
    SeedReconciliationError,
    reconcile_seed_reports,
)
from modssc.runtime.execution import RunIdentity
from modssc.runtime.protocol import effective_config_sha256, protocol_sha256
from modssc.runtime.software import software_sha256

_DEFAULT_METRICS = object()
_DEFAULT_HASHES = object()
_CONFIG_HASH = "a" * 64
_REPORT_CONFIG = {"method": {"params": {}}}
_VERSIONS = {"python": "x", "modssc": "x", "numpy": "x", "git_sha": "x"}
_EFFECTIVE_CONFIG_HASH = effective_config_sha256(_REPORT_CONFIG)
_PROTOCOL_HASH = protocol_sha256(_REPORT_CONFIG)
_SOFTWARE_HASH = software_sha256(_VERSIONS)


def _report(
    seed: Any,
    *,
    status: str = "success",
    metrics: Any = _DEFAULT_METRICS,
    run_info: Any = None,
    hashes: Any = _DEFAULT_HASHES,
    config: Any = _REPORT_CONFIG,
    versions: Any = _VERSIONS,
    portable_identity: bool = True,
) -> dict[str, Any]:
    if metrics is _DEFAULT_METRICS:
        metrics = {"test": {"accuracy": float(seed) / 10.0}} if status == "success" else None
    if hashes is _DEFAULT_HASHES:
        hashes = {
            "config_hash": _CONFIG_HASH,
            "effective_config_hash": effective_config_sha256(config),
            "protocol_sha256": protocol_sha256(config),
            "software_sha256": software_sha256(versions),
        }
    report = {
        "run": {
            "seed": seed,
            "name": f"seed-{seed}",
            "run_id": f"run-{seed}",
            "status": status,
            "error_code": None,
        },
        "hashes": hashes,
        "config": config,
        "versions": versions,
        "metrics": metrics,
        "run_info": run_info,
        "task_info": {"kind": "classification"},
        "graph_info": None,
        "error": None,
    }
    if portable_identity:
        try:
            identity = RunIdentity(
                config_sha256=hashes["protocol_sha256"],
                seed=seed,
                code_sha256=hashes["software_sha256"],
            )
        except (KeyError, TypeError, ValueError):
            pass
        else:
            report["execution_identity"] = identity.to_dict()
            report["hashes"]["execution_identity_sha256"] = identity.sha256
            report["run"]["run_id"] = identity.short_id
    return report


def test_reconciliation_builds_an_exact_partition_and_native_aggregate() -> None:
    reports = [
        _report(
            2,
            status="failed",
            run_info={
                "run_time_seconds": 2,
                "gpu_device": "GPU-B",
                "hardware_mismatch": True,
            },
        ),
        _report(
            3,
            metrics={"test": {"accuracy": 0.9, "policy": "final"}},
            run_info={
                "run_time_seconds": 3.0,
                "gpu_device": "GPU-A",
                "hardware_mismatch": True,
            },
        ),
        _report(
            1,
            status="not_evaluable",
            run_info={
                "run_time_seconds": False,
                "gpu_device": "",
                "hardware_mismatch": False,
            },
        ),
    ]
    sources = [Path("two/run.json"), Path("three/run.json"), Path("one/run.json")]

    result = reconcile_seed_reports(
        requested_seeds=[3, 1, 2, 4],
        reports=reports,
        source_paths=sources,
    )

    assert result.categories() == {
        "success": [3],
        "failed": [2],
        "not_evaluable": [1],
        "missing": [4],
    }
    assert set().union(*map(set, result.categories().values())) == {1, 2, 3, 4}
    assert sum(len(values) for values in result.categories().values()) == 4
    assert result.status == "partial_failure"
    assert result.certifiable is False
    assert result.summary() == {
        "requested_seeds": [3, 1, 2, 4],
        "requested_run_count": 4,
        "completed_run_count": 3,
        "successful_run_count": 1,
        "failed_run_count": 1,
        "not_evaluable_run_count": 1,
        "missing_run_count": 1,
        "success_seeds": [3],
        "failed_seeds": [2],
        "not_evaluable_seeds": [1],
        "missing_seeds": [4],
        "categories": {
            "success": [3],
            "failed": [2],
            "not_evaluable": [1],
            "missing": [4],
        },
        "execution_identity_complete": False,
        "status": "partial_failure",
        "certifiable": False,
    }
    assert result.metrics["test"]["accuracy"]["values"] == [0.9]
    assert "policy" not in result.metrics["test"]
    assert result.run_info["gpu_devices"] == ["GPU-A", "GPU-B"]
    assert result.run_info["gpu_device"] == "Mixed"
    assert result.run_info["hardware_mismatch_count"] == 2
    assert result.run_info["run_time_seconds"]["values"] == [3.0, 2.0]
    assert [run["seed"] for run in result.runs] == [3, 1, 2]
    assert result.runs[0]["run_json"] == "three/run.json"
    assert result.runs[0]["run_dir"] == "three"


def test_reconciliation_statuses_cover_complete_and_non_success_outcomes() -> None:
    complete = reconcile_seed_reports(
        requested_seeds=[1],
        reports=[
            _report(
                1,
                run_info={"run_time_seconds": 1.0, "gpu_device": "CPU"},
            )
        ],
    )
    assert complete.status == "success"
    assert complete.certifiable is True
    assert complete.execution_identity_complete is True
    assert complete.run_info["gpu_device"] == "CPU"
    assert complete.runs[0]["run_json"] is None
    assert complete.runs[0]["run_dir"] is None

    not_evaluable = reconcile_seed_reports(
        requested_seeds=[1],
        reports=[_report(1, status="not_evaluable")],
    )
    assert not_evaluable.status == "not_evaluable"

    failed = reconcile_seed_reports(
        requested_seeds=[1],
        reports=[_report(1, status="failed", run_info="unavailable")],
    )
    assert failed.status == "failed"
    assert failed.metrics == {}
    assert failed.run_info == {
        "gpu_devices": [],
        "gpu_device": None,
        "hardware_mismatch": False,
        "hardware_mismatch_count": 0,
    }

    missing = reconcile_seed_reports(requested_seeds=[1], reports=[])
    assert missing.status == "failed"

    not_evaluable_and_missing = reconcile_seed_reports(
        requested_seeds=[1, 2],
        reports=[_report(1, status="not_evaluable")],
    )
    assert not_evaluable_and_missing.status == "failed"

    not_evaluable_and_failed = reconcile_seed_reports(
        requested_seeds=[1, 2],
        reports=[
            _report(1, status="not_evaluable"),
            _report(2, status="failed"),
        ],
    )
    assert not_evaluable_and_failed.status == "failed"


@pytest.mark.parametrize(
    ("requested_seeds", "message"),
    [
        ([], "must not be empty"),
        ([True], "non-negative integer"),
        ([-1], "non-negative integer"),
        (["1"], "non-negative integer"),
        ([1, 1], "contains duplicates"),
    ],
)
def test_reconciliation_rejects_invalid_requested_seeds(
    requested_seeds: list[Any],
    message: str,
) -> None:
    with pytest.raises(SeedReconciliationError, match=message):
        reconcile_seed_reports(requested_seeds=requested_seeds, reports=[])


def test_reconciliation_rejects_source_count_mismatch() -> None:
    with pytest.raises(SeedReconciliationError, match="one value per report"):
        reconcile_seed_reports(
            requested_seeds=[1],
            reports=[_report(1)],
            source_paths=[],
        )


@pytest.mark.parametrize(
    ("report", "message"),
    [
        ([], r"reports\[0\] must be a mapping"),
        ({"run": []}, "report.run must be a mapping"),
        (_report(True), "report.run.seed must be a non-negative integer"),
        (_report(1, status="unknown"), "report.run.status"),
        (_report(1, metrics=[], status="failed"), "report.metrics"),
        (_report(1, metrics=None), "successful report must contain metrics"),
    ],
)
def test_reconciliation_rejects_malformed_reports(report: Any, message: str) -> None:
    with pytest.raises(SeedReconciliationError, match=message):
        reconcile_seed_reports(requested_seeds=[1], reports=[report])


def test_reconciliation_rejects_unexpected_and_duplicate_observations() -> None:
    with pytest.raises(SeedReconciliationError, match="unexpected observed seed: 2"):
        reconcile_seed_reports(requested_seeds=[1], reports=[_report(2)])

    with pytest.raises(SeedReconciliationError, match="duplicate observed seed: 1"):
        reconcile_seed_reports(
            requested_seeds=[1],
            reports=[_report(1), _report(1)],
        )


def test_reconciliation_rejects_incompatible_successful_metric_schemas() -> None:
    with pytest.raises(SeedReconciliationError, match="metric records are incompatible"):
        reconcile_seed_reports(
            requested_seeds=[1, 2],
            reports=[
                _report(1, metrics={"test": {"accuracy": 0.8}}),
                _report(2, metrics={"test": {"macro_f1": 0.7}}),
            ],
        )


def test_reconciliation_validates_an_expected_config_hash_per_seed() -> None:
    report = _report(1)

    result = reconcile_seed_reports(
        requested_seeds=[1],
        reports=[report],
        expected_config_hashes={1: _CONFIG_HASH},
        expected_protocol_hashes={1: _PROTOCOL_HASH},
    )

    assert result.status == "success"
    assert result.runs[0]["hashes"] == report["hashes"]


@pytest.mark.parametrize(
    ("expected_hashes", "message"),
    [
        ([], "must be a mapping"),
        ({True: _CONFIG_HASH}, "seed must be a non-negative integer"),
        ({1: ""}, r"expected_config_hashes\[1\] must be a lowercase SHA-256"),
        ({2: _CONFIG_HASH}, "keys must exactly match requested_seeds"),
    ],
)
def test_reconciliation_rejects_invalid_expected_config_hashes(
    expected_hashes: Any,
    message: str,
) -> None:
    with pytest.raises(SeedReconciliationError, match=message):
        reconcile_seed_reports(
            requested_seeds=[1],
            reports=[],
            expected_config_hashes=expected_hashes,
        )


@pytest.mark.parametrize(
    ("hashes", "message"),
    [
        (None, r"reports\[0\]\.hashes must be a mapping"),
        ({}, "hashes.config_hash must be a lowercase SHA-256"),
        (
            {
                "config_hash": "e" * 64,
                "effective_config_hash": _EFFECTIVE_CONFIG_HASH,
                "protocol_sha256": _PROTOCOL_HASH,
                "software_sha256": _SOFTWARE_HASH,
            },
            "config hash mismatch for seed 1",
        ),
    ],
)
def test_reconciliation_rejects_missing_or_mismatched_report_identity(
    hashes: Any,
    message: str,
) -> None:
    report = _report(1)
    report["hashes"] = (
        {
            **hashes,
            "execution_identity_sha256": report["hashes"]["execution_identity_sha256"],
        }
        if isinstance(hashes, dict)
        and {"config_hash", "effective_config_hash", "protocol_sha256", "software_sha256"}
        <= set(hashes)
        else hashes
    )

    with pytest.raises(SeedReconciliationError, match=message):
        reconcile_seed_reports(
            requested_seeds=[1],
            reports=[report],
            expected_config_hashes={1: _CONFIG_HASH},
        )


def test_reconciliation_rejects_protocol_mismatch_and_software_heterogeneity() -> None:
    with pytest.raises(SeedReconciliationError, match="protocol hash mismatch for seed 1"):
        reconcile_seed_reports(
            requested_seeds=[1],
            reports=[_report(1)],
            expected_protocol_hashes={1: "e" * 64},
        )

    different_versions = {**_VERSIONS, "numpy": "y"}
    with pytest.raises(SeedReconciliationError, match="software hash differs"):
        reconcile_seed_reports(
            requested_seeds=[1, 2],
            reports=[_report(1), _report(2, versions=different_versions)],
            expected_config_hashes={1: _CONFIG_HASH, 2: _CONFIG_HASH},
        )


@pytest.mark.parametrize("field", ["protocol_sha256", "software_sha256"])
def test_reconciliation_requires_complete_run_identity(field: str) -> None:
    hashes = {
        "config_hash": _CONFIG_HASH,
        "effective_config_hash": _EFFECTIVE_CONFIG_HASH,
        "protocol_sha256": _PROTOCOL_HASH,
        "software_sha256": _SOFTWARE_HASH,
    }
    hashes.pop(field)

    with pytest.raises(SeedReconciliationError, match=rf"hashes\.{field}"):
        reconcile_seed_reports(
            requested_seeds=[1],
            reports=[_report(1, hashes=hashes)],
        )


def test_expected_protocol_identity_may_differ_by_seed() -> None:
    first_config = {"run": {"seed": 1}, "method": {"params": {"alpha": 0.1}}}
    second_config = {"run": {"seed": 2}, "method": {"params": {"alpha": 0.2}}}
    first_protocol = protocol_sha256(first_config)
    second_protocol = protocol_sha256(second_config)
    first_report = _report(1, config=first_config)
    second_report = _report(2, config=second_config)

    result = reconcile_seed_reports(
        requested_seeds=[1, 2],
        reports=[first_report, second_report],
        expected_protocol_hashes={1: first_protocol, 2: second_protocol},
    )

    assert result.status == "success"

    with pytest.raises(SeedReconciliationError, match="without expected identities"):
        reconcile_seed_reports(
            requested_seeds=[1, 2],
            reports=[first_report, second_report],
        )

    normalized_cohort = reconcile_seed_reports(
        requested_seeds=[1, 2],
        reports=[first_report, second_report],
        expected_config_hashes={1: _CONFIG_HASH, 2: _CONFIG_HASH},
    )
    assert normalized_cohort.status == "success"


def test_reconciliation_recomputes_protocol_and_software_hashes_from_report_payloads() -> None:
    falsified_protocol = _report(1)
    falsified_protocol["hashes"] = {
        **falsified_protocol["hashes"],
        "protocol_sha256": "0" * 64,
    }
    falsified_identity = RunIdentity(
        config_sha256="0" * 64,
        seed=1,
        code_sha256=falsified_protocol["hashes"]["software_sha256"],
    )
    falsified_protocol["execution_identity"] = falsified_identity.to_dict()
    falsified_protocol["hashes"]["execution_identity_sha256"] = falsified_identity.sha256
    falsified_protocol["run"]["run_id"] = falsified_identity.short_id
    with pytest.raises(SeedReconciliationError, match="protocol hash does not match report config"):
        reconcile_seed_reports(requested_seeds=[1], reports=[falsified_protocol])

    falsified_versions = _report(1)
    falsified_versions["versions"] = {**falsified_versions["versions"], "numpy": "tampered"}
    with pytest.raises(
        SeedReconciliationError,
        match="software hash does not match report versions",
    ):
        reconcile_seed_reports(requested_seeds=[1], reports=[falsified_versions])

    falsified_config = _report(1)
    falsified_config["config"] = {"method": {"params": {"tampered": True}}}
    with pytest.raises(
        SeedReconciliationError,
        match="effective config hash does not match report config",
    ):
        reconcile_seed_reports(requested_seeds=[1], reports=[falsified_config])


def test_reconciliation_authenticates_portable_execution_identity() -> None:
    report = _report(1, portable_identity=False)
    identity = RunIdentity(
        config_sha256=report["hashes"]["protocol_sha256"],
        seed=1,
        code_sha256=report["hashes"]["software_sha256"],
    )
    report["execution_identity"] = identity.to_dict()
    report["hashes"]["execution_identity_sha256"] = identity.sha256
    report["run"]["run_id"] = identity.short_id

    result = reconcile_seed_reports(requested_seeds=[1], reports=[report])

    assert result.runs[0]["execution_identity"] == identity.to_dict()
    assert result.runs[0]["execution_identity_sha256"] == identity.sha256

    tampered = {**report, "run": {**report["run"], "run_id": "host-dependent-id"}}
    with pytest.raises(SeedReconciliationError, match="portable execution identity"):
        reconcile_seed_reports(requested_seeds=[1], reports=[tampered])


def test_reconciliation_rejects_each_malformed_execution_identity_component() -> None:
    missing_payload = _report(1)
    missing_payload.pop("execution_identity")
    with pytest.raises(SeedReconciliationError, match="incomplete execution identity"):
        reconcile_seed_reports(requested_seeds=[1], reports=[missing_payload])

    missing_digest = _report(1)
    missing_digest["hashes"].pop("execution_identity_sha256")
    with pytest.raises(SeedReconciliationError, match="incomplete execution identity"):
        reconcile_seed_reports(requested_seeds=[1], reports=[missing_digest])

    invalid_payload = _report(1)
    invalid_payload["execution_identity"] = {}
    with pytest.raises(SeedReconciliationError, match="execution identity is invalid"):
        reconcile_seed_reports(requested_seeds=[1], reports=[invalid_payload])

    mismatched_digest = _report(1)
    mismatched_digest["hashes"]["execution_identity_sha256"] = "0" * 64
    with pytest.raises(SeedReconciliationError, match="identity hash does not match"):
        reconcile_seed_reports(requested_seeds=[1], reports=[mismatched_digest])

    def _replace_identity(report: dict[str, Any], identity: RunIdentity) -> None:
        report["execution_identity"] = identity.to_dict()
        report["hashes"]["execution_identity_sha256"] = identity.sha256
        report["run"]["run_id"] = identity.short_id

    mismatched_seed = _report(1)
    _replace_identity(
        mismatched_seed,
        RunIdentity(
            config_sha256=_PROTOCOL_HASH,
            seed=2,
            code_sha256=_SOFTWARE_HASH,
        ),
    )
    with pytest.raises(SeedReconciliationError, match="identity seed does not match"):
        reconcile_seed_reports(requested_seeds=[1], reports=[mismatched_seed])

    mismatched_protocol = _report(1)
    _replace_identity(
        mismatched_protocol,
        RunIdentity(
            config_sha256="0" * 64,
            seed=1,
            code_sha256=_SOFTWARE_HASH,
        ),
    )
    with pytest.raises(SeedReconciliationError, match="identity protocol does not match"):
        reconcile_seed_reports(requested_seeds=[1], reports=[mismatched_protocol])

    mismatched_software = _report(1)
    _replace_identity(
        mismatched_software,
        RunIdentity(
            config_sha256=_PROTOCOL_HASH,
            seed=1,
            code_sha256="0" * 64,
        ),
    )
    with pytest.raises(SeedReconciliationError, match="identity software does not match"):
        reconcile_seed_reports(requested_seeds=[1], reports=[mismatched_software])


def test_legacy_execution_identity_requires_explicit_opt_in() -> None:
    report = _report(1, portable_identity=False)

    with pytest.raises(SeedReconciliationError, match="missing required execution identity"):
        reconcile_seed_reports(requested_seeds=[1], reports=[report])

    result = reconcile_seed_reports(
        requested_seeds=[1],
        reports=[report],
        require_execution_identity=False,
    )
    assert result.status == "success"
    assert result.execution_identity_complete is False
    assert result.certifiable is False


def test_reconciliation_reports_unhashable_config_and_invalid_software_manifest() -> None:
    unhashable_config = _report(1)
    unhashable_config["config"] = {"invalid_json_value": {1}}
    with pytest.raises(SeedReconciliationError, match="cannot hash report config"):
        reconcile_seed_reports(requested_seeds=[1], reports=[unhashable_config])

    invalid_versions = _report(1)
    invalid_versions["versions"] = {"software_manifest": {"schema_version": 999}}
    with pytest.raises(SeedReconciliationError, match="cannot hash report versions"):
        reconcile_seed_reports(requested_seeds=[1], reports=[invalid_versions])
