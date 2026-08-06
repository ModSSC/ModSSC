from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bench.campaign.attempts import seal_attempt_record
from bench.campaign.errors import CampaignError
from bench.campaign.executor import execute_task
from bench.campaign.manifest import load_manifest
from bench.campaign.reconcile import reconcile_campaign
from bench.utils.io import atomic_write_json
from tools.hpc.cli import main
from tools.hpc.daily_report import _resource_policy, generate_daily_report

from .helpers import (
    FakeRunner,
    build_test_campaign,
    fake_versions,
    rewrite_success_digest,
)


def _successful_first_task(*, repo: Path, campaign: Path, results: Path, work: Path) -> None:
    execute_task(
        campaign / "manifest.jsonl",
        repo_root=repo,
        result_root=results,
        work_root=work,
        site_id="local",
        index=0,
        runner=FakeRunner(),
        version_collector=fake_versions,
    )


def _failed_second_task_with_measurements(*, campaign: Path, results: Path) -> None:
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    task = tasks[1]
    attempt = results / "attempts" / task.task_id[:2] / task.task_id / "attempt-two"
    (attempt / "run").mkdir(parents=True)
    atomic_write_json(
        attempt / "attempt.json",
        seal_attempt_record(
            {
                "task_id": task.task_id,
                "row_sha256": task.row_sha256,
                "attempt_id": "attempt-two",
                "status": "failed",
                "site_id": "local",
                "finished_at": "2026-01-01T01:00:00+00:00",
                "error_type": "RuntimeError",
                "error": "test infrastructure failure",
                "traceback": "",
                "failure_phase": "run",
                "failure_class": "infrastructure",
                "retryable": True,
                "resource_change_required": False,
                "scheduler": {},
            }
        ),
    )
    atomic_write_json(
        attempt / "run" / "run.json",
        {
            "run": {
                "started_at": "2026-01-01T00:30:00+00:00",
                "finished_at": "2026-01-01T01:00:00+00:00",
            },
            "run_info": {
                "peak_ram_mib": 1024,
                "resource_usage": {"peak_vram_mib": 4096},
            },
        },
    )


def _prepared_report_inputs(tmp_path: Path) -> tuple[Path, Path]:
    repo, _, campaign = build_test_campaign(tmp_path)
    results = tmp_path / "results"
    _successful_first_task(
        repo=repo,
        campaign=campaign,
        results=results,
        work=tmp_path / "work",
    )
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    result_dir = results / tasks[0].output_relpath
    success_json = result_dir / "run" / "run.json"
    success = json.loads(success_json.read_text(encoding="utf-8"))
    success["run_info"] = {
        "run_time_seconds": 3600.0,
        "peak_ram_bytes": 2 * 1024**3,
        "peak_vram_bytes": 8 * 1024**3,
    }
    atomic_write_json(success_json, success)
    rewrite_success_digest(result_dir)
    _failed_second_task_with_measurements(campaign=campaign, results=results)
    reconcile_dir = tmp_path / "reconcile"
    reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=reconcile_dir,
    )
    return campaign, reconcile_dir / "reconcile.json"


def test_daily_report_counts_observed_attempts_and_memory(tmp_path) -> None:
    campaign, reconcile_path = _prepared_report_inputs(tmp_path)
    output = tmp_path / "daily"

    result = generate_daily_report(
        campaign / "manifest.jsonl",
        reconcile_path=reconcile_path,
        output_dir=output,
    )

    assert result.status == "incomplete"
    report = json.loads((output / "daily-usage.json").read_text(encoding="utf-8"))
    total = report["total"]
    assert total["task_count"] == 2
    assert total["success_count"] == 1
    assert total["failure_count"] == 1
    assert total["success_rate"] == pytest.approx(0.5)
    assert total["failure_rate"] == pytest.approx(0.5)
    assert total["retry_queue_task_count"] == 1
    assert total["tasks_with_failed_attempts_count"] == 1
    assert total["failed_attempt_count"] == 1
    assert total["observed_task_hours"] == pytest.approx(1.5)
    assert total["runtime_p50_seconds"] == pytest.approx(2700.0)
    assert total["runtime_p95_seconds"] == pytest.approx(3510.0)
    assert total["peak_ram_bytes"] == pytest.approx(2 * 1024**3)
    assert total["peak_vram_bytes"] == pytest.approx(8 * 1024**3)
    assert total["projected_total_task_hours"] == pytest.approx(2.0)
    assert total["projected_remaining_task_hours"] == pytest.approx(1.0)
    assert total["observed_gpu_hours"] is None
    assert report["by_resource_site"][0]["site"] == "local"
    assert report["by_resource_site"][0]["resource_profile"] == "cpu_test"

    with (output / "daily-usage-summary.csv").open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    assert {row["group_kind"] for row in summary_rows} == {
        "resource",
        "resource_site",
        "site",
        "total",
    }
    with (output / "daily-usage-runs.csv").open(newline="", encoding="utf-8") as stream:
        run_rows = list(csv.DictReader(stream))
    assert {row["attempt_kind"] for row in run_rows} == {"failed", "success"}
    assert {row["runtime_source"] for row in run_rows} == {
        "run.started_at/finished_at",
        "run_info.run_time_seconds",
    }


def test_daily_report_cli_writes_json_and_csv(tmp_path) -> None:
    campaign, reconcile_path = _prepared_report_inputs(tmp_path)
    output = tmp_path / "daily-cli"

    code = main(
        [
            "daily-report",
            "--manifest",
            str(campaign / "manifest.jsonl"),
            "--reconcile",
            str(reconcile_path),
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    assert (output / "daily-usage.json").is_file()
    assert (output / "daily-usage-summary.csv").is_file()
    assert (output / "daily-usage-runs.csv").is_file()


def test_daily_report_rejects_reconcile_for_another_manifest(tmp_path) -> None:
    campaign, reconcile_path = _prepared_report_inputs(tmp_path)
    payload = json.loads(reconcile_path.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    atomic_write_json(reconcile_path, payload)

    with pytest.raises(CampaignError, match="BINDING_INVALID"):
        generate_daily_report(
            campaign / "manifest.jsonl",
            reconcile_path=reconcile_path,
            output_dir=tmp_path / "daily-invalid",
        )


def test_daily_report_accounts_gpu_hours_and_recommends_safe_resources(tmp_path) -> None:
    campaign, reconcile_path = _prepared_report_inputs(tmp_path)
    resource_catalog = tmp_path / "resources.json"
    atomic_write_json(
        resource_catalog,
        {
            "schema_version": 1,
            "resources": [
                {
                    "site_id": "local",
                    "profile_id": "cpu_test",
                    "architecture": "A100",
                    "accelerators_per_task": 1,
                    "configured_walltime_seconds": 7200,
                    "max_walltime_seconds": 7200,
                    "initial_concurrency": 64,
                    "promoted_concurrency": 128,
                    "promotion_min_successes": 200,
                    "promotion_max_failure_rate": 0.02,
                }
            ],
        },
    )
    allocation = tmp_path / "allocation.yaml"
    allocation.write_text(
        """\
schema_version: 1
updated_at: '2026-07-23T00:00:00+02:00'
reserve_fraction: 0.15
architectures:
  A100:
    total_hours: 100
    consumed_hours: 10
    other_committed_hours: 5
""",
        encoding="utf-8",
    )

    generate_daily_report(
        campaign / "manifest.jsonl",
        reconcile_path=reconcile_path,
        output_dir=tmp_path / "daily-gpu",
        resource_catalog_path=resource_catalog,
        allocation_path=allocation,
    )

    report = json.loads((tmp_path / "daily-gpu" / "daily-usage.json").read_text())
    assert report["total"]["observed_gpu_hours"] == pytest.approx(1.5)
    assert report["total"]["projected_total_gpu_hours"] == pytest.approx(2.0)
    assert report["total"]["projected_remaining_gpu_hours"] == pytest.approx(1.0)
    policy = report["by_resource_site"][0]
    assert policy["recommended_concurrency"] == 64
    assert policy["promotion_eligible"] is False
    assert policy["recommended_walltime_seconds"] == 4500
    assert policy["recommended_walltime"] == "01:15:00"
    assert policy["walltime_status"] == "calibrated"
    assert policy["projected_completion_hours"] == pytest.approx(1.0)
    architecture = report["by_architecture"][0]
    assert architecture["architecture"] == "A100"
    budget = report["allocations"]["by_architecture"][0]
    assert budget["reserved_hours"] == pytest.approx(15.0)
    assert budget["spendable_hours"] == pytest.approx(70.0)
    assert budget["reserve_guard"] == "pass"


def test_daily_report_rejects_unknown_explained_oom(tmp_path) -> None:
    campaign, reconcile_path = _prepared_report_inputs(tmp_path)
    with pytest.raises(CampaignError, match="unknown explained OOM"):
        generate_daily_report(
            campaign / "manifest.jsonl",
            reconcile_path=reconcile_path,
            output_dir=tmp_path / "daily-invalid-oom",
            explained_oom_task_ids=["not-in-manifest"],
        )


def test_resource_policy_promotes_only_after_success_failure_and_oom_gates(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    task = tasks[0]
    repeated = [task] * 200
    resource = {
        "architecture": "A100",
        "accelerators_per_task": 1,
        "configured_walltime_seconds": 72000,
        "max_walltime_seconds": 72000,
        "initial_concurrency": 64,
        "promoted_concurrency": 128,
        "promotion_min_successes": 200,
        "promotion_max_failure_rate": 0.02,
    }
    summary = {
        "success_count": 200,
        "failure_count": 0,
        "success_runtime_p95_seconds": 60000,
        "projected_total_task_hours": 200.0,
    }
    states = {task.task_id: {"status": "success", "latest_failure_class": None}}

    promoted = _resource_policy(
        tasks=repeated,
        states=states,
        summary=summary,
        resource=resource,
        explained_oom_task_ids=set(),
    )
    assert promoted["promotion_eligible"] is True
    assert promoted["recommended_concurrency"] == 128
    assert promoted["walltime_status"] == "exceeds_cap"
    assert promoted["recommended_walltime_seconds"] == 75000
    assert promoted["recommended_walltime"] == "20:50:00"
    assert promoted["max_walltime_seconds"] == 72000

    states[task.task_id]["latest_failure_class"] = "resource_oom"
    blocked = _resource_policy(
        tasks=repeated,
        states=states,
        summary=summary,
        resource=resource,
        explained_oom_task_ids=set(),
    )
    assert blocked["promotion_eligible"] is False
    explained = _resource_policy(
        tasks=repeated,
        states=states,
        summary=summary,
        resource=resource,
        explained_oom_task_ids={task.task_id},
    )
    assert explained["promotion_eligible"] is True
