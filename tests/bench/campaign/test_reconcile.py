from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from bench.campaign.aggregate import _critical_95, aggregate_successes
from bench.campaign.attempts import seal_attempt_record, seal_authorization_event
from bench.campaign.errors import CampaignError
from bench.campaign.executor import _acquire_lock, _release_lock, execute_task
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from bench.campaign.reconcile import materialize_reconcile_paths, reconcile_campaign
from bench.utils.io import atomic_write_json
from tools.hpc.scheduler_failure import record_scheduler_failure
from tools.hpc.slurm_renderer import render_slurm_sites

from .helpers import (
    FakeRunner,
    build_test_campaign,
    fake_versions,
    rewrite_success_digest,
    write_yaml,
)


def _run_first(repo: Path, campaign: Path, results: Path, work: Path) -> None:
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


def _write_failure_attempt(
    parent: Path,
    task,
    *,
    attempt_id: str,
    failure_class: str,
    finished_at: str,
) -> Path:
    retryable, resource_change_required = {
        "deterministic": (False, False),
        "infrastructure": (True, False),
        "resource_oom": (False, True),
        "resource_timeout": (False, True),
    }[failure_class]
    attempt = parent / attempt_id
    attempt.mkdir(parents=True)
    atomic_write_json(
        attempt / "attempt.json",
        seal_attempt_record(
            {
                "task_id": task.task_id,
                "row_sha256": task.row_sha256,
                "attempt_id": attempt_id,
                "status": "failed",
                "site_id": "local",
                "finished_at": finished_at,
                "error_type": "TestFailure",
                "error": failure_class,
                "traceback": "",
                "failure_phase": "run",
                "failure_class": failure_class,
                "retryable": retryable,
                "resource_change_required": resource_change_required,
                "scheduler": {},
            }
        ),
    )
    return attempt


def test_reconcile_success_missing_and_retry_indices(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    results = tmp_path / "results"
    _run_first(repo, campaign, results, tmp_path / "work")
    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "reconcile",
    )

    assert report.status == "incomplete"
    assert report.counts == {"missing": 1, "success": 1}
    assert report.retry_count == 1
    assert (tmp_path / "reconcile" / "retry" / "cpu_test.indices").read_text() == "1\n"
    assert len((tmp_path / "reconcile" / "successful-run-json.txt").read_text().splitlines()) == 1
    retry_meta, retry_tasks = load_manifest(tmp_path / "reconcile" / "retry.jsonl")
    assert retry_meta["source_manifest_sha256"] is not None
    assert [task.task_index for task in retry_tasks] == [1]

    execute_task(
        tmp_path / "reconcile" / "retry.jsonl",
        repo_root=repo,
        result_root=results,
        work_root=tmp_path / "retry-work",
        site_id="local",
        index=1,
        runner=FakeRunner(),
        version_collector=fake_versions,
    )
    completed = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "completed",
    )
    assert completed.counts == {"success": 2}
    assert completed.retry_count == 0


def test_reconcile_seals_portable_paths_and_keeps_runtime_bindings_external(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    results = tmp_path / "results"
    _run_first(repo, campaign, results, tmp_path / "work")
    output = tmp_path / "reconcile"

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=output,
    )
    report_path = Path(report.report_path)
    sealed = json.loads(report_path.read_text(encoding="utf-8"))
    binding_path = tmp_path / ".reconcile.runtime-roots.json"

    assert sealed["result_roots"] == ["result://root-000"]
    assert str(results.resolve()) not in json.dumps(sealed, sort_keys=True)
    assert binding_path.is_file()
    assert not any(
        entry["path"] == binding_path.name
        for entry in json.loads((output / "BUNDLE.json").read_text(encoding="utf-8"))["files"]
    )

    materialized = materialize_reconcile_paths(report_path, sealed)
    assert materialized["result_roots"] == [str(results.resolve())]
    assert materialized["tasks"][0]["result_dirs"] == [
        str((results / load_manifest(campaign / "manifest.jsonl")[1][0].output_relpath).resolve())
    ]


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_reconcile_portable_paths_require_the_authenticated_runtime_binding(
    tmp_path, mutation: str
) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    output = tmp_path / "reconcile"
    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[tmp_path / "results"],
        output_dir=output,
    )
    report_path = Path(report.report_path)
    sealed = json.loads(report_path.read_text(encoding="utf-8"))
    binding_path = tmp_path / ".reconcile.runtime-roots.json"
    if mutation == "missing":
        binding_path.unlink()
        error = "BINDING_REQUIRED"
    else:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        binding["reconcile_sha256"] = "0" * 64
        atomic_write_json(binding_path, binding)
        error = "BINDING_INVALID"

    with pytest.raises(CampaignError, match=error):
        materialize_reconcile_paths(report_path, sealed)


def test_reconcile_builds_submittable_retry_campaign_with_original_index(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path, with_site=True)
    results = tmp_path / "results"
    _run_first(repo, campaign, results, tmp_path / "work")

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "reconcile",
    )

    retry_campaign = tmp_path / "reconcile" / "retry-campaign"
    assert report.retry_campaign_path == str(retry_campaign.resolve())
    retry_meta, retry_tasks = load_manifest(retry_campaign / "manifest.jsonl")
    assert (
        retry_meta["source_manifest_sha256"]
        == sha256((campaign / "manifest.jsonl").read_bytes()).hexdigest()
    )
    assert [task.task_index for task in retry_tasks] == [1]

    retry_manifest_sha256 = sha256((retry_campaign / "manifest.jsonl").read_bytes()).hexdigest()
    assert not (retry_campaign / "submit").exists()
    submission_dir = tmp_path / "submissions" / "retry"
    render_slurm_sites(
        site_paths=[repo / "site.yaml"],
        campaign_dir=retry_campaign,
        submission_dir=submission_dir,
    )
    resources = json.loads(
        (retry_campaign / "profiles" / "resources.json").read_text(encoding="utf-8")
    )
    assert resources["manifest_sha256"] == retry_manifest_sha256
    array_index = retry_campaign / resources["array_indices"][0]["path"]
    assert array_index.read_text(encoding="utf-8") == "1\n"
    assert resources["array_indices"][0]["sha256"] == sha256(array_index.read_bytes()).hexdigest()
    wrapper = (submission_dir / "local" / "cpu_test.slurm").read_text(encoding="utf-8")
    assert f"export MODSSC_CAMPAIGN_MANIFEST_SHA256={retry_manifest_sha256}" in wrapper
    assert f"export MODSSC_ARRAY_INDEX_SHA256={resources['array_indices'][0]['sha256']}" in wrapper

    executed = execute_task(
        retry_campaign / "manifest.jsonl",
        repo_root=repo,
        result_root=results,
        work_root=tmp_path / "retry-work",
        site_id="local",
        index=1,
        runner=FakeRunner(),
        version_collector=fake_versions,
    )
    assert executed.task_id == retry_tasks[0].task_id


def test_reconcile_refuses_existing_retry_campaign(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path, with_site=True)
    destination = tmp_path / "reconcile"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="BUNDLE_EXISTS"):
        reconcile_campaign(
            campaign / "manifest.jsonl",
            result_roots=[tmp_path / "results"],
            output_dir=tmp_path / "reconcile",
        )

    assert marker.read_text(encoding="utf-8") == "operator-owned\n"


def test_retry_campaign_resource_planning_is_operational_and_atomic(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _, campaign = build_test_campaign(tmp_path, with_site=True)

    def fail_plan(**_kwargs: object) -> Path:
        raise CampaignError("E_TEST_PLAN", "injected resource-plan failure")

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[tmp_path / "results"],
        output_dir=tmp_path / "reconcile",
    )
    monkeypatch.setattr("tools.hpc.slurm_renderer.plan_resource_sites", fail_plan)
    with pytest.raises(CampaignError, match="injected resource-plan failure"):
        render_slurm_sites(
            site_paths=[repo / "site.yaml"],
            campaign_dir=Path(str(report.retry_campaign_path)),
        )
    assert not Path(str(report.retry_campaign_path), "profiles/resources.json").exists()


def test_reconcile_requires_sites_to_cover_each_retry_task_exactly_once(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path, with_site=True)
    site = yaml.safe_load((repo / "site.yaml").read_text(encoding="utf-8"))
    site["site_id"] = "different-site"
    unmatched_site = repo / "unmatched-site.yaml"
    write_yaml(unmatched_site, site)

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[tmp_path / "results"],
        output_dir=tmp_path / "reconcile",
    )
    with pytest.raises(CampaignError, match="cover every campaign task exactly once"):
        render_slurm_sites(
            site_paths=[unmatched_site],
            campaign_dir=Path(str(report.retry_campaign_path)),
        )


@pytest.mark.parametrize("emit_retry", [False, True])
def test_reconcile_does_not_render_retry_wrappers_without_retry_tasks(
    tmp_path, emit_retry: bool
) -> None:
    repo, _, campaign = build_test_campaign(tmp_path, with_site=True)
    results = tmp_path / "results"
    if emit_retry:
        for index in (0, 1):
            execute_task(
                campaign / "manifest.jsonl",
                repo_root=repo,
                result_root=results,
                work_root=tmp_path / f"work-{index}",
                site_id="local",
                index=index,
                runner=FakeRunner(),
                version_collector=fake_versions,
            )

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "reconcile",
        emit_retry=emit_retry,
    )

    assert report.retry_count == 0
    assert report.retry_campaign_path is None
    assert not (tmp_path / "reconcile" / "retry-campaign").exists()


def test_reconcile_marks_stale_lock_retryable(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    results = tmp_path / "results"
    lock = results / "locks" / f"{tasks[0].task_id}.lock"
    lock.mkdir(parents=True)
    atomic_write_json(
        lock / "owner.json",
        {
            "task_id": tasks[0].task_id,
            "created_at": "2020-01-01T00:00:00+00:00",
        },
    )
    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "reconcile",
        stale_after=timedelta(hours=1),
    )

    assert report.counts == {"missing": 1, "stale": 1}
    assert report.retry_count == 2
    assert lock.exists()


def test_reconcile_keeps_an_eighty_hour_match_segment_running_by_default(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    results = tmp_path / "results"
    lock = results / "locks" / f"{tasks[0].task_id}.lock"
    lock.mkdir(parents=True)
    atomic_write_json(
        lock / "owner.json",
        {
            "task_id": tasks[0].task_id,
            "created_at": (datetime.now(UTC) - timedelta(hours=80)).isoformat(),
        },
    )

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "reconcile",
    )

    assert report.counts == {"missing": 1, "running": 1}
    assert report.retry_count == 1


def test_reconcile_never_retries_after_a_deterministic_failure(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    results = tmp_path / "results"
    parent = results / "attempts" / tasks[0].task_id[:2] / tasks[0].task_id
    for index, failure_class in enumerate(("deterministic", "infrastructure", "resource_oom")):
        _write_failure_attempt(
            parent,
            tasks[0],
            attempt_id=f"attempt-{index}",
            failure_class=failure_class,
            finished_at=f"2026-01-01T00:00:0{index}+00:00",
        )

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "reconcile",
    )

    assert report.counts == {"blocked": 1, "missing": 1}
    assert report.retry_count == 1


def test_reconcile_allows_three_infrastructure_retries_then_blocks(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    results = tmp_path / "results"
    parent = results / "attempts" / tasks[0].task_id[:2] / tasks[0].task_id
    for index in range(3):
        _write_failure_attempt(
            parent,
            tasks[0],
            attempt_id=f"attempt-{index}",
            failure_class="infrastructure",
            finished_at=f"2026-01-01T00:00:0{index}+00:00",
        )

    retryable = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "retryable",
    )
    assert retryable.counts == {"failed": 1, "missing": 1}
    assert retryable.retry_count == 2

    _write_failure_attempt(
        parent,
        tasks[0],
        attempt_id="attempt-3",
        failure_class="infrastructure",
        finished_at="2026-01-01T00:00:03+00:00",
    )
    blocked = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "blocked",
    )
    assert blocked.counts == {"blocked": 1, "missing": 1}
    assert blocked.retry_count == 1


def test_reconcile_uses_only_the_latest_authorization_or_attempt_event(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    task = tasks[0]
    results = tmp_path / "results"
    event_id = "authorization-expired-1"
    event_dir = results / "events" / task.task_id[:2] / task.task_id / event_id
    event_dir.mkdir(parents=True)
    atomic_write_json(
        event_dir / "event.json",
        seal_authorization_event(
            {
                "task_id": task.task_id,
                "row_sha256": task.row_sha256,
                "event_id": event_id,
                "event_class": "authorization_expired",
                "site_id": "local",
                "preflight_report_sha256": "a" * 64,
                "expires_at": "2026-01-01T00:00:00+00:00",
                "observed_at": "2026-01-01T00:01:00+00:00",
            }
        ),
    )
    _write_failure_attempt(
        results / "attempts" / task.task_id[:2] / task.task_id,
        task,
        attempt_id="infrastructure-after-expiry",
        failure_class="infrastructure",
        finished_at="2026-01-01T00:02:00+00:00",
    )

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "reconcile",
    )

    assert report.counts == {"failed": 1, "missing": 1}
    assert report.retry_count == 2


@pytest.mark.parametrize("contract", ["preflight", "environment"])
def test_pre_run_contract_failures_publish_blocking_attempts(tmp_path: Path, contract: str) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    execution_kwargs: dict[str, object] = {}
    expected_error = "PREFLIGHT_INVALID"
    if contract == "preflight":
        report = tmp_path / "preflight.json"
        atomic_write_json(report, {"status": "blocked"})
        execution_kwargs["preflight_report_path"] = report
    else:
        spec = repo / "campaign.yaml"
        payload = yaml.safe_load(spec.read_text(encoding="utf-8"))
        payload["code"]["environment_lock_sha256"] = "expected-environment"
        write_yaml(spec, payload)
        campaign = tmp_path / "locked-campaign"
        generate_campaign(spec, repo_root=repo, output_dir=campaign)
        execution_kwargs["environment_lock_sha256"] = "different-environment"
        expected_error = "ENVIRONMENT_MISMATCH"

    results = tmp_path / "results"
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    for task in tasks:
        with pytest.raises(CampaignError, match=expected_error):
            execute_task(
                campaign / "manifest.jsonl",
                repo_root=repo,
                result_root=results,
                work_root=tmp_path / "work",
                site_id="local",
                index=task.task_index,
                runner=FakeRunner(),
                version_collector=fake_versions,
                **execution_kwargs,
            )

        attempt_parent = results / "attempts" / task.task_id[:2] / task.task_id
        attempts = list(attempt_parent.iterdir())
        assert len(attempts) == 1
        attempt = json.loads((attempts[0] / "attempt.json").read_text(encoding="utf-8"))
        assert attempt["failure_phase"] == "precondition"
        assert attempt["failure_class"] == "deterministic"
        assert attempt["retryable"] is False
        assert attempt["resource_change_required"] is False

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "reconcile",
    )
    assert report.counts == {"blocked": 2}
    assert report.retry_count == 0


def test_reconcile_prioritises_later_success_over_precondition_attempt(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    results = tmp_path / "results"
    invalid_preflight = tmp_path / "preflight.json"
    atomic_write_json(invalid_preflight, {"status": "blocked"})

    with pytest.raises(CampaignError, match="PREFLIGHT_INVALID"):
        execute_task(
            campaign / "manifest.jsonl",
            repo_root=repo,
            result_root=results,
            work_root=tmp_path / "failed-work",
            site_id="local",
            index=0,
            preflight_report_path=invalid_preflight,
            runner=FakeRunner(),
            version_collector=fake_versions,
        )
    _run_first(repo, campaign, results, tmp_path / "successful-work")

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "reconcile",
    )
    assert report.counts == {"missing": 1, "success": 1}
    assert report.retry_count == 1


def test_reconcile_detects_conflicting_successes_across_roots(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    root_one = tmp_path / "root-one"
    root_two = tmp_path / "root-two"
    _run_first(repo, campaign, root_one, tmp_path / "work")
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    first = root_one / tasks[0].output_relpath
    second = root_two / tasks[0].output_relpath
    second.parent.mkdir(parents=True)
    shutil.copytree(first, second)
    run_json = second / "run" / "run.json"
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    payload["run"]["finished_at"] = "2026-01-02T00:00:00+00:00"
    atomic_write_json(run_json, payload)
    rewrite_success_digest(second)

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[root_one, root_two],
        output_dir=tmp_path / "reconcile",
    )
    assert report.status == "invalid"
    assert report.counts == {"conflict": 1, "missing": 1}
    assert report.retry_count == 1


def test_reconcile_deduplicates_identical_mirrored_successes(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    root_one = tmp_path / "root-one"
    root_two = tmp_path / "root-two"
    _run_first(repo, campaign, root_one, tmp_path / "work")
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    first = root_one / tasks[0].output_relpath
    second = root_two / tasks[0].output_relpath
    second.parent.mkdir(parents=True)
    shutil.copytree(first, second)

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[root_one, root_two],
        output_dir=tmp_path / "reconcile",
    )
    assert report.status == "incomplete"
    assert report.counts == {"missing": 1, "success": 1}


def test_reconcile_marks_tampered_effective_configuration_corrupt(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    results = tmp_path / "results"
    _run_first(repo, campaign, results, tmp_path / "work")
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    effective_config = results / tasks[0].output_relpath / "effective.yaml"
    effective_config.write_text("tampered: true\n", encoding="utf-8")

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "reconcile",
    )

    assert report.status == "invalid"
    assert report.counts == {"corrupt": 1, "missing": 1}


def test_reconcile_requires_explicit_reprofile_after_oom(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    results = tmp_path / "results"

    def oom_runner(*_args, **_kwargs):
        raise MemoryError("CUDA out of memory")

    with pytest.raises(CampaignError, match="TASK_FAILED"):
        execute_task(
            campaign / "manifest.jsonl",
            repo_root=repo,
            result_root=results,
            work_root=tmp_path / "work",
            site_id="local",
            index=0,
            runner=oom_runner,
            version_collector=fake_versions,
        )
    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "reconcile",
    )
    records = (tmp_path / "reconcile" / "reprofile-required.jsonl").read_text().splitlines()

    assert report.counts == {"missing": 1, "resource_blocked": 1}
    assert report.retry_count == 1
    assert len(records) == 1
    reprofile = json.loads(records[0])
    assert reprofile["failure_class"] == "resource_oom"
    assert len(reprofile["task_ids"]) == 2
    assert reprofile["seeds"] == [1, 2]
    assert "every seed" in reprofile["action"]


def test_aggregate_rejects_mixed_resource_profiles_within_one_cell(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    mixed = [tasks[0], replace(tasks[1], resource_profile="h100_long")]
    states = [
        {"task_id": task.task_id, "status": "missing", "run_json_paths": []} for task in mixed
    ]

    with pytest.raises(CampaignError, match="reprofile every seed"):
        aggregate_successes(tasks=mixed, states=states, output_dir=tmp_path / "aggregate")


def test_aggregate_accepts_a_declared_single_seed_diagnostic_cell(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    results = tmp_path / "results"
    _run_first(repo, campaign, results, tmp_path / "work")
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    task = replace(tasks[0], required_seed_count=1)
    run_json = results / task.output_relpath / "run" / "run.json"

    aggregation = aggregate_successes(
        tasks=[task],
        states=[
            {
                "task_id": task.task_id,
                "status": "success",
                "run_json_paths": [str(run_json)],
            }
        ],
        output_dir=tmp_path / "aggregate",
    )

    assert aggregation["complete_cells"] == 1
    assert aggregation["incomplete_cells"] == 0


def test_aggregate_reports_population_std_but_uses_sample_std_for_ci95(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    values = [0.49, 0.51]
    states = []
    for task, value in zip(tasks, values, strict=True):
        run_json = tmp_path / "runs" / task.task_id / "run.json"
        atomic_write_json(run_json, {"metrics": {"test": {"accuracy": value}}})
        states.append(
            {
                "task_id": task.task_id,
                "status": "success",
                "run_json_paths": [str(run_json)],
            }
        )

    aggregate_successes(
        tasks=tasks,
        states=states,
        output_dir=tmp_path / "aggregate",
    )
    with (tmp_path / "aggregate" / "aggregates.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = next(item for item in rows if item["split"] == "test" and item["metric"] == "accuracy")

    population_std = 0.01
    sample_std = math.sqrt(0.0002)
    half_width = _critical_95(2) * sample_std / math.sqrt(2)
    assert float(row["population_std"]) == pytest.approx(population_std)
    assert float(row["std"]) == pytest.approx(sample_std)
    assert int(row["std_ddof"]) == 1
    assert float(row["ci95_low"]) == pytest.approx(0.5 - half_width)
    assert float(row["ci95_high"]) == pytest.approx(0.5 + half_width)


def test_reconcile_prioritises_scheduler_resource_failure_over_orphan_lock(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    results = tmp_path / "results"
    lock_dir = results / "locks" / f"{tasks[0].task_id}.lock"
    task_lock = _acquire_lock(
        lock_dir,
        tasks[0],
        site_id="local",
        reclaim_stale_after=None,
    )
    try:
        recorded = record_scheduler_failure(
            campaign / "manifest.jsonl",
            meta_path=campaign / "manifest.meta.json",
            result_root=results,
            site_id="local",
            index=0,
            failure_class="resource_timeout",
            scheduler_state="TIMEOUT",
            exit_code=143,
            scheduler_metadata={
                "slurm_job_id": "12345_0",
                "slurm_array_job_id": "12345",
                "slurm_array_task_id": "0",
            },
        )
        assert recorded.orphan_lock_action == "guard_busy"

        report = reconcile_campaign(
            campaign / "manifest.jsonl",
            result_roots=[results],
            output_dir=tmp_path / "reconcile",
        )
        payload = json.loads(Path(report.report_path).read_text(encoding="utf-8"))

        assert report.counts == {"missing": 1, "resource_blocked": 1}
        assert report.retry_count == 1
        assert payload["tasks"][0]["latest_failure_class"] == "resource_timeout"
        assert payload["tasks"][0]["locks"]
    finally:
        assert _release_lock(task_lock)
