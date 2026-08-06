from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import bench.campaign.executor as executor_module
from bench.campaign.build_manifest import collect_environment_identity, environment_identity_sha256
from bench.campaign.errors import CampaignError, TaskLockedError
from bench.campaign.executor import (
    _acquire_lock,
    _classify_failure,
    _release_lock,
    _verify_dataset_content_state,
    _verify_effective_sampling_seeds,
    _verify_execution_target,
    _verify_preflight_report,
    execute_task,
    validate_result_directory,
)
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import load_manifest
from bench.campaign.models import CampaignTask
from bench.campaign.preflight_coverage import build_task_coverage
from bench.schema import ExperimentConfig
from bench.seed_sweep import apply_global_seed
from bench.utils.io import atomic_write_json
from modssc.data_loader import download_dataset, verify_dataset_content

from .helpers import FakeRunner, build_test_campaign, fake_versions


def _execute(
    *,
    repo: Path,
    campaign_dir: Path,
    result_root: Path,
    work_root: Path,
    runner: FakeRunner,
):
    return execute_task(
        campaign_dir / "manifest.jsonl",
        repo_root=repo,
        result_root=result_root,
        work_root=work_root,
        site_id="local",
        index=0,
        runner=runner,
        version_collector=fake_versions,
    )


def _preflight_policy_case(
    tmp_path: Path,
    *,
    expired: bool,
) -> tuple[dict[str, Any], CampaignTask, Path, dict[str, Any], datetime]:
    _, _, campaign_dir = build_test_campaign(tmp_path)
    meta, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    task = replace(
        tasks[0],
        campaign_id="article10-preflight-policy-test",
        resource_profile="a100_dev",
        campaign_stage="production",
        claim_eligible=True,
    )
    checked_at = datetime(2026, 7, 23, 12, tzinfo=UTC)
    expires_at = checked_at if expired else checked_at + timedelta(hours=1)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": (checked_at - timedelta(hours=1)).isoformat(),
        "expires_at": expires_at.isoformat(),
        "max_authorization_age_hours": 24.0,
        "status": "pass",
        "campaign_id": task.campaign_id,
        "manifest_sha256": meta["manifest_sha256"],
        "required_architecture": "A100",
        "task_coverage": build_task_coverage([task.task_id], architecture="A100"),
        "claim_scope_id": task.claim_scope_id,
        "campaign_stage": task.campaign_stage,
        "claim_eligible": task.claim_eligible,
        "gate_policy_id": task.gate_policy_id,
        "gate_policy_sha256": task.gate_policy_sha256,
    }
    report = tmp_path / "preflight-policy.json"
    atomic_write_json(report, payload)
    return meta, task, report, payload, checked_at


def test_effective_sampling_seed_check_accepts_legacy_v1_semantics(tmp_path) -> None:
    repo, config_path, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    task = tasks[0]
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    effective = apply_global_seed(
        raw,
        seed=task.seed,
        seeded_sections=None if task.seeded_sections is None else list(task.seeded_sections),
    )
    cfg = ExperimentConfig.from_dict(effective)
    legacy = replace(
        task,
        schema_version=1,
        split_seed=int(cfg.sampling.seed),
        sampling_component_seeds=None,
    )

    _verify_effective_sampling_seeds(legacy, cfg)

    effective["sampling"]["seed"] = None
    cfg_without_override = ExperimentConfig.from_dict(effective)
    legacy_without_override = replace(legacy, split_seed=task.seed)
    _verify_effective_sampling_seeds(legacy_without_override, cfg_without_override)


def test_effective_sampling_seed_check_rejects_v2_component_drift(tmp_path) -> None:
    _, config_path, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    task = tasks[0]
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    effective = apply_global_seed(
        raw,
        seed=task.seed,
        seeded_sections=None if task.seeded_sections is None else list(task.seeded_sections),
    )
    cfg = ExperimentConfig.from_dict(effective)
    assert task.sampling_component_seeds is not None
    altered = replace(
        task,
        sampling_component_seeds={
            **task.sampling_component_seeds,
            "labeling": task.sampling_component_seeds["labeling"] + 1,
        },
    )

    with pytest.raises(CampaignError, match="component seeds differ"):
        _verify_effective_sampling_seeds(altered, cfg)


@pytest.mark.parametrize(
    ("error", "expected_class"),
    [
        (
            CampaignError("E_CAMPAIGN_PREFLIGHT_INVALID", "CUDA out of memory"),
            "resource_oom",
        ),
        (
            CampaignError("E_CAMPAIGN_PREFLIGHT_INVALID", "preflight timeout"),
            "resource_timeout",
        ),
    ],
)
def test_precondition_classification_preserves_resource_failures(
    error: CampaignError, expected_class: str
) -> None:
    failure = _classify_failure(error, failure_phase="precondition")
    assert failure == {
        "failure_class": expected_class,
        "retryable": False,
        "resource_change_required": True,
    }


def test_run_task_applies_one_seed_publishes_and_is_idempotent(tmp_path) -> None:
    repo, _, campaign_dir = build_test_campaign(tmp_path)
    runner = FakeRunner()
    result_root = tmp_path / "results"
    first = _execute(
        repo=repo,
        campaign_dir=campaign_dir,
        result_root=result_root,
        work_root=tmp_path / "work",
        runner=runner,
    )
    second = _execute(
        repo=repo,
        campaign_dir=campaign_dir,
        result_root=result_root,
        work_root=tmp_path / "work",
        runner=runner,
    )

    assert first.status == "success" and not first.skipped
    assert second.status == "success" and second.skipped
    assert len(runner.calls) == 1
    effective = runner.calls[0]["raw"]
    assert effective["run"]["seed"] == 1
    assert "seeds" not in effective["run"]
    assert effective["sampling"]["seed"] == 1
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    assert effective["run"]["model_seed"] == tasks[0].model_seed
    assert runner.calls[0]["cfg"].run.model_seed == tasks[0].model_seed
    validate_result_directory(Path(first.result_dir), tasks[0])
    assert not (result_root / "locks" / f"{tasks[0].task_id}.lock").exists()


def test_existing_success_skips_before_preconditions_without_new_attempt(tmp_path) -> None:
    repo, _, campaign_dir = build_test_campaign(tmp_path)
    result_root = tmp_path / "results"
    runner = FakeRunner()
    first = _execute(
        repo=repo,
        campaign_dir=campaign_dir,
        result_root=result_root,
        work_root=tmp_path / "first-work",
        runner=runner,
    )
    blocked_preflight = tmp_path / "blocked-preflight.json"
    atomic_write_json(blocked_preflight, {"status": "blocked"})

    second = execute_task(
        campaign_dir / "manifest.jsonl",
        repo_root=repo,
        result_root=result_root,
        work_root=tmp_path / "second-work",
        site_id="wrong-site",
        index=0,
        preflight_report_path=blocked_preflight,
        environment_lock_sha256="wrong-environment",
        runner=runner,
        version_collector=fake_versions,
    )

    assert first.status == "success"
    assert second.status == "success"
    assert second.skipped is True
    assert len(runner.calls) == 1
    assert not (result_root / "attempts").exists()


def test_run_task_rejects_result_symlink_escape(tmp_path) -> None:
    repo, _, campaign_dir = build_test_campaign(tmp_path)
    result_root = tmp_path / "results"
    outside = tmp_path / "outside"
    result_root.mkdir()
    outside.mkdir()
    (result_root / "tasks").symlink_to(outside, target_is_directory=True)

    with pytest.raises(CampaignError, match="RESULT_PATH_OUTSIDE_ROOT"):
        _execute(
            repo=repo,
            campaign_dir=campaign_dir,
            result_root=result_root,
            work_root=tmp_path / "work",
            runner=FakeRunner(),
        )

    assert not list(outside.iterdir())


def test_success_staging_is_validated_before_publish_and_cleaned_on_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    task = tasks[0]
    result_root = tmp_path / "results"
    original_validate = executor_module.validate_result_directory
    staging_seen: list[Path] = []

    def reject_staging(result_dir: Path, candidate) -> tuple[Path, dict, str]:
        if ".staging" in Path(result_dir).parts:
            staging_seen.append(Path(result_dir))
            raise CampaignError("E_TEST_STAGING_INVALID", "injected staging validation failure")
        return original_validate(result_dir, candidate)

    monkeypatch.setattr(executor_module, "validate_result_directory", reject_staging)

    with pytest.raises(CampaignError, match="injected staging validation failure"):
        _execute(
            repo=repo,
            campaign_dir=campaign_dir,
            result_root=result_root,
            work_root=tmp_path / "work",
            runner=FakeRunner(),
        )

    assert len(staging_seen) == 1
    assert not (result_root / task.output_relpath).exists()
    staging_root = result_root / ".staging"
    assert not staging_root.exists() or not list(staging_root.iterdir())


def test_failure_staging_is_cleaned_when_envelope_write_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    task = tasks[0]
    result_root = tmp_path / "results"

    def fail_write(*args, **kwargs) -> None:
        _ = args, kwargs
        raise OSError("injected write failure")

    monkeypatch.setattr(executor_module, "atomic_write_json", fail_write)
    with pytest.raises(OSError, match="injected write failure"):
        executor_module._publish_failure(
            result_root=result_root,
            task=task,
            attempt_id="attempt-one",
            attempt_work=tmp_path / "work",
            run_json_path=None,
            error=RuntimeError("boom"),
            site_id="local",
        )

    staging_root = result_root / ".staging"
    assert not staging_root.exists() or not list(staging_root.iterdir())
    assert not (result_root / "attempts").exists()


def test_scientific_task_requires_matching_architecture_preflight(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, campaign_dir = build_test_campaign(tmp_path)
    meta, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    task = replace(
        tasks[0],
        campaign_id="article10-canary-r3-wave1-v1",
        resource_profile="a100_dev",
        campaign_stage="production",
        claim_eligible=True,
    )

    with pytest.raises(CampaignError, match="PREFLIGHT_REQUIRED"):
        _verify_preflight_report(task, meta, None)

    report = tmp_path / "preflight.json"
    checked_at = datetime(2026, 7, 23, 12, tzinfo=UTC)
    atomic_write_json(
        report,
        {
            "schema_version": 1,
            "created_at": (checked_at - timedelta(hours=1)).isoformat(),
            "expires_at": (checked_at + timedelta(hours=1)).isoformat(),
            "max_authorization_age_hours": 24.0,
            "status": "pass",
            "campaign_id": task.campaign_id,
            "manifest_sha256": meta["manifest_sha256"],
            "required_architecture": "V100",
            "task_coverage": build_task_coverage(["different-task"], architecture="V100"),
            "claim_scope_id": task.claim_scope_id,
            "campaign_stage": task.campaign_stage,
            "claim_eligible": task.claim_eligible,
            "gate_policy_id": task.gate_policy_id,
            "gate_policy_sha256": task.gate_policy_sha256,
        },
    )
    payload = yaml.safe_load(report.read_text(encoding="utf-8"))
    missing_coverage = dict(payload)
    missing_coverage.pop("task_coverage")
    atomic_write_json(report, missing_coverage)
    with pytest.raises(CampaignError, match="no immutable task coverage"):
        _verify_preflight_report(task, meta, report, now=checked_at)

    atomic_write_json(report, payload)
    with pytest.raises(CampaignError, match="absent from the preflight coverage"):
        _verify_preflight_report(task, meta, report, now=checked_at)

    payload["required_architecture"] = "A100"
    payload["task_coverage"] = build_task_coverage([task.task_id], architecture="A100")
    atomic_write_json(report, payload)
    _verify_preflight_report(task, meta, report, now=checked_at)

    payload["task_coverage"]["sha256"] = "0" * 64
    atomic_write_json(report, payload)
    with pytest.raises(CampaignError, match="coverage sha256 differs"):
        _verify_preflight_report(task, meta, report, now=checked_at)

    payload["task_coverage"] = build_task_coverage([task.task_id], architecture="A100")

    payload["expires_at"] = checked_at.isoformat()
    atomic_write_json(report, payload)
    with pytest.raises(CampaignError, match="expired"):
        _verify_preflight_report(task, meta, report, now=checked_at)

    payload["expires_at"] = (checked_at + timedelta(hours=1)).isoformat()
    atomic_write_json(report, payload)
    _verify_preflight_report(task, meta, report, now=checked_at)
    payload["expires_at"] = checked_at.isoformat()
    atomic_write_json(report, payload)
    with pytest.raises(CampaignError, match="expired"):
        _verify_preflight_report(task, meta, report, now=checked_at)


def test_fresh_preflight_without_queue_policy_remains_valid(tmp_path: Path) -> None:
    meta, task, report, _, checked_at = _preflight_policy_case(tmp_path, expired=False)

    assert _verify_preflight_report(task, meta, report, now=checked_at) is None


def test_fresh_preflight_ignores_operational_scheduler_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta, task, report, _, checked_at = _preflight_policy_case(tmp_path, expired=False)
    monkeypatch.setenv("MODSSC_EXECUTION_JOB_ID", "12345")
    monkeypatch.setenv("MODSSC_PREFLIGHT_EXPIRY_POLICY", "ignore_expiry")

    _verify_preflight_report(task, meta, report, now=checked_at)


def test_fresh_preflight_ignores_submission_adapter_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta, task, report, _, checked_at = _preflight_policy_case(tmp_path, expired=False)
    monkeypatch.setenv("MODSSC_EXECUTION_JOB_ID", "12345")
    monkeypatch.setenv("MODSSC_PREFLIGHT_EXPIRY_POLICY", "submitted_while_fresh")

    _verify_preflight_report(task, meta, report, now=checked_at)


def test_fresh_preflight_persists_dataset_evidence(
    tmp_path: Path,
) -> None:
    meta, task, report, payload, checked_at = _preflight_policy_case(tmp_path, expired=False)
    content_sha256 = "content-sha"
    content_task = replace(task, expected_dataset_content_sha256=content_sha256)
    payload["checks"] = [
        {
            "name": "datasets",
            "evidence_by_request": {
                content_task.dataset_request_sha256: {
                    "content_sha256": content_sha256,
                    "content_manifest_sha256": "manifest-sha",
                    "cache_state_sha256": "state-sha",
                    "cache_fingerprint": "cache-fingerprint",
                }
            },
        }
    ]
    atomic_write_json(report, payload)
    evidence = _verify_preflight_report(content_task, meta, report, now=checked_at)

    assert evidence is not None
    assert evidence["preflight_expiry_policy"] == "fresh"
    assert evidence["preflight_expired_at_execution"] == "false"


def test_fresh_preflight_does_not_parse_scheduler_job_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta, task, report, payload, checked_at = _preflight_policy_case(tmp_path, expired=False)
    payload["scheduler"] = {"job_id": "67890"}
    atomic_write_json(report, payload)
    monkeypatch.setenv("MODSSC_EXECUTION_JOB_ID", "12345")
    monkeypatch.setenv("MODSSC_PREFLIGHT_EXPIRY_POLICY", "generated_by_dependency")
    monkeypatch.setenv("MODSSC_PREFLIGHT_JOB_ID", "67891")

    content_task = replace(task, expected_dataset_content_sha256="content-sha")
    payload["checks"] = [
        {
            "name": "datasets",
            "evidence_by_request": {
                content_task.dataset_request_sha256: {
                    "content_sha256": "content-sha",
                    "content_manifest_sha256": "manifest-sha",
                    "cache_state_sha256": "state-sha",
                    "cache_fingerprint": "cache-fingerprint",
                }
            },
        }
    ]
    atomic_write_json(report, payload)
    evidence = _verify_preflight_report(content_task, meta, report, now=checked_at)

    assert evidence is not None
    assert evidence["preflight_expiry_policy"] == "fresh"
    assert "preflight_job_id" not in evidence


def test_task_detects_dataset_mutation_after_preflight(tmp_path) -> None:
    repo, config_path, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    dataset = download_dataset("toy", cache_dir=tmp_path / "cache")
    evidence = verify_dataset_content("toy", cache_dir=tmp_path / "cache", rehash=True)
    task = replace(
        tasks[0],
        expected_dataset_content_sha256=evidence["content_sha256"],
    )
    proof = {
        **evidence,
        "preflight_report_sha256": "preflight-sha",
        "preflight_expires_at": "2026-07-30T06:00:01+00:00",
        "preflight_expired_at_execution": "true",
        "preflight_expiry_policy": "submitted_while_fresh",
        "preflight_validated_at": "2026-07-29T10:49:14+00:00",
    }
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    verified = _verify_dataset_content_state(raw, task, proof)
    assert verified is not None
    assert verified["content_sha256"] == evidence["content_sha256"]
    assert verified["preflight_expiry_policy"] == "submitted_while_fresh"
    assert verified["preflight_expired_at_execution"] == "true"
    assert verified["preflight_validated_at"] == "2026-07-29T10:49:14+00:00"

    fingerprint = str(dataset.meta["dataset_cache_fingerprint"])
    array_path = tmp_path / "cache" / "processed" / fingerprint / "train_X.npy"
    values = np.load(array_path)
    values.flat[0] = values.flat[0] + 1
    np.save(array_path, values)
    stat = array_path.stat()
    os.utime(array_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

    with pytest.raises(CampaignError, match="changed after preflight"):
        _verify_dataset_content_state(raw, task, proof)


def test_success_envelope_persists_preflight_authorization_metadata(tmp_path: Path) -> None:
    repo, _, campaign_dir = build_test_campaign(tmp_path)
    source = _execute(
        repo=repo,
        campaign_dir=campaign_dir,
        result_root=tmp_path / "source-results",
        work_root=tmp_path / "source-work",
        runner=FakeRunner(),
    )
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    proof = {
        "content_sha256": "content-sha",
        "content_manifest_sha256": "manifest-sha",
        "cache_state_sha256": "state-sha",
        "cache_fingerprint": "cache-fingerprint",
        "preflight_report_sha256": "report-sha",
        "preflight_expires_at": "2026-07-30T06:00:01+00:00",
        "preflight_expired_at_execution": "true",
        "preflight_expiry_policy": "generated_by_dependency",
        "preflight_job_id": "67890",
    }

    published = executor_module._publish_success(
        result_root=tmp_path / "published-results",
        task=tasks[0],
        attempt_id="authorization-proof",
        run_dir=Path(source.result_dir) / "run",
        effective_config_path=Path(source.result_dir) / "effective.yaml",
        site_id="local",
        code_versions=fake_versions(),
        dataset_content_proof=proof,
    )

    task_payload = yaml.safe_load((published / "task.json").read_text(encoding="utf-8"))
    assert task_payload["dataset_content_proof"] == proof


def test_result_validation_rejects_tampered_partition_bundle(tmp_path) -> None:
    repo, _, campaign_dir = build_test_campaign(tmp_path)
    result = _execute(
        repo=repo,
        campaign_dir=campaign_dir,
        result_root=tmp_path / "results",
        work_root=tmp_path / "work",
        runner=FakeRunner(),
    )
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    result_dir = Path(result.result_dir)
    replay_arrays = result_dir / "run" / "sampling_split" / "arrays.npz"
    replay_arrays.write_bytes(b"tampered")

    with pytest.raises(CampaignError, match="split artifact digest"):
        validate_result_directory(result_dir, tasks[0])


def test_result_validation_rejects_tampered_effective_configuration(tmp_path) -> None:
    repo, _, campaign_dir = build_test_campaign(tmp_path)
    result = _execute(
        repo=repo,
        campaign_dir=campaign_dir,
        result_root=tmp_path / "results",
        work_root=tmp_path / "work",
        runner=FakeRunner(),
    )
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    result_dir = Path(result.result_dir)
    effective_config = result_dir / "effective.yaml"

    assert effective_config.is_file()
    effective_config.write_text("tampered: true\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="effective configuration digest differs"):
        validate_result_directory(result_dir, tasks[0])


def test_result_validation_checks_pre_registered_split_fingerprint(tmp_path) -> None:
    repo, _, campaign_dir = build_test_campaign(tmp_path)
    result = _execute(
        repo=repo,
        campaign_dir=campaign_dir,
        result_root=tmp_path / "results",
        work_root=tmp_path / "work",
        runner=FakeRunner(),
    )
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    pinned = replace(tasks[0], expected_split_fingerprint="different-split")

    with pytest.raises(CampaignError, match="split fingerprint differs"):
        validate_result_directory(Path(result.result_dir), pinned)


def test_run_task_refuses_changed_source_config(tmp_path) -> None:
    repo, config_path, campaign_dir = build_test_campaign(tmp_path)
    config_path.write_text(config_path.read_text(encoding="utf-8") + "# changed\n")

    with pytest.raises(CampaignError, match="CONFIG_CHANGED"):
        _execute(
            repo=repo,
            campaign_dir=campaign_dir,
            result_root=tmp_path / "results",
            work_root=tmp_path / "work",
            runner=FakeRunner(),
        )


def test_run_task_respects_existing_lock(tmp_path) -> None:
    repo, _, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    lock = tmp_path / "results" / "locks" / f"{tasks[0].task_id}.lock"
    lock.mkdir(parents=True)

    with pytest.raises(TaskLockedError):
        _execute(
            repo=repo,
            campaign_dir=campaign_dir,
            result_root=tmp_path / "results",
            work_root=tmp_path / "work",
            runner=FakeRunner(),
        )
    assert not (tmp_path / "results" / "attempts").exists()


def test_run_task_reclaims_only_explicitly_stale_lock(tmp_path) -> None:
    repo, _, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    result_root = tmp_path / "results"
    lock = result_root / "locks" / f"{tasks[0].task_id}.lock"
    lock.mkdir(parents=True)
    atomic_write_json(
        lock / "owner.json",
        {"task_id": tasks[0].task_id, "created_at": "2020-01-01T00:00:00+00:00"},
    )

    result = execute_task(
        campaign_dir / "manifest.jsonl",
        repo_root=repo,
        result_root=result_root,
        work_root=tmp_path / "work",
        site_id="local",
        index=0,
        reclaim_stale_lock_after=timedelta(hours=1),
        runner=FakeRunner(),
        version_collector=fake_versions,
    )

    assert result.status == "success"
    assert len(list((result_root / "stale-locks").iterdir())) == 1


def test_active_lock_cannot_be_reclaimed_and_release_checks_owner_token(tmp_path) -> None:
    _, _, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    task = tasks[0]
    lock_dir = tmp_path / "results" / "locks" / f"{task.task_id}.lock"
    first = _acquire_lock(
        lock_dir,
        task,
        site_id="local",
        reclaim_stale_after=None,
    )

    with pytest.raises(TaskLockedError):
        _acquire_lock(
            lock_dir,
            task,
            site_id="local",
            reclaim_stale_after=timedelta(0),
        )

    owner = yaml.safe_load((lock_dir / "owner.json").read_text(encoding="utf-8"))
    owner["owner_token"] = "different-owner"
    atomic_write_json(lock_dir / "owner.json", owner)
    assert not _release_lock(first)
    assert lock_dir.is_dir()


def test_run_task_refuses_wrong_logical_site(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _, campaign_dir = build_test_campaign(tmp_path)
    with pytest.raises(CampaignError, match="SITE_MISMATCH"):
        execute_task(
            campaign_dir / "manifest.jsonl",
            repo_root=repo,
            result_root=tmp_path / "results",
            work_root=tmp_path / "work",
            site_id="wrong",
            index=0,
            runner=FakeRunner(),
            version_collector=fake_versions,
        )


def test_execution_target_accepts_matching_logical_site(tmp_path) -> None:
    _, _, campaign_dir = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    task = replace(tasks[0], assigned_site="local-cpu")
    _verify_execution_target(task, site_id="local-cpu")


def test_run_task_verifies_environment_manifest_against_active_runtime(tmp_path) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base")
    identity = collect_environment_identity()
    digest = environment_identity_sha256(identity)
    environment_manifest = tmp_path / "environment.json"
    atomic_write_json(
        environment_manifest,
        {
            "schema_version": 1,
            "environment_lock": identity,
            "environment_lock_sha256": digest,
        },
    )
    spec = repo / "campaign.yaml"
    raw_spec = yaml.safe_load(spec.read_text(encoding="utf-8"))
    raw_spec["code"]["environment_lock_sha256"] = digest
    spec.write_text(yaml.safe_dump(raw_spec, sort_keys=False), encoding="utf-8")
    campaign = tmp_path / "locked-campaign"
    generate_campaign(spec, repo_root=repo, output_dir=campaign)

    result = execute_task(
        campaign / "manifest.jsonl",
        repo_root=repo,
        result_root=tmp_path / "results",
        work_root=tmp_path / "work",
        site_id="local",
        index=0,
        environment_manifest_path=environment_manifest,
        runner=FakeRunner(),
        version_collector=fake_versions,
    )
    assert result.status == "success"

    bad_identity = {**identity, "python": "0.0"}
    atomic_write_json(
        environment_manifest,
        {
            "schema_version": 1,
            "environment_lock": bad_identity,
            "environment_lock_sha256": environment_identity_sha256(bad_identity),
        },
    )
    with pytest.raises(CampaignError, match="ENVIRONMENT_MISMATCH"):
        execute_task(
            campaign / "manifest.jsonl",
            repo_root=repo,
            result_root=tmp_path / "other-results",
            work_root=tmp_path / "other-work",
            site_id="local",
            index=0,
            environment_manifest_path=environment_manifest,
            runner=FakeRunner(),
            version_collector=fake_versions,
        )


def test_failed_task_keeps_attempt_and_releases_lock(tmp_path) -> None:
    repo, _, campaign_dir = build_test_campaign(tmp_path)

    def failing_runner(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("boom")

    with pytest.raises(CampaignError, match="TASK_FAILED"):
        execute_task(
            campaign_dir / "manifest.jsonl",
            repo_root=repo,
            result_root=tmp_path / "results",
            work_root=tmp_path / "work",
            site_id="local",
            index=0,
            runner=failing_runner,
            version_collector=fake_versions,
        )
    _, tasks = load_manifest(campaign_dir / "manifest.jsonl")
    attempts = list(
        (tmp_path / "results" / "attempts" / tasks[0].task_id[:2] / tasks[0].task_id).iterdir()
    )
    assert len(attempts) == 1
    assert (attempts[0] / "attempt.json").is_file()
    assert not (tmp_path / "results" / "locks" / f"{tasks[0].task_id}.lock").exists()
