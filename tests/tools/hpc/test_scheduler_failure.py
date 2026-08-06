from __future__ import annotations

import json

import pytest

from bench.campaign.errors import CampaignError
from bench.campaign.executor import _acquire_lock, _release_lock
from bench.campaign.manifest import load_manifest
from bench.utils.io import atomic_write_json
from tests.bench.campaign.helpers import build_test_campaign
from tools.hpc.scheduler_failure import record_scheduler_failure


def _scheduler() -> dict[str, str]:
    return {
        "slurm_job_id": "90211_0",
        "slurm_array_job_id": "90211",
        "slurm_array_task_id": "0",
        "slurm_cluster_name": "test-cluster",
    }


def test_scheduler_failure_is_atomic_idempotent_and_quarantines_free_lock(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    task = tasks[0]
    results = tmp_path / "results"
    lock_dir = results / "locks" / f"{task.task_id}.lock"
    lock_dir.mkdir(parents=True)
    atomic_write_json(
        lock_dir / "owner.json",
        {"schema_version": 1, "task_id": task.task_id, "owner_token": "dead-process"},
    )

    first = record_scheduler_failure(
        campaign / "manifest.jsonl",
        meta_path=campaign / "manifest.meta.json",
        result_root=results,
        site_id="local",
        index=0,
        failure_class="resource_oom",
        scheduler_state="OUT_OF_MEMORY",
        exit_code=137,
        scheduler_metadata=_scheduler(),
    )
    second = record_scheduler_failure(
        campaign / "manifest.jsonl",
        meta_path=campaign / "manifest.meta.json",
        result_root=results,
        site_id="local",
        index=0,
        failure_class="resource_oom",
        scheduler_state="OUT_OF_MEMORY|",
        exit_code=137,
        scheduler_metadata=_scheduler(),
    )

    assert first.skipped is False
    assert first.orphan_lock_action == "quarantined"
    assert first.orphan_lock_quarantine is not None
    assert second.skipped is True
    assert second.attempt_dir == first.attempt_dir
    assert second.orphan_lock_action == "absent"
    assert not lock_dir.exists()
    assert len(list((results / "orphaned-locks").iterdir())) == 1
    attempts = [
        path
        for path in (results / "attempts" / task.task_id[:2] / task.task_id).iterdir()
        if path.is_dir()
    ]
    assert len(attempts) == 1
    assert str(attempts[0]) == first.attempt_dir
    payload = json.loads((attempts[0] / "attempt.json").read_text(encoding="utf-8"))
    assert payload["failure_class"] == "resource_oom"
    assert payload["retryable"] is False
    assert payload["resource_change_required"] is True
    assert payload["scheduler_identity"]["array_job_id"] == "90211"
    assert not any((results / ".staging").iterdir())


def test_scheduler_failure_does_not_reclaim_lock_while_guard_is_held(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    task = tasks[0]
    results = tmp_path / "results"
    lock_dir = results / "locks" / f"{task.task_id}.lock"
    active_lock = _acquire_lock(
        lock_dir,
        task,
        site_id="local",
        reclaim_stale_after=None,
    )
    try:
        result = record_scheduler_failure(
            campaign / "manifest.jsonl",
            meta_path=campaign / "manifest.meta.json",
            result_root=results,
            site_id="local",
            index=0,
            failure_class="resource_timeout",
            scheduler_state="TERM",
            exit_code=143,
            scheduler_metadata=_scheduler(),
        )
        assert result.orphan_lock_action == "guard_busy"
        assert lock_dir.is_dir()
    finally:
        assert _release_lock(active_lock)


def test_scheduler_failure_requires_slurm_identity_and_rejects_conflicts(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    arguments = {
        "meta_path": campaign / "manifest.meta.json",
        "result_root": tmp_path / "results",
        "site_id": "local",
        "index": 0,
    }
    with pytest.raises(CampaignError, match="SCHEDULER_IDENTITY"):
        record_scheduler_failure(
            campaign / "manifest.jsonl",
            failure_class="resource_timeout",
            scheduler_metadata={},
            **arguments,
        )

    record_scheduler_failure(
        campaign / "manifest.jsonl",
        failure_class="resource_timeout",
        scheduler_metadata=_scheduler(),
        **arguments,
    )
    with pytest.raises(CampaignError, match="SCHEDULER_FAILURE_CONFLICT"):
        record_scheduler_failure(
            campaign / "manifest.jsonl",
            failure_class="resource_oom",
            scheduler_metadata=_scheduler(),
            **arguments,
        )
