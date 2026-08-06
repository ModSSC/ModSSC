from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any

import pytest

from bench.campaign.checkpoint import checkpoint_identity
from bench.campaign.errors import CampaignError
from bench.campaign.executor import execute_task
from bench.campaign.manifest import load_manifest, sha256_file
from bench.campaign.reconcile import reconcile_campaign
from bench.utils.io import atomic_write_json
from modssc.runtime.continuation import (
    continuation_requested,
    raise_planned_continuation,
)
from tools.hpc.slurm_renderer import render_slurm_sites

from .helpers import FakeRunner, build_test_campaign, fake_versions


def _write_trainer_checkpoint(root: Path, *, payload: bytes = b"trainer-state") -> None:
    checkpoint_path = root / "checkpoint.pt"
    checkpoint_path.write_bytes(payload)
    atomic_write_json(
        root / "checkpoint.json",
        {
            "schema_version": 1,
            "task_id": os.environ["MODSSC_TASK_ID"],
            "identity_sha256": os.environ["MODSSC_CHECKPOINT_IDENTITY_SHA256"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
    )


class _ContinueThenSucceed:
    def __init__(self) -> None:
        self.calls = 0
        self.success = FakeRunner()

    def __call__(self, config_path: Path, *, raw: dict[str, Any], cfg: Any) -> Any:
        self.calls += 1
        checkpoint_root = Path(os.environ["MODSSC_CHECKPOINT_ROOT"])
        assert os.environ["MODSSC_TASK_ID"]
        if self.calls == 1:
            assert os.environ["MODSSC_CHECKPOINT_RESUME"] == "0"
            _write_trainer_checkpoint(checkpoint_root)
            os.kill(os.getpid(), signal.SIGUSR1)
            assert continuation_requested()
            raise_planned_continuation()
        assert os.environ["MODSSC_CHECKPOINT_RESUME"] == "1"
        assert (checkpoint_root / "checkpoint.pt").read_bytes() == b"trainer-state"
        return self.success(config_path, raw=raw, cfg=cfg)


def _execute(
    *,
    repo: Path,
    campaign: Path,
    results: Path,
    work: Path,
    checkpoints: Path,
    runner: Any,
) -> Any:
    return execute_task(
        campaign / "manifest.jsonl",
        repo_root=repo,
        result_root=results,
        work_root=work,
        checkpoint_root=checkpoints,
        site_id="local",
        index=0,
        runner=runner,
        version_collector=fake_versions,
    )


def test_planned_continuation_is_authenticated_and_not_counted_as_failure(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path, with_site=True)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    task = tasks[0]
    results = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"
    runner = _ContinueThenSucceed()

    first = _execute(
        repo=repo,
        campaign=campaign,
        results=results,
        work=tmp_path / "first-work",
        checkpoints=checkpoints,
        runner=runner,
    )

    assert first.status == "continuation"
    attempt = json.loads((Path(first.attempt_dir) / "attempt.json").read_text(encoding="utf-8"))
    assert attempt["status"] == "continuation"
    assert attempt["event_class"] == "planned_continuation"
    assert attempt["failure_class"] is None
    task_dir = checkpoints / "tasks" / task.task_id[:2] / task.task_id
    marker = json.loads((task_dir / "CONTINUE.json").read_text(encoding="utf-8"))
    assert marker["identity_sha256"] == checkpoint_identity(task)["identity_sha256"]
    assert marker["checkpoint_manifest_sha256"] == attempt["checkpoint_manifest_sha256"]

    report = reconcile_campaign(
        campaign / "manifest.jsonl",
        result_roots=[results],
        output_dir=tmp_path / "reconcile",
    )
    assert report.counts == {"continuation_pending": 1, "missing": 1}
    assert report.retry_count == 1
    assert report.continuation_count == 1
    assert (tmp_path / "reconcile" / "continuation.jsonl").is_file()
    assert report.continuation_campaign_path == str(
        (tmp_path / "reconcile" / "continuation-campaign").resolve()
    )
    continuation_campaign = tmp_path / "reconcile" / "continuation-campaign"
    submission_dir = tmp_path / "submissions" / "continuation"
    render_slurm_sites(
        site_paths=[repo / "site.yaml"],
        campaign_dir=continuation_campaign,
        submission_dir=submission_dir,
    )
    continuation_wrapper = submission_dir / "local" / "cpu_test.slurm"
    assert "MODSSC_CAMPAIGN_CHECKPOINT_ROOT" in continuation_wrapper.read_text(encoding="utf-8")

    second = _execute(
        repo=repo,
        campaign=campaign,
        results=results,
        work=tmp_path / "second-work",
        checkpoints=checkpoints,
        runner=runner,
    )
    assert second.status == "success"
    assert runner.calls == 2
    assert not (task_dir / "CONTINUE.json").exists()
    assert len(list((task_dir / "history").glob("resumed-*.CONTINUE.json"))) == 1


def test_periodic_live_checkpoint_survives_infrastructure_failure(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    results = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"

    def crash_after_checkpoint(_config_path: Path, *, raw: dict[str, Any], cfg: Any) -> Any:
        _ = raw, cfg
        _write_trainer_checkpoint(Path(os.environ["MODSSC_CHECKPOINT_ROOT"]), payload=b"step-40")
        raise RuntimeError("worker transport failed")

    with pytest.raises(CampaignError, match="TASK_FAILED"):
        _execute(
            repo=repo,
            campaign=campaign,
            results=results,
            work=tmp_path / "failed-work",
            checkpoints=checkpoints,
            runner=crash_after_checkpoint,
        )

    success = FakeRunner()

    def resume(config_path: Path, *, raw: dict[str, Any], cfg: Any) -> Any:
        assert os.environ["MODSSC_CHECKPOINT_RESUME"] == "1"
        checkpoint_root = Path(os.environ["MODSSC_CHECKPOINT_ROOT"])
        assert (checkpoint_root / "checkpoint.pt").read_bytes() == b"step-40"
        return success(config_path, raw=raw, cfg=cfg)

    completed = _execute(
        repo=repo,
        campaign=campaign,
        results=results,
        work=tmp_path / "resume-work",
        checkpoints=checkpoints,
        runner=resume,
    )
    assert completed.status == "success"


def test_crash_before_first_checkpoint_reuses_empty_live_workspace(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    task = tasks[0]
    results = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"

    def crash_before_checkpoint(_config_path: Path, *, raw: dict[str, Any], cfg: Any) -> Any:
        _ = raw, cfg
        workspace = Path(os.environ["MODSSC_CHECKPOINT_ROOT"])
        assert os.environ["MODSSC_CHECKPOINT_RESUME"] == "0"
        assert workspace.is_dir()
        assert not any(workspace.iterdir())
        raise RuntimeError("worker crashed before its first checkpoint")

    with pytest.raises(CampaignError, match="TASK_FAILED"):
        _execute(
            repo=repo,
            campaign=campaign,
            results=results,
            work=tmp_path / "failed-work",
            checkpoints=checkpoints,
            runner=crash_before_checkpoint,
        )

    task_dir = checkpoints / "tasks" / task.task_id[:2] / task.task_id
    live = task_dir / "live"
    assert live.is_dir()
    assert not any(live.iterdir())

    success = FakeRunner()

    def restart_without_checkpoint(config_path: Path, *, raw: dict[str, Any], cfg: Any) -> Any:
        assert os.environ["MODSSC_CHECKPOINT_RESUME"] == "0"
        assert Path(os.environ["MODSSC_CHECKPOINT_ROOT"]) == live
        assert not any(live.iterdir())
        return success(config_path, raw=raw, cfg=cfg)

    completed = _execute(
        repo=repo,
        campaign=campaign,
        results=results,
        work=tmp_path / "restart-work",
        checkpoints=checkpoints,
        runner=restart_without_checkpoint,
    )
    assert completed.status == "success"


def test_tampered_live_and_sealed_checkpoint_is_rejected(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path)
    _, tasks = load_manifest(campaign / "manifest.jsonl")
    task = tasks[0]
    results = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"
    runner = _ContinueThenSucceed()
    _execute(
        repo=repo,
        campaign=campaign,
        results=results,
        work=tmp_path / "first-work",
        checkpoints=checkpoints,
        runner=runner,
    )
    task_dir = checkpoints / "tasks" / task.task_id[:2] / task.task_id
    latest = json.loads((task_dir / "LATEST.json").read_text(encoding="utf-8"))
    (task_dir / "live" / "checkpoint.pt").write_bytes(b"tampered-live")
    snapshot = task_dir / "snapshots" / latest["payload_sha256"]
    (snapshot / "payload" / "checkpoint.pt").write_bytes(b"tampered-snapshot")

    with pytest.raises(CampaignError, match="CHECKPOINT_INVALID"):
        _execute(
            repo=repo,
            campaign=campaign,
            results=results,
            work=tmp_path / "second-work",
            checkpoints=checkpoints,
            runner=runner,
        )
