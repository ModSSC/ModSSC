from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from bench.campaign.attempts import seal_attempt_record
from bench.campaign.manifest import (
    finalize_task_row,
    load_manifest,
    sha256_file,
    write_manifest,
)
from bench.utils.io import atomic_write_json
from tests.bench.campaign.helpers import build_test_campaign
from tools.hpc.match_continuation_controller import (
    ControllerBusy,
    ControllerConfig,
    ControllerError,
    SlurmScheduler,
    _ControllerLock,
    _validate_continuation_campaign,
    bootstrap_controller,
    run_controller,
)
from tools.hpc.resources import plan_resource_sites
from tools.hpc.slurm_renderer import render_slurm_sites


class _FakeScheduler(SlurmScheduler):
    def __init__(self, *, recovered: dict[str, str] | None = None) -> None:
        self.commands: list[list[str]] = []
        self.lookups: list[str] = []
        self.recovered = recovered or {}
        self.counter = 8000

    def find(self, job_name: str) -> str | None:
        self.lookups.append(job_name)
        return self.recovered.get(job_name)

    def submit(self, command: list[str]) -> str:
        assert command[0] == "sbatch"
        self.commands.append(list(command))
        self.counter += 1
        return str(self.counter)


def _fixture(
    tmp_path: Path,
    *,
    max_segments: int = 4,
    profile_id: str = "h100_long",
) -> tuple[ControllerConfig, Path, Any]:
    repo, _, base = build_test_campaign(tmp_path / "source")
    base_meta, base_tasks = load_manifest(base / "manifest.jsonl")
    payload = base_tasks[0].to_dict()
    for field in ("task_id", "task_index", "output_relpath", "row_sha256"):
        payload.pop(field)
    payload.update(
        {
            "track": "paper",
            "protocol_id": "sohn-2020-cifar10-table2-250",
            "method_id": "fixmatch",
            "resource_profile": profile_id,
            "assigned_site": "slurm-gpu",
            "required_seed_count": 1,
        }
    )
    task = finalize_task_row(payload, task_index=0)
    campaign = tmp_path / "campaign"
    write_manifest(
        [task],
        output_dir=campaign,
        campaign_id=task.campaign_id,
        spec_sha256=str(base_meta["spec_sha256"]),
        expected_git_sha=task.expected_git_sha,
        expected_git_diff_sha256=task.expected_git_diff_sha256,
        environment_lock_sha256=task.environment_lock_sha256,
    )
    site_path = tmp_path / "site.yaml"
    site_payload = {
        "schema_version": 1,
        "site_id": "slurm-gpu",
        "scheduler": "slurm",
        "environment_lock_sha256": "from_environment",
        "setup": [],
        "profiles": {
            "h100_dev": {
                "architecture": "H100",
                "accelerators_per_task": 1,
                "fixed_walltime": True,
                "concurrency": 1,
                "initial_concurrency": 1,
                "max_walltime": "02:00:00",
                "directives": {
                    "account": "test@h100",
                    "constraint": "h100",
                    "nodes": 1,
                    "ntasks": 1,
                    "gres": "gpu:1",
                    "cpus-per-task": 2,
                    "time": "02:00:00",
                    "qos": "test-dev",
                },
            },
            "h100_long": {
                "architecture": "H100",
                "accelerators_per_task": 1,
                "fixed_walltime": True,
                "concurrency": 5,
                "initial_concurrency": 5,
                "max_walltime": "100:00:00",
                "setup": ["export MODSSC_PLANNED_SEGMENT_SECONDS=288000"],
                "directives": {
                    "account": "test@h100",
                    "constraint": "h100",
                    "nodes": 1,
                    "ntasks": 1,
                    "gres": "gpu:1",
                    "cpus-per-task": 2,
                    "time": "100:00:00",
                    "signal": "B:USR1@300",
                    "qos": "test-long",
                },
            },
            "h100_t3_adaptive": {
                "architecture": "H100",
                "accelerators_per_task": 1,
                "fixed_walltime": True,
                "concurrency": 9,
                "initial_concurrency": 9,
                "promoted_concurrency": 9,
                "max_walltime": "20:00:00",
                "setup": ["export MODSSC_PLANNED_SEGMENT_SECONDS=68400"],
                "directives": {
                    "account": "test@h100",
                    "constraint": "h100",
                    "nodes": 1,
                    "ntasks": 1,
                    "gres": "gpu:1",
                    "cpus-per-task": 2,
                    "time": "20:00:00",
                    "signal": "B:USR1@300",
                    "qos": "test-t3",
                },
            },
        },
    }
    site_path.write_text(yaml.safe_dump(site_payload), encoding="utf-8")
    plan_resource_sites(site_paths=[site_path], tasks=[task], campaign_dir=campaign)
    render_slurm_sites(
        site_paths=[site_path],
        campaign_dir=campaign,
    )
    allocation = tmp_path / "allocation.yaml"
    allocation.write_text("schema_version: 1\n", encoding="utf-8")
    environment = tmp_path / "environment.json"
    environment.write_text("{}\n", encoding="utf-8")
    checkpoint_base = tmp_path / "checkpoints"
    result_root = tmp_path / "results" / task.campaign_id
    config = ControllerConfig.build(
        repo_root=Path.cwd(),
        campaign_dir=campaign,
        result_root=result_root,
        state_dir=tmp_path / "controller",
        site_path=site_path,
        allocation_path=allocation,
        environment_manifest_path=environment,
        checkpoint_base=checkpoint_base,
        max_segments=max_segments,
        controller_profile="h100_dev",
    )
    return config, result_root, task


def _continuation_attempt(result_root: Path, task: Any, *, name: str = "attempt-1") -> None:
    attempt = result_root / "attempts" / task.task_id[:2] / task.task_id / name
    attempt.mkdir(parents=True)
    atomic_write_json(
        attempt / "attempt.json",
        seal_attempt_record(
            {
                "task_id": task.task_id,
                "row_sha256": task.row_sha256,
                "attempt_id": name,
                "status": "continuation",
                "site_id": task.assigned_site,
                "finished_at": "2026-07-25T00:00:00+00:00",
                "event_class": "planned_continuation",
                "failure_class": None,
                "retryable": False,
                "resource_change_required": False,
                "signal_number": 10,
                "checkpoint_payload_sha256": "a" * 64,
                "checkpoint_manifest_sha256": "b" * 64,
                "checkpoint_reference": (
                    f"checkpoint://tasks/{task.task_id[:2]}/{task.task_id}/CONTINUE.json"
                ),
                "scheduler": {},
            }
        ),
    )


def _write_config(config: ControllerConfig) -> Path:
    state_dir = Path(config.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "controller-config.json"
    atomic_write_json(path, config.to_dict())
    return path


def _success_result(result_root: Path, task: Any) -> None:
    result_dir = result_root / task.output_relpath
    run_dir = result_dir / "run"
    replay_dir = run_dir / "sampling_split"
    replay_dir.mkdir(parents=True)
    split_fingerprint = task.expected_split_fingerprint or "split-fingerprint"
    atomic_write_json(
        replay_dir / "split.json",
        {
            "dataset_fingerprint": task.expected_dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
        },
    )
    (replay_dir / "arrays.npz").write_bytes(b"immutable-arrays")
    files = {
        name: {"sha256": sha256_file(replay_dir / name)} for name in ("split.json", "arrays.npz")
    }
    atomic_write_json(
        replay_dir / "MANIFEST.json",
        {
            "schema_version": 1,
            "format": "modssc.sampling.storage.v1",
            "dataset_fingerprint": task.expected_dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
            "files": files,
        },
    )
    atomic_write_json(
        run_dir / "run.json",
        {
            "run": {
                "name": "match-controller-test",
                "seed": task.seed,
                "run_id": "controller-test",
                "started_at": "2026-07-25T00:00:00+00:00",
                "finished_at": "2026-07-25T00:00:01+00:00",
                "status": "success",
                "benchmark_mode": True,
                "config_path": "effective.yaml",
                "error_code": None,
            },
            "hashes": {"config_hash": "a", "effective_config_hash": "b"},
            "resolution": {
                "device": {"requested": "cuda", "resolved": "cuda"},
                "backend": {"requested": {}, "resolved": {}},
                "dtype": {"requested": {}, "resolved": {}},
                "normalization": {"requested": {}, "resolved": {}},
                "splits": {"requested": ["test"], "resolved": {}},
                "limits": {"requested": None, "resolved": None, "changes": []},
            },
            "protocol": {
                "kind": task.method_kind,
                "use_test_split": True,
                "report_splits": ["test"],
                "split_for_model_selection": None,
            },
            "task_info": {
                "method_id": task.method_id,
                "dataset_id": task.dataset_id,
                "method_kind": task.method_kind,
            },
            "versions": {
                "python": "3.12",
                "modssc": "0",
                "numpy": "0",
                "git_sha": task.expected_git_sha,
                "git_dirty": False,
                "git_diff_sha256": task.expected_git_diff_sha256,
            },
            "artifacts": {
                "dataset": {"fingerprint": task.expected_dataset_fingerprint},
                "sampling": {
                    "split_fingerprint": split_fingerprint,
                    "replay": {
                        "format": "modssc.sampling.storage.v1",
                        "path": "sampling_split",
                        "manifest": "MANIFEST.json",
                        "manifest_sha256": sha256_file(replay_dir / "MANIFEST.json"),
                    },
                },
                "method": {"profile": task.method_profile},
            },
            "metrics": {"test": {"accuracy": 0.95}},
            "config": {},
            "hpo": None,
            "fallback_events": [],
            "error": None,
        },
    )
    (result_dir / "effective.yaml").write_text("immutable: true\n", encoding="utf-8")
    atomic_write_json(
        result_dir / "task.json",
        {
            "schema_version": 1,
            "task": task.to_dict(),
            "site_id": task.assigned_site,
            "environment_lock_sha256": task.environment_lock_sha256,
        },
    )
    atomic_write_json(
        result_dir / "SUCCESS.json",
        {
            "schema_version": 1,
            "task_id": task.task_id,
            "row_sha256": task.row_sha256,
            "status": "success",
            "run_json_sha256": sha256_file(run_dir / "run.json"),
            "effective_config_path": "effective.yaml",
            "effective_config_sha256": sha256_file(result_dir / "effective.yaml"),
            "dataset_content_sha256": task.expected_dataset_content_sha256,
        },
    )


def _current_snapshot(config: ControllerConfig) -> dict[str, Any]:
    state_dir = Path(config.state_dir)
    current = json.loads((state_dir / "CURRENT.json").read_text(encoding="utf-8"))
    return json.loads((state_dir / current["snapshot"]).read_text(encoding="utf-8"))


def test_bootstrap_and_continuation_chain_are_idempotent(tmp_path: Path) -> None:
    config, result_root, task = _fixture(tmp_path)
    scheduler = _FakeScheduler()
    bootstrap = bootstrap_controller(
        config=config,
        after_job_id="7000",
        scheduler=scheduler,
    )

    assert bootstrap["controller_job_id"] == "8001"
    assert "--dependency=afterany:7000" in scheduler.commands[0]
    repeated_bootstrap = bootstrap_controller(
        config=config,
        after_job_id="7000",
        scheduler=scheduler,
    )
    assert repeated_bootstrap["controller_job_id"] == "8001"
    assert len(scheduler.commands) == 1
    with pytest.raises(ControllerError, match="different initial job"):
        bootstrap_controller(
            config=config,
            after_job_id="7001",
            scheduler=scheduler,
        )
    config_path = Path(bootstrap["config_path"])
    _continuation_attempt(result_root, task)

    result = run_controller(
        config_path,
        segment_index=1,
        scheduler=scheduler,
        require_slurm=False,
    )

    assert result == {
        "status": "scheduled",
        "segment_index": 2,
        "preflight_job_id": "8002",
        "array_job_id": "8003",
        "controller_job_id": "8004",
    }
    assert len(scheduler.commands) == 4
    preflight, array, controller = scheduler.commands[1:]
    assert "--time=00:30:00" in preflight
    assert not any(value.startswith("--dependency=") for value in preflight)
    assert "--dependency=afterok:8002" in array
    assert not any(value.startswith("--time=") for value in array)
    assert not any(value.startswith("--gres=") for value in array)
    assert (
        f"--chdir={Path(config.state_dir) / 'reconciliations' / 'segment-001-attempt-001' / 'continuation-campaign'}"
        in array
    )
    array_export = next(value for value in array if value.startswith("--export="))
    assert f"MODSSC_ROOT={config.repo_root}" in array_export
    assert f"MODSSC_CAMPAIGN_CHECKPOINTS={config.checkpoint_base}" in array_export
    assert f"MODSSC_CAMPAIGN_RESULTS={result_root.parent}" in array_export
    assert "MODSSC_PREFLIGHT_EXPIRY_POLICY=generated_by_dependency" in array_export
    assert "MODSSC_PREFLIGHT_JOB_ID=8002" in array_export
    assert "--dependency=afterany:8003" in controller

    continuation = (
        Path(config.state_dir)
        / "reconciliations"
        / "segment-001-attempt-001"
        / "continuation-campaign"
    )
    _, continuation_tasks = load_manifest(continuation / "manifest.jsonl")
    assert [item.to_dict() for item in continuation_tasks] == [task.to_dict()]

    repeated = run_controller(
        config_path,
        segment_index=1,
        scheduler=scheduler,
        require_slurm=False,
    )
    assert repeated["status"] == "scheduled"
    assert len(scheduler.commands) == 4
    snapshot = _current_snapshot(config)
    assert snapshot["state"]["status"] == "scheduled"
    assert snapshot["state"]["last_observed_segment"] == 1
    assert snapshot["state"]["active_submission"]["pending_task_ids"] == [task.task_id]
    assert snapshot["journal"][-1]["event"] == "continuation_scheduled"

    operation = Path.cwd() / "tools/hpc/slurm/run-operation.sh"
    payload = operation.read_text(encoding="utf-8")
    assert "SLURMD_NODENAME:?" in payload
    assert '"$MODSSC_PYTHON"' in payload
    assert "continuation)" in payload
    assert "-m tools.hpc.match_continuation_controller run" in payload
    assert operation.stat().st_mode & 0o111


def test_controller_accepts_segmented_h100_t3_profile(tmp_path: Path) -> None:
    config, result_root, task = _fixture(tmp_path, profile_id="h100_t3_adaptive")
    config_path = _write_config(config)
    _continuation_attempt(result_root, task)
    scheduler = _FakeScheduler()

    result = run_controller(
        config_path,
        segment_index=1,
        scheduler=scheduler,
        require_slurm=False,
    )

    assert result["status"] == "scheduled"
    wrapper = (
        Path(config.state_dir)
        / "submissions"
        / "segment-002"
        / "slurm-gpu"
        / "h100_t3_adaptive.slurm"
    )
    text = wrapper.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-0%9" in text
    assert "#SBATCH --time=20:00:00" in text
    assert "#SBATCH --qos=test-t3" in text
    assert "export MODSSC_PLANNED_SEGMENT_SECONDS=68400" in text


def test_controller_blocks_noncontinuation_state_without_submitting(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    scheduler = _FakeScheduler()
    config_path = _write_config(config)

    result = run_controller(
        config_path,
        segment_index=1,
        scheduler=scheduler,
        require_slurm=False,
    )

    assert result["status"] == "blocked"
    assert result["statuses"] == ["missing"]
    assert scheduler.commands == []
    assert _current_snapshot(config)["state"]["status"] == "blocked"


def test_controller_stops_at_explicit_segment_limit(tmp_path: Path) -> None:
    config, result_root, task = _fixture(tmp_path, max_segments=2)
    config_path = _write_config(config)
    _continuation_attempt(result_root, task)
    scheduler = _FakeScheduler()
    first = run_controller(
        config_path,
        segment_index=1,
        scheduler=scheduler,
        require_slurm=False,
    )
    assert first["status"] == "scheduled"
    _continuation_attempt(result_root, task, name="attempt-2")

    second = run_controller(
        config_path,
        segment_index=2,
        scheduler=scheduler,
        require_slurm=False,
    )

    assert second["status"] == "max_segments_exceeded"
    assert len(scheduler.commands) == 3
    assert _current_snapshot(config)["state"]["status"] == "max_segments_exceeded"


def test_continuation_refuses_tampered_gpu_contract(tmp_path: Path) -> None:
    config, result_root, task = _fixture(tmp_path)
    config_path = _write_config(config)
    _continuation_attempt(result_root, task)
    run_controller(
        config_path,
        segment_index=1,
        scheduler=_FakeScheduler(),
        require_slurm=False,
    )
    continuation = (
        Path(config.state_dir)
        / "reconciliations"
        / "segment-001-attempt-001"
        / "continuation-campaign"
    )
    wrapper = next((Path(config.state_dir) / "submissions" / "segment-002").glob("*/*.slurm"))
    wrapper.write_text(
        wrapper.read_text(encoding="utf-8").replace(
            "#SBATCH --gres=gpu:1",
            "#SBATCH --gres=gpu:2",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ControllerError, match="fixed Match Slurm contract"):
        _validate_continuation_campaign(
            config,
            continuation_dir=continuation,
            pending_task_ids={task.task_id},
            wrapper_paths=[wrapper],
        )

    wrapper.write_text(
        wrapper.read_text(encoding="utf-8").replace(
            "#SBATCH --gres=gpu:2",
            "#SBATCH --gres=gpu:1",
        ),
        encoding="utf-8",
    )
    resources_path = continuation / "profiles" / "resources.json"
    resources = json.loads(resources_path.read_text(encoding="utf-8"))
    resources["resources"][0]["fixed_walltime"] = False
    resources_path.write_text(json.dumps(resources), encoding="utf-8")
    with pytest.raises(ControllerError, match="fixed mono-H100"):
        _validate_continuation_campaign(
            config,
            continuation_dir=continuation,
            pending_task_ids={task.task_id},
        )


def test_controller_does_not_count_a_cancelled_array_as_a_segment(
    tmp_path: Path,
) -> None:
    config, result_root, task = _fixture(tmp_path)
    config_path = _write_config(config)
    _continuation_attempt(result_root, task)
    scheduler = _FakeScheduler()
    run_controller(
        config_path,
        segment_index=1,
        scheduler=scheduler,
        require_slurm=False,
    )

    stalled = run_controller(
        config_path,
        segment_index=2,
        scheduler=scheduler,
        require_slurm=False,
    )

    assert stalled == {
        "status": "blocked",
        "segment_index": 2,
        "reason": "segment_made_no_progress",
        "task_ids": [task.task_id],
    }
    assert len(scheduler.commands) == 3


def test_controller_finishes_after_a_valid_resumed_result(tmp_path: Path) -> None:
    config, result_root, task = _fixture(tmp_path)
    config_path = _write_config(config)
    _continuation_attempt(result_root, task)
    scheduler = _FakeScheduler()
    run_controller(
        config_path,
        segment_index=1,
        scheduler=scheduler,
        require_slurm=False,
    )
    _success_result(result_root, task)

    completed = run_controller(
        config_path,
        segment_index=2,
        scheduler=scheduler,
        require_slurm=False,
    )
    repeated = run_controller(
        config_path,
        segment_index=2,
        scheduler=scheduler,
        require_slurm=False,
    )

    assert completed == {"status": "complete", "segment_index": 2}
    assert repeated == completed
    assert len(scheduler.commands) == 3
    assert _current_snapshot(config)["state"]["status"] == "complete"


def test_submission_intent_recovers_named_job_instead_of_duplicating(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    scheduler = _FakeScheduler(
        recovered={
            f"msc-{config.controller_id[:12]}-s001-controller": "7777",
        }
    )

    result = bootstrap_controller(
        config=config,
        after_job_id="7000",
        scheduler=scheduler,
    )

    assert result["controller_job_id"] == "7777"
    assert scheduler.commands == []
    record = _current_snapshot(config)["state"]["bootstrap"]["controller"]
    assert record == {
        "job_name": f"msc-{config.controller_id[:12]}-s001-controller",
        "job_id": "7777",
        "recovered": True,
    }


@pytest.mark.parametrize("job_id", ["0", "abc", "1:2"])
def test_bootstrap_rejects_invalid_dependency_id(tmp_path: Path, job_id: str) -> None:
    config, _, _ = _fixture(tmp_path)
    with pytest.raises(ControllerError, match="positive Slurm job id"):
        bootstrap_controller(
            config=config,
            after_job_id=job_id,
            scheduler=_FakeScheduler(),
        )


def test_config_digest_and_campaign_binding_fail_closed(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    config_path = _write_config(config)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["max_segments"] = 99
    atomic_write_json(config_path, payload)
    with pytest.raises(ControllerError, match="digest mismatch"):
        ControllerConfig.load(config_path)

    with pytest.raises(ControllerError, match="greater than or equal to 2"):
        ControllerConfig.build(
            repo_root=Path(config.repo_root),
            campaign_dir=Path(config.campaign_dir),
            result_root=Path(config.result_root),
            state_dir=tmp_path / "other",
            site_path=Path(config.site_path),
            allocation_path=Path(config.allocation_path),
            environment_manifest_path=Path(config.environment_manifest_path),
            checkpoint_base=Path(config.checkpoint_base),
            max_segments=1,
            controller_profile="h100_dev",
        )

    pinned, _, _ = _fixture(tmp_path / "pinned")
    pinned_path = _write_config(pinned)
    Path(pinned.site_path).write_text("schema_version: 999\n", encoding="utf-8")
    with pytest.raises(ControllerError, match="site profile digest mismatch"):
        ControllerConfig.load(pinned_path)

    source_pinned, _, _ = _fixture(tmp_path / "source-pinned")
    source_pinned_path = _write_config(source_pinned)
    manifest_path = Path(source_pinned.campaign_dir) / "manifest.jsonl"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ControllerError, match="source campaign manifest digest mismatch"):
        ControllerConfig.load(source_pinned_path)


def test_run_requires_a_compute_allocation_and_strict_segment_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _fixture(tmp_path)
    config_path = _write_config(config)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)
    with pytest.raises(ControllerError, match="inside a Slurm allocation"):
        run_controller(config_path, segment_index=1, scheduler=_FakeScheduler())
    with pytest.raises(ControllerError, match="positive integer"):
        run_controller(
            config_path,
            segment_index=0,
            scheduler=_FakeScheduler(),
            require_slurm=False,
        )
    with pytest.raises(ControllerError, match="sequence contains a gap"):
        run_controller(
            config_path,
            segment_index=2,
            scheduler=_FakeScheduler(),
            require_slurm=False,
        )


def test_lock_and_state_digest_prevent_concurrent_or_tampered_control(
    tmp_path: Path,
) -> None:
    config, _, _ = _fixture(tmp_path)
    state_dir = Path(config.state_dir)
    with _ControllerLock(state_dir), pytest.raises(ControllerBusy, match="owns the lock"):
        _ControllerLock(state_dir).__enter__()

    scheduler = _FakeScheduler()
    bootstrap_controller(config=config, after_job_id="7000", scheduler=scheduler)
    current = json.loads((state_dir / "CURRENT.json").read_text(encoding="utf-8"))
    snapshot_path = state_dir / current["snapshot"]
    snapshot_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ControllerError, match="snapshot digest mismatch"):
        run_controller(
            state_dir / "controller-config.json",
            segment_index=1,
            scheduler=scheduler,
            require_slurm=False,
        )


def test_state_pointer_and_reconcile_report_fail_closed(tmp_path: Path) -> None:
    missing_config, _, _ = _fixture(tmp_path / "missing-current")
    bootstrap_controller(
        config=missing_config,
        after_job_id="7000",
        scheduler=_FakeScheduler(),
    )
    Path(missing_config.state_dir, "CURRENT.json").unlink()
    with pytest.raises(ControllerError, match="CURRENT pointer is missing"):
        bootstrap_controller(
            config=missing_config,
            after_job_id="7000",
            scheduler=_FakeScheduler(),
        )

    config, result_root, task = _fixture(tmp_path / "report")
    config_path = _write_config(config)
    _continuation_attempt(result_root, task)
    scheduler = _FakeScheduler()
    run_controller(
        config_path,
        segment_index=1,
        scheduler=scheduler,
        require_slurm=False,
    )
    report_path = (
        Path(config.state_dir) / "reconciliations" / "segment-001-attempt-001" / "reconcile.json"
    )
    report_path.write_text(
        report_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ControllerError, match="reconcile report digest mismatch"):
        run_controller(
            config_path,
            segment_index=1,
            scheduler=scheduler,
            require_slurm=False,
        )


def test_slurm_scheduler_recovers_and_parses_simulated_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = SlurmScheduler(environment={"PATH": "/fake"})
    responses = iter(
        (
            subprocess.CompletedProcess(["squeue"], 1, "", "not available"),
            subprocess.CompletedProcess(["sacct"], 0, "9100\n9100.batch\n", ""),
            subprocess.CompletedProcess(["sbatch"], 0, "9200;test-cluster\n", ""),
        )
    )
    monkeypatch.setattr(scheduler, "_run", lambda _command: next(responses))

    assert scheduler.find("deterministic-name") == "9100"
    assert scheduler.submit(["sbatch", "--parsable", "payload.sh"]) == "9200"


@pytest.mark.parametrize(
    ("completed", "message"),
    [
        (
            subprocess.CompletedProcess(["sbatch"], 1, "", "policy rejected"),
            "policy rejected",
        ),
        (
            subprocess.CompletedProcess(["sbatch"], 0, "Submitted batch job 42\n", ""),
            "invalid job id",
        ),
    ],
)
def test_slurm_scheduler_rejects_simulated_submission_errors(
    monkeypatch: pytest.MonkeyPatch,
    completed: subprocess.CompletedProcess[str],
    message: str,
) -> None:
    scheduler = SlurmScheduler()
    monkeypatch.setattr(scheduler, "_run", lambda _command: completed)
    with pytest.raises(ControllerError, match=message):
        scheduler.submit(["sbatch", "--parsable", "payload.sh"])


def test_slurm_scheduler_rejects_ambiguous_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = SlurmScheduler()
    completed = subprocess.CompletedProcess(["squeue"], 0, "9100\n9200\n", "")
    monkeypatch.setattr(scheduler, "_run", lambda _command: completed)
    with pytest.raises(ControllerError, match="multiple Slurm jobs"):
        scheduler.find("duplicated-name")


@pytest.mark.parametrize(
    "responses",
    [
        (
            subprocess.CompletedProcess(["squeue"], 1, "", "unavailable"),
            subprocess.CompletedProcess(["sacct"], 1, "", "unavailable"),
        ),
        (
            subprocess.CompletedProcess(["squeue"], 0, "", ""),
            subprocess.CompletedProcess(["sacct"], 1, "", "unavailable"),
        ),
        (
            subprocess.CompletedProcess(["squeue"], 1, "", "unavailable"),
            subprocess.CompletedProcess(["sacct"], 0, "", ""),
        ),
    ],
)
def test_slurm_scheduler_fails_closed_when_recovery_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
    responses: tuple[subprocess.CompletedProcess[str], ...],
) -> None:
    scheduler = SlurmScheduler()
    completed = iter(responses)
    monkeypatch.setattr(scheduler, "_run", lambda _command: next(completed))

    with pytest.raises(ControllerError, match="cannot safely recover"):
        scheduler.find("possibly-submitted-job")
