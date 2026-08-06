from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from bench.campaign.cli import main
from bench.campaign.errors import CampaignError
from bench.campaign.generate import generate_campaign
from bench.campaign.manifest import finalize_task_row, load_manifest, write_manifest
from bench.campaign.models import TaskExecutionResult
from tools.hpc.resources import format_duration, parse_duration, plan_resource_sites
from tools.hpc.scheduler_failure import main as scheduler_failure_main
from tools.hpc.slurm_renderer import render_slurm_sites

from .helpers import build_test_campaign


def test_public_campaign_cli_omits_operational_resource_commands(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    commands = " ".join(output.split())
    assert "daily-report" not in commands
    assert "preflight" not in commands
    assert "{'option_strings'" not in output


def test_run_task_cli_uses_distinct_planned_continuation_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "bench.campaign.cli.execute_task",
        lambda *_args, **_kwargs: TaskExecutionResult(
            task_id="task",
            status="continuation",
            result_dir=None,
            attempt_dir=str(tmp_path / "attempt"),
        ),
    )

    code = main(
        [
            "run-task",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--index",
            "0",
            "--result-root",
            str(tmp_path / "results"),
            "--work-root",
            str(tmp_path / "work"),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--site-id",
            "local",
        ]
    )

    assert code == 85


def test_slurm_gpu_v100_highmem_profile_renders_distinct_bounded_retry_arrays(
    tmp_path: Path,
) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    _, source_tasks = load_manifest(campaign / "manifest.jsonl")
    source = source_tasks[0].to_dict()
    for key in ("schema_version", "task_index", "task_id", "output_relpath", "row_sha256"):
        source.pop(key)

    def retarget(*, profile: str, seed: int, task_index: int):
        payload = {
            **source,
            "resource_profile": profile,
            "assigned_site": "slurm-gpu",
            "required_seed_count": 11,
            "seed": seed,
            "data_seed": seed,
            "split_seed": seed,
            "sampling_component_seeds": {
                "partition": seed,
                "split": seed,
                "labeling": seed,
                "imbalance": seed,
            },
            "model_seed": seed,
        }
        return finalize_task_row(payload, task_index=task_index)

    standard = retarget(profile="v100_dev", seed=0, task_index=0)
    highmem_tasks = [
        retarget(profile="v100_dev_highmem", seed=seed, task_index=seed) for seed in range(11)
    ]
    assert standard.task_id != highmem_tasks[0].task_id

    repo_root = Path(__file__).resolve().parents[3]
    site_path = repo_root / "tools" / "hpc" / "config" / "profiles" / "slurm.example.yaml"
    site = yaml.safe_load(site_path.read_text(encoding="utf-8"))
    profile = site["profiles"]["v100_dev_highmem"]
    directives = profile["directives"]
    assert profile["concurrency"] == 10
    assert profile["initial_concurrency"] == 10
    assert profile["promoted_concurrency"] == 10
    assert profile["array_block_size"] == 10
    assert profile["max_walltime"] == "02:00:00"
    assert "account" not in directives
    assert "constraint" not in directives
    assert "partition" not in directives
    assert "qos" not in directives
    assert directives["time"] == "02:00:00"
    assert directives["mem"] == "80G"

    meta, _ = load_manifest(campaign / "manifest.jsonl")
    write_manifest(
        highmem_tasks,
        output_dir=campaign,
        campaign_id=highmem_tasks[0].campaign_id,
        spec_sha256=str(meta["spec_sha256"]),
        expected_git_sha=str(meta["expected_git_sha"]),
        expected_git_diff_sha256=meta["expected_git_diff_sha256"],
        environment_lock_sha256=str(meta["environment_lock_sha256"]),
    )
    plan_resource_sites(site_paths=[site_path], tasks=highmem_tasks, campaign_dir=campaign)
    scripts = render_slurm_sites(site_paths=[site_path], campaign_dir=campaign)
    assert [path.name for path in scripts] == [
        "v100_dev_highmem.block000.slurm",
        "v100_dev_highmem.block001.slurm",
    ]
    first_script = scripts[0].read_text(encoding="utf-8")
    second_script = scripts[1].read_text(encoding="utf-8")
    assert "#SBATCH --array=0-9%10" in first_script
    assert "#SBATCH --array=0-0%10" in second_script
    assert "#SBATCH --mem=80G" in first_script
    assert "export MODSSC_RESOURCE_PROFILE=v100_dev_highmem" in first_script

    resources = json.loads((campaign / "profiles" / "resources.json").read_text(encoding="utf-8"))
    assert resources["resources"][0]["profile_id"] == "v100_dev_highmem"
    assert [entry["task_count"] for entry in resources["array_indices"]] == [10, 1]


def test_hpc_renderer_creates_one_seed_array_wrapper(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path, with_site=True)
    script = campaign / "submit" / "local" / "cpu_test.slurm"
    text = script.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-1%2" in text
    assert "SLURM_JOB_ID:?" in text
    assert "SLURMD_NODENAME:?" in text
    assert "array-task.sh" in text
    assert "MODSSC_CAMPAIGN_CHECKPOINT_ROOT" in text
    assert "MODSSC_CAMPAIGN_CHECKPOINTS" in text
    assert "${MODSSC_CAMPAIGN_CHECKPOINT_ROOT:-" not in text
    assert "for " not in text
    index_path = campaign / "profiles" / "local.cpu_test.indices"
    assert index_path.read_text() == "0\n1\n"
    resources = json.loads((campaign / "profiles" / "resources.json").read_text(encoding="utf-8"))
    assert (
        resources["manifest_sha256"]
        == hashlib.sha256((campaign / "manifest.jsonl").read_bytes()).hexdigest()
    )
    assert resources["array_indices"] == [
        {
            "block": 0,
            "path": "profiles/local.cpu_test.indices",
            "profile_id": "cpu_test",
            "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "site_id": "local",
            "task_count": 2,
        }
    ]
    assert f"export MODSSC_ARRAY_INDEX_SHA256={resources['array_indices'][0]['sha256']}" in text
    assert f"export MODSSC_CAMPAIGN_MANIFEST_SHA256={resources['manifest_sha256']}" in text
    assert resources["resources"] == [
        {
            "accelerators_per_task": 0,
            "architecture": "CPU",
            "configured_walltime_seconds": 600,
            "initial_concurrency": 2,
            "max_walltime_seconds": 600,
            "profile_id": "cpu_test",
            "promoted_concurrency": 2,
            "promotion_max_failure_rate": 0.02,
            "promotion_min_successes": 200,
            "site_id": "local",
        }
    ]


def test_local_site_generates_resource_catalog_without_slurm_artifacts(tmp_path) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base")
    site_path = repo / "local-site.yaml"
    site_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "site_id": "local",
                "scheduler": "local",
                "environment_lock_sha256": "unlocked",
                "profiles": {
                    "cpu_test": {
                        "architecture": "CPU",
                        "accelerators_per_task": 0,
                        "concurrency": 2,
                        "initial_concurrency": 2,
                        "promoted_concurrency": 2,
                        "walltime": "00:10:00",
                        "max_walltime": "00:10:00",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "local-campaign"

    generate_campaign(
        repo / "campaign.yaml",
        repo_root=repo,
        output_dir=output,
    )
    render_slurm_sites(site_paths=[site_path], campaign_dir=output)

    assert not (output / "submit").exists()
    assert not list(output.rglob("*.slurm"))
    resources = json.loads((output / "profiles/resources.json").read_text(encoding="utf-8"))
    assert resources["array_indices"] == []
    assert resources["resources"] == [
        {
            "accelerators_per_task": 0,
            "architecture": "CPU",
            "configured_walltime_seconds": 600,
            "initial_concurrency": 2,
            "max_walltime_seconds": 600,
            "profile_id": "cpu_test",
            "promoted_concurrency": 2,
            "promotion_max_failure_rate": 0.02,
            "promotion_min_successes": 200,
            "site_id": "local",
        }
    ]


@pytest.mark.parametrize(
    ("architecture", "accelerators", "message"),
    [
        ("A100", 0, "requires architecture=CPU"),
        ("CPU", 1, "requires accelerators_per_task=0"),
    ],
)
def test_local_site_rejects_accelerator_resources(
    tmp_path, architecture: str, accelerators: int, message: str
) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base")
    site_path = repo / "invalid-local-site.yaml"
    site_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "site_id": "local",
                "scheduler": "local",
                "profiles": {
                    "cpu_test": {
                        "architecture": architecture,
                        "accelerators_per_task": accelerators,
                        "concurrency": 2,
                        "max_walltime": "00:10:00",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(CampaignError, match=message):
        output = tmp_path / "invalid-local"
        generate_campaign(repo / "campaign.yaml", repo_root=repo, output_dir=output)
        render_slurm_sites(site_paths=[site_path], campaign_dir=output)


def test_array_wrapper_records_scheduler_resource_failures() -> None:
    script = Path("tools/hpc/slurm/array-task.sh").read_text(encoding="utf-8")

    assert "exec python -m bench.campaign run-task" not in script
    assert "trap handle_usr1 USR1" in script
    assert "trap handle_term TERM" in script
    assert '--checkpoint-root "$MODSSC_CAMPAIGN_CHECKPOINT_ROOT"' in script
    assert "CHILD_STATUS == 85" in script
    assert "-m tools.hpc.scheduler_failure" in script
    assert "resource_timeout" in script
    assert "resource_oom" in script
    assert "command -v sacct" in script
    assert "--format=State%32" in script
    assert "OUT_OF_MEMORY" in script
    assert "TIMEOUT" in script
    assert "MODSSC_PLANNED_SEGMENT_SECONDS" in script
    assert 'kill -USR1 "$PARENT_PID"' in script
    assert "Pinned Slurm execution requires MODSSC_ENVIRONMENT_MANIFEST" in script


def test_generic_slurm_h100_dev_requests_five_minute_continuation_signal() -> None:
    site = yaml.safe_load(
        Path("tools/hpc/config/profiles/slurm.example.yaml").read_text(encoding="utf-8")
    )
    profile = site["profiles"]["h100_dev"]

    assert profile["architecture"] == "H100"
    assert profile["fixed_walltime"] is True
    assert profile["max_walltime"] == "02:00:00"
    assert profile["directives"]["time"] == "02:00:00"
    assert "account" not in profile["directives"]
    assert "partition" not in profile["directives"]
    assert "qos" not in profile["directives"]
    assert "constraint" not in profile["directives"]
    assert profile["directives"]["signal"] == "B:USR1@300"
    long_profile = site["profiles"]["h100_long"]
    assert long_profile["fixed_walltime"] is True
    assert long_profile["directives"]["signal"] == "B:USR1@300"
    assert long_profile["concurrency"] == 5
    assert "export MODSSC_PLANNED_SEGMENT_SECONDS=288000" in long_profile["setup"]
    assert site["profiles"]["h100_long_adaptive"]["concurrency"] == 9


def test_array_wrapper_rejects_invalid_segment_before_starting_training(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "python-calls.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$FAKE_PYTHON_CALL_LOG"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    indices = tmp_path / "indices"
    indices.write_text("0\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("manifest\n", encoding="utf-8")
    repository = Path.cwd()
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_PYTHON_CALL_LOG": str(call_log),
        "MODSSC_ROOT": str(repository),
        "MODSSC_PYTHON": str(fake_python),
        "MODSSC_CAMPAIGN_MANIFEST": str(manifest),
        "MODSSC_CAMPAIGN_MANIFEST_SHA256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "MODSSC_CAMPAIGN_META": str(tmp_path / "manifest.meta.json"),
        "MODSSC_ARRAY_INDEX_FILE": str(indices),
        "MODSSC_ARRAY_INDEX_SHA256": hashlib.sha256(indices.read_bytes()).hexdigest(),
        "MODSSC_CAMPAIGN_RESULT_ROOT": str(tmp_path / "results"),
        "MODSSC_CAMPAIGN_SITE_ID": "local",
        "MODSSC_CAMPAIGN_ID": "test-campaign",
        "MODSSC_PREFLIGHT_REPORT": str(tmp_path / "preflight.json"),
        "MODSSC_PLANNED_SEGMENT_SECONDS": "invalid",
        "SLURM_JOB_ID": "9090_0",
        "SLURMD_NODENAME": subprocess.check_output(["hostname", "-s"], text=True).strip(),
        "SLURM_ARRAY_TASK_ID": "0",
        "JOBSCRATCH": str(tmp_path / "job-scratch"),
    }

    completed = subprocess.run(
        ["bash", str(repository / "tools/hpc/slurm/array-task.sh")],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 64
    assert "must be an integer greater than 300" in completed.stderr
    assert not call_log.exists()


def test_array_wrapper_converts_sacct_oom_to_scheduler_attempt(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "python-calls.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$*" >> "$FAKE_PYTHON_CALL_LOG"\n'
        'if [[ "$*" == *"run-task"* ]]; then exit 137; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_sacct = fake_bin / "sacct"
    fake_sacct.write_text("#!/bin/bash\nprintf 'OUT_OF_MEMORY|\\n'\n", encoding="utf-8")
    fake_sacct.chmod(0o755)
    indices = tmp_path / "indices"
    indices.write_text("0\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("manifest\n", encoding="utf-8")
    repository = Path.cwd()
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_PYTHON_CALL_LOG": str(call_log),
        "MODSSC_ROOT": str(repository),
        "MODSSC_PYTHON": str(fake_python),
        "MODSSC_CAMPAIGN_MANIFEST": str(manifest),
        "MODSSC_CAMPAIGN_MANIFEST_SHA256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "MODSSC_CAMPAIGN_META": str(tmp_path / "manifest.meta.json"),
        "MODSSC_ARRAY_INDEX_FILE": str(indices),
        "MODSSC_ARRAY_INDEX_SHA256": hashlib.sha256(indices.read_bytes()).hexdigest(),
        "MODSSC_CAMPAIGN_RESULT_ROOT": str(tmp_path / "results"),
        "MODSSC_CAMPAIGN_SITE_ID": "local",
        "MODSSC_CAMPAIGN_ID": "test-campaign",
        "MODSSC_PREFLIGHT_REPORT": str(tmp_path / "preflight.json"),
        "SLURM_JOB_ID": "9191_0",
        "SLURMD_NODENAME": subprocess.check_output(["hostname", "-s"], text=True).strip(),
        "SLURM_ARRAY_JOB_ID": "9191",
        "SLURM_ARRAY_TASK_ID": "0",
        "JOBSCRATCH": str(tmp_path / "job-scratch"),
    }

    completed = subprocess.run(
        ["bash", str(repository / "tools/hpc/slurm/array-task.sh")],
        cwd=tmp_path,
        env=environment,
        check=False,
    )
    calls = call_log.read_text(encoding="utf-8").splitlines()

    assert completed.returncode == 137
    assert sum("run-task" in line for line in calls) == 1
    scheduler_calls = [line for line in calls if "tools.hpc.scheduler_failure" in line]
    assert len(scheduler_calls) == 1
    assert "--failure-class resource_oom" in scheduler_calls[0]
    assert "--scheduler-state OUT_OF_MEMORY|" in scheduler_calls[0]


def test_array_wrapper_forwards_usr1_and_accepts_planned_continuation(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "python-calls.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$*" >> "$FAKE_PYTHON_CALL_LOG"\n'
        'if [[ "$*" == *"run-task"* ]]; then\n'
        "  trap 'exit 85' USR1\n"
        '  kill -USR1 "$PPID"\n'
        "  while true; do sleep 0.01; done\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    indices = tmp_path / "indices"
    indices.write_text("0\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("manifest\n", encoding="utf-8")
    repository = Path.cwd()
    checkpoint_root = tmp_path / "persistent-checkpoints"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_PYTHON_CALL_LOG": str(call_log),
        "MODSSC_ROOT": str(repository),
        "MODSSC_PYTHON": str(fake_python),
        "MODSSC_CAMPAIGN_MANIFEST": str(manifest),
        "MODSSC_CAMPAIGN_MANIFEST_SHA256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "MODSSC_CAMPAIGN_META": str(tmp_path / "manifest.meta.json"),
        "MODSSC_ARRAY_INDEX_FILE": str(indices),
        "MODSSC_ARRAY_INDEX_SHA256": hashlib.sha256(indices.read_bytes()).hexdigest(),
        "MODSSC_CAMPAIGN_RESULT_ROOT": str(tmp_path / "results"),
        "MODSSC_CAMPAIGN_CHECKPOINT_ROOT": str(checkpoint_root),
        "MODSSC_CAMPAIGN_SITE_ID": "local",
        "MODSSC_CAMPAIGN_ID": "test-campaign",
        "MODSSC_PREFLIGHT_REPORT": str(tmp_path / "preflight.json"),
        "SLURM_JOB_ID": "9393_0",
        "SLURMD_NODENAME": subprocess.check_output(["hostname", "-s"], text=True).strip(),
        "SLURM_ARRAY_JOB_ID": "9393",
        "SLURM_ARRAY_TASK_ID": "0",
        "JOBSCRATCH": str(tmp_path / "job-scratch"),
    }

    completed = subprocess.run(
        ["bash", str(repository / "tools/hpc/slurm/array-task.sh")],
        cwd=tmp_path,
        env=environment,
        check=False,
        timeout=10,
    )
    calls = call_log.read_text(encoding="utf-8").splitlines()

    assert completed.returncode == 0
    assert len(calls) == 1
    assert "run-task" in calls[0]
    assert f"--checkpoint-root {checkpoint_root}" in calls[0]
    assert "tools.hpc.scheduler_failure" not in calls[0]


def test_array_wrapper_refuses_tampered_index_before_selecting_task(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "python-calls.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$FAKE_PYTHON_CALL_LOG"\nexit 0\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    indices = tmp_path / "indices"
    indices.write_text("0\n1\n", encoding="utf-8")
    expected_index_digest = hashlib.sha256(indices.read_bytes()).hexdigest()
    indices.write_text("0\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("manifest\n", encoding="utf-8")
    repository = Path.cwd()
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_PYTHON_CALL_LOG": str(call_log),
        "MODSSC_ROOT": str(repository),
        "MODSSC_PYTHON": str(fake_python),
        "MODSSC_CAMPAIGN_MANIFEST": str(manifest),
        "MODSSC_CAMPAIGN_MANIFEST_SHA256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "MODSSC_CAMPAIGN_META": str(tmp_path / "manifest.meta.json"),
        "MODSSC_ARRAY_INDEX_FILE": str(indices),
        "MODSSC_ARRAY_INDEX_SHA256": expected_index_digest,
        "MODSSC_CAMPAIGN_RESULT_ROOT": str(tmp_path / "results"),
        "MODSSC_CAMPAIGN_SITE_ID": "local",
        "MODSSC_CAMPAIGN_ID": "test-campaign",
        "MODSSC_PREFLIGHT_REPORT": str(tmp_path / "preflight.json"),
        "SLURM_JOB_ID": "9292_0",
        "SLURMD_NODENAME": subprocess.check_output(["hostname", "-s"], text=True).strip(),
        "SLURM_ARRAY_TASK_ID": "0",
        "JOBSCRATCH": str(tmp_path / "job-scratch"),
    }

    completed = subprocess.run(
        ["bash", str(repository / "tools/hpc/slurm/array-task.sh")],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 65
    assert "Array index SHA-256 mismatch" in completed.stderr
    assert not call_log.exists()


def test_record_scheduler_failure_cli_is_idempotent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    results = tmp_path / "results"
    monkeypatch.setenv("MODSSC_EXECUTION_JOB_ID", "8181_0")
    monkeypatch.setenv("MODSSC_EXECUTION_ARRAY_JOB_ID", "8181")
    monkeypatch.setenv("MODSSC_EXECUTION_ARRAY_TASK_ID", "0")

    arguments = [
        "--manifest",
        str(campaign / "manifest.jsonl"),
        "--meta",
        str(campaign / "manifest.meta.json"),
        "--index",
        "0",
        "--result-root",
        str(results),
        "--site-id",
        "local",
        "--failure-class",
        "resource_timeout",
        "--scheduler-state",
        "TIMEOUT",
        "--exit-code",
        "143",
    ]
    assert scheduler_failure_main(arguments) == 0
    assert scheduler_failure_main(arguments) == 0


def test_hpc_renderer_splits_arrays_into_bounded_blocks(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path, with_site=True, array_block_size=1)

    scripts = sorted((campaign / "submit" / "local").glob("cpu_test.block*.slurm"))
    assert [path.name for path in scripts] == [
        "cpu_test.block000.slurm",
        "cpu_test.block001.slurm",
    ]
    assert all("#SBATCH --array=0-0%2" in path.read_text(encoding="utf-8") for path in scripts)
    assert (campaign / "profiles" / "local.cpu_test.block000.indices").read_text() == "0\n"
    assert (campaign / "profiles" / "local.cpu_test.block001.indices").read_text() == "1\n"


@pytest.mark.parametrize("unsafe_field", ["site_id", "profile_id", "directive"])
def test_hpc_renderer_rejects_path_and_shell_injection_atomically(
    tmp_path, unsafe_field: str
) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base", with_site=True)
    site_path = repo / "site.yaml"
    site = yaml.safe_load(site_path.read_text(encoding="utf-8"))
    if unsafe_field == "site_id":
        site["site_id"] = "../outside"
    elif unsafe_field == "profile_id":
        site["profiles"]["cpu;touch-pwned"] = site["profiles"].pop("cpu_test")
    else:
        site["profiles"]["cpu_test"]["directives"]["mem"] = "1G\n#SBATCH --account=evil"
    site_path.write_text(yaml.safe_dump(site, sort_keys=False), encoding="utf-8")
    output = tmp_path / f"invalid-{unsafe_field}"

    generate_campaign(repo / "campaign.yaml", repo_root=repo, output_dir=output)
    with pytest.raises(CampaignError, match="SITE_INVALID"):
        render_slurm_sites(site_paths=[site_path], campaign_dir=output)
    assert not (output / "submit").exists()


def test_hpc_renderer_rejects_site_template_values_outside_template_mode(tmp_path) -> None:
    repo, _, _ = build_test_campaign(tmp_path / "base", with_site=True)
    site_path = repo / "site.yaml"
    site = yaml.safe_load(site_path.read_text(encoding="utf-8"))
    site["profiles"]["cpu_test"]["directives"]["account"] = "REPLACE_WITH_ACCOUNT"
    site_path.write_text(yaml.safe_dump(site, sort_keys=False), encoding="utf-8")

    with pytest.raises(CampaignError, match="TEMPLATE_PLACEHOLDER"):
        production = tmp_path / "production"
        generate_campaign(repo / "campaign.yaml", repo_root=repo, output_dir=production)
        render_slurm_sites(site_paths=[site_path], campaign_dir=production)

    generated = generate_campaign(
        repo / "campaign.yaml",
        repo_root=repo,
        output_dir=tmp_path / "template-preview",
        _allow_template_placeholders=True,
    )
    render_slurm_sites(
        site_paths=[site_path],
        campaign_dir=Path(generated.output_dir),
        allow_template_placeholders=True,
    )
    assert Path(generated.output_dir, "submit", "local", "cpu_test.slurm").is_file()


def test_cli_reconcile_returns_incomplete_status(tmp_path) -> None:
    _, _, campaign = build_test_campaign(tmp_path)
    code = main(
        [
            "reconcile",
            "--manifest",
            str(campaign / "manifest.jsonl"),
            "--result-root",
            str(tmp_path / "results"),
            "--output-dir",
            str(tmp_path / "reconcile"),
        ]
    )
    assert code == 1


def test_cli_reconcile_emits_neutral_submittable_retry(tmp_path) -> None:
    repo, _, campaign = build_test_campaign(tmp_path, with_site=True)
    code = main(
        [
            "reconcile",
            "--manifest",
            str(campaign / "manifest.jsonl"),
            "--result-root",
            str(tmp_path / "results"),
            "--output-dir",
            str(tmp_path / "reconcile"),
        ]
    )

    assert code == 1
    retry = tmp_path / "reconcile/retry-campaign"
    assert not (retry / "submit").exists()
    submission_dir = tmp_path / "submissions" / "retry"
    render_slurm_sites(
        site_paths=[repo / "site.yaml"],
        campaign_dir=retry,
        submission_dir=submission_dir,
    )
    assert (retry / "profiles/resources.json").is_file()
    assert (submission_dir / "local/cpu_test.slurm").is_file()


def test_slurm_duration_supports_long_qos_and_rejects_invalid_values() -> None:
    assert parse_duration("1-04:00:00") == 100800
    assert parse_duration("100:00:00") == 360000
    assert format_duration(360000) == "100:00:00"
    with pytest.raises(CampaignError, match="invalid duration"):
        parse_duration("01:60:00")
    with pytest.raises(CampaignError, match="must be positive"):
        format_duration(0)
