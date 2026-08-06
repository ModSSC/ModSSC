from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
JOB_ENV = REPO_ROOT / "tools/hpc/sites/slurm/job_env.sh"
ARRAY_TASK = REPO_ROOT / "tools/hpc/slurm/array-task.sh"
RUN_OPERATION = REPO_ROOT / "tools/hpc/slurm/run-operation.sh"
PROFILE = REPO_ROOT / "tools/hpc/config/profiles/slurm.example.yaml"


def _hostname() -> str:
    return subprocess.check_output(["hostname", "-s"], text=True).strip()


def _base_environment(tmp_path: Path) -> dict[str, str]:
    interpreter = tmp_path / "prepared-python"
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o700)
    return {
        **os.environ,
        "SLURM_JOB_ID": "12345",
        "SLURMD_NODENAME": _hostname(),
        "MODSSC_ROOT": str(REPO_ROOT),
        "MODSSC_SCRATCH": str(tmp_path / "scratch"),
        "MODSSC_PYTHON": str(interpreter),
    }


def test_generic_job_environment_refuses_login_node_execution(tmp_path: Path) -> None:
    env = _base_environment(tmp_path)
    env.pop("SLURM_JOB_ID")

    completed = subprocess.run(
        ["bash", "-c", f'source "{JOB_ENV}"; modssc_slurm_env cpu'],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode != 0
    assert "Slurm allocation" in completed.stderr


def test_generic_job_environment_refuses_allocation_shell(tmp_path: Path) -> None:
    env = _base_environment(tmp_path)
    env.pop("SLURMD_NODENAME")

    completed = subprocess.run(
        ["bash", "-c", f'source "{JOB_ENV}"; modssc_slurm_env cpu'],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode != 0
    assert "compute node" in completed.stderr


def test_generic_job_environment_requires_a_pinned_interpreter(tmp_path: Path) -> None:
    env = _base_environment(tmp_path)
    env.pop("MODSSC_PYTHON")

    completed = subprocess.run(
        ["bash", "-c", f'source "{JOB_ENV}"; modssc_slurm_env cpu'],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode != 0
    assert "pre-existing pinned interpreter" in completed.stderr


def test_generic_job_environment_uses_prepared_interpreter_and_scratch(
    tmp_path: Path,
) -> None:
    env = _base_environment(tmp_path)
    command = (
        f'source "{JOB_ENV}"; modssc_slurm_env gpu; '
        'printf \'%s\\n\' "$MODSSC_PYTHON" "$MODSSC_CACHE_ROOT" '
        '"$MODSSC_GRAPH_CACHE_DIR" "$MODSSC_ACCELERATOR_ARCH"'
    )

    completed = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines == [
        env["MODSSC_PYTHON"],
        str(tmp_path / "scratch/modssc_cache"),
        str(tmp_path / "scratch/modssc_cache/graph"),
        "gpu",
    ]
    assert (tmp_path / "scratch/modssc_cache/graph").is_dir()


def test_generic_profile_references_only_existing_public_setup() -> None:
    payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))

    assert payload["scheduler"] == "slurm"
    assert payload["setup"] == ['source "$MODSSC_ROOT/tools/hpc/sites/slurm/job_env.sh"']
    assert JOB_ENV.is_file()
    for profile in payload["profiles"].values():
        directives = profile["directives"]
        assert not ({"account", "partition", "qos", "constraint"} & set(directives))
        assert profile["setup"][0].startswith("modssc_slurm_env ")


def test_public_slurm_payloads_are_compute_guarded_and_install_nothing() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in (JOB_ENV, ARRAY_TASK, RUN_OPERATION)
    )

    assert "SLURM_JOB_ID" in combined
    assert "SLURMD_NODENAME" in combined
    assert "pip install" not in combined
    assert "conda install" not in combined
    assert '"$MODSSC_PYTHON" -m tools.hpc.scheduler_failure' in combined
    assert '"$MODSSC_PYTHON" "${RUN_TASK_ARGS[@]}"' in combined


def test_generic_operation_dispatches_through_the_pinned_interpreter(tmp_path: Path) -> None:
    arguments = tmp_path / "arguments.txt"
    interpreter = tmp_path / "python"
    interpreter.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {str(arguments)!r}\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o700)
    env = _base_environment(tmp_path)
    env["MODSSC_PYTHON"] = str(interpreter)

    completed = subprocess.run(
        ["bash", str(RUN_OPERATION), "reconcile", "--manifest", "manifest.jsonl"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert arguments.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "bench.campaign",
        "reconcile",
        "--manifest",
        "manifest.jsonl",
    ]


def test_generic_operation_rejects_unknown_action(tmp_path: Path) -> None:
    env = _base_environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(RUN_OPERATION), "unknown-operation"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 64
    assert "Unsupported campaign operation" in completed.stderr


def test_generic_operation_requires_a_pinned_interpreter(tmp_path: Path) -> None:
    env = _base_environment(tmp_path)
    env.pop("MODSSC_PYTHON")

    completed = subprocess.run(
        ["bash", str(RUN_OPERATION), "reconcile", "--manifest", "manifest.jsonl"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode != 0
    assert "pre-existing pinned interpreter" in completed.stderr
