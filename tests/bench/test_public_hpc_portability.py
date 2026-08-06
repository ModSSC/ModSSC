from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tracked_public_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        REPO_ROOT / raw.decode("utf-8")
        for raw in completed.stdout.split(b"\0")
        if raw and (REPO_ROOT / raw.decode("utf-8")).is_file()
    ]


def test_public_tree_has_no_private_hpc_site_identity() -> None:
    # Build the forbidden strings from neutral fragments so the policy test
    # does not itself disclose the values it is designed to reject.
    forbidden = (
        "jean" + " zay",
        "jean" + "-zay",
        "jean" + "zay",
        "uqj" + "57xk",
        "/" + "lustre/",
        "dqp" + "@",
        "id" + "ris",
        "qos" + "_gpu",
        "module" + " load openjdk",
        "$" + "scratch/",
    )
    offenders: list[str] = []
    for path in _tracked_public_files():
        if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".npz", ".pkl"}:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(REPO_ROOT).as_posix().lower()
        if any(token in text or token in relative for token in forbidden):
            offenders.append(relative)
    assert offenders == []


def test_public_hpc_layer_only_exposes_generic_slurm_templates() -> None:
    assert (REPO_ROOT / "tools/hpc/sites/slurm/job_env.sh").is_file()
    assert (REPO_ROOT / "tools/hpc/slurm/array-task.sh").is_file()
    assert (REPO_ROOT / "tools/hpc/slurm/run-operation.sh").is_file()
    assert (REPO_ROOT / "tools/hpc/slurm/runtime-context.sh").is_file()
    assert (REPO_ROOT / "tools/hpc/slurm_renderer.py").is_file()
    assert (REPO_ROOT / "tools/hpc/config/profiles/slurm.example.yaml").is_file()
    assert not any(
        path.is_dir()
        for path in (REPO_ROOT / "tools/hpc/sites").iterdir()
        if path.name not in {"slurm", "regional", "private"}
    )


def test_generic_profile_and_documentation_reference_existing_public_payloads() -> None:
    profile = (REPO_ROOT / "tools/hpc/config/profiles/slurm.example.yaml").read_text(
        encoding="utf-8"
    )
    guide = (REPO_ROOT / "docs/development/hpc-campaigns.md").read_text(encoding="utf-8")
    expected = (
        "tools/hpc/sites/slurm/job_env.sh",
        "tools/hpc/slurm/array-task.sh",
        "tools/hpc/slurm/run-operation.sh",
        "tools/hpc/config/profiles/slurm.example.yaml",
        "tools/hpc/config/allocations/slurm.example.yaml",
    )
    for relative in expected:
        assert (REPO_ROOT / relative).is_file()
        assert relative in guide
    assert 'source "$MODSSC_ROOT/tools/hpc/sites/slurm/job_env.sh"' in profile
    assert "sites/" + "slurm_gpu" not in guide


def test_generic_slurm_payloads_use_the_pinned_interpreter() -> None:
    job_env = (REPO_ROOT / "tools/hpc/sites/slurm/job_env.sh").read_text(encoding="utf-8")
    array_task = (REPO_ROOT / "tools/hpc/slurm/array-task.sh").read_text(encoding="utf-8")
    operation = (REPO_ROOT / "tools/hpc/slurm/run-operation.sh").read_text(encoding="utf-8")

    assert "MODSSC_PYTHON must name the pre-existing pinned interpreter" in job_env
    assert "command -v python" not in job_env
    assert '"$MODSSC_PYTHON" "${RUN_TASK_ARGS[@]}"' in array_task
    assert '"$MODSSC_PYTHON" -m tools.hpc.scheduler_failure' in array_task
    assert '[[ ! -x "$MODSSC_PYTHON" ]]' in array_task
    assert '[[ ! -x "$MODSSC_PYTHON" ]]' in operation
    assert "command -v python" not in operation


def test_bench_layer_contains_no_scheduler_rendering_or_environment_reads() -> None:
    offenders: list[str] = []
    forbidden = ("#SBATCH", ".slurm", "SLURM_")
    for path in (REPO_ROOT / "bench").rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".yaml", ".yml", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(token in text for token in forbidden):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == []
