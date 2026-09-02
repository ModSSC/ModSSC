from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from bench import main as bench_main
from modssc.runtime.continuation import (
    PLANNED_CONTINUATION_EXIT_CODE,
    PlannedContinuation,
    continuation_requested,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _minimal_config(*, resume_policy: str) -> dict[str, Any]:
    return {
        "run": {
            "name": "continuation_cli",
            "seed": 1,
            "output_dir": "runs",
            "resume_policy": resume_policy,
        },
        "dataset": {"id": "toy", "options": {"seed": 1}},
        "sampling": {"seed": 1, "plan": {"split": {"kind": "holdout"}}},
        "preprocess": {
            "seed": 1,
            "fit_on": "train_labeled",
            "plan": {"output_key": "features.X", "steps": [{"id": "core.to_numpy"}]},
        },
        "method": {
            "kind": "inductive",
            "id": "pseudo_label",
            "device": {"device": "cpu", "dtype": "float32"},
            "params": {},
        },
        "evaluation": {
            "split_for_model_selection": "val",
            "report_splits": ["val", "test"],
            "metrics": ["accuracy"],
        },
    }


def test_runner_translates_planned_continuation_only_for_resumable_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _minimal_config(resume_policy="auto")
    monkeypatch.setattr(bench_main, "load_yaml", lambda _path: raw)

    def _checkpointed_run(*_args: object, **_kwargs: object) -> int:
        signal.raise_signal(signal.SIGUSR1)
        assert continuation_requested()
        raise PlannedContinuation()

    monkeypatch.setattr(bench_main, "_run_experiment_body", _checkpointed_run)

    assert bench_main.run_experiment(Path("config.yaml")) == PLANNED_CONTINUATION_EXIT_CODE
    assert not continuation_requested()

    raw["run"]["resume_policy"] = "never"

    def _invalid_non_resumable_continuation(*_args: object, **_kwargs: object) -> int:
        raise PlannedContinuation(0)

    monkeypatch.setattr(
        bench_main,
        "_run_experiment_body",
        _invalid_non_resumable_continuation,
    )
    with pytest.raises(PlannedContinuation):
        bench_main.run_experiment(Path("config.yaml"))


def test_cli_help_documents_retryable_exit_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["modssc-bench", "--help"])

    with pytest.raises(SystemExit) as raised:
        bench_main.main()

    assert raised.value.code == 0
    assert "Exit status 75 (EX_TEMPFAIL)" in capsys.readouterr().out


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="requires POSIX SIGUSR1")
def test_cli_process_handles_usr1_and_exits_with_ex_tempfail(tmp_path: Path) -> None:
    config_path = tmp_path / "continuation.yaml"
    config_path.write_text(
        json.dumps(_minimal_config(resume_policy="auto")),
        encoding="utf-8",
    )
    probe = """
import sys
import time

from bench import main as bench_main
from modssc.runtime.continuation import continuation_requested, raise_planned_continuation


def checkpointing_probe(*_args, **_kwargs):
    print("CONTINUATION_HANDLER_READY", flush=True)
    deadline = time.monotonic() + 10.0
    while not continuation_requested():
        if time.monotonic() >= deadline:
            raise RuntimeError("SIGUSR1 was not observed")
        time.sleep(0.01)
    raise_planned_continuation()


bench_main._run_experiment_body = checkpointing_probe
sys.argv = ["modssc-bench", "--config", sys.argv[1]]
raise SystemExit(bench_main.main())
"""
    env = os.environ.copy()
    python_path = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", probe, str(config_path)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 15.0)
        if not ready:
            process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(
                f"continuation probe did not become ready\nstdout={stdout}\nstderr={stderr}"
            )
        assert process.stdout.readline().strip() == "CONTINUATION_HANDLER_READY"

        os.kill(process.pid, signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=15.0)

        assert process.returncode == PLANNED_CONTINUATION_EXIT_CODE, (stdout, stderr)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)
