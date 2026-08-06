from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.bench.campaign.helpers import build_test_campaign


def _module_command(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "tools.hpc.submit_chained_arrays", *arguments]


def _campaign_with_two_blocks(tmp_path: Path) -> tuple[Path, list[Path]]:
    _, _, campaign = build_test_campaign(
        tmp_path / "campaign source with spaces",
        with_site=True,
        array_block_size=1,
    )
    wrappers = sorted((campaign / "submit" / "local").glob("cpu_test.block*.slurm"))
    assert len(wrappers) == 2
    return campaign, wrappers


def _fake_sbatch(tmp_path: Path, *, output: str | None = None) -> tuple[Path, Path]:
    fake_bin = tmp_path / "fake scheduler bin"
    fake_bin.mkdir()
    call_log = tmp_path / "sbatch calls.log"
    fake = fake_bin / "sbatch"
    if output is None:
        response = (
            'counter=0\nif [[ -f "$FAKE_SBATCH_COUNTER" ]]; then '
            'read -r counter < "$FAKE_SBATCH_COUNTER"; fi\n'
            'counter=$((counter + 1))\nprintf "%s\\n" "$counter" > "$FAKE_SBATCH_COUNTER"\n'
            'printf "%s;test-cluster\\n" "$((7300 + counter))"\n'
        )
    else:
        response = f"printf '%s\\n' {shlex.quote(output)}\n"
    fake.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'printf "%q " "$@" >> "$FAKE_SBATCH_CALL_LOG"\nprintf "\\n" >> "$FAKE_SBATCH_CALL_LOG"\n'
        + response,
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake_bin, call_log


def _environment(tmp_path: Path, fake_bin: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_SBATCH_CALL_LOG": str(tmp_path / "sbatch calls.log"),
        "FAKE_SBATCH_COUNTER": str(tmp_path / "sbatch counter"),
    }


def _preflight_report(
    campaign: Path, *, overrides: dict[str, Any] | None = None, name: str = "preflight report.json"
) -> Path:
    resources = json.loads((campaign / "profiles" / "resources.json").read_text(encoding="utf-8"))
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "campaign_id": "test-campaign",
        "manifest_sha256": resources["manifest_sha256"],
        "required_architecture": "CPU",
        "created_at": (now - timedelta(hours=1)).isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "max_allocation_age_hours": 24.0,
    }
    payload.update(overrides or {})
    path = campaign / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_chained_submit_overrides_time_and_throttle_and_preserves_paths_with_spaces(
    tmp_path: Path,
) -> None:
    campaign, wrappers = _campaign_with_two_blocks(tmp_path)
    fake_bin, call_log = _fake_sbatch(tmp_path)
    preflight = _preflight_report(campaign)

    completed = subprocess.run(
        _module_command(
            "--throttle",
            "2",
            "--time",
            "00:09:00",
            "--preflight-report",
            str(preflight),
            *(str(path) for path in wrappers),
        ),
        cwd=Path.cwd(),
        env=_environment(tmp_path, fake_bin),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    calls = [shlex.split(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
    export = calls[0][4]
    assert export == calls[1][4]
    fields = dict(
        item.split("=", maxsplit=1) for item in export.removeprefix("--export=ALL,").split(",")
    )
    assert fields == {"MODSSC_PREFLIGHT_REPORT": str(preflight.resolve())}
    assert calls == [
        [
            "--parsable",
            "--array=0-0%2",
            "--time=00:09:00",
            f"--chdir={campaign.resolve()}",
            export,
            str(wrappers[0].resolve()),
        ],
        [
            "--parsable",
            "--array=0-0%2",
            "--time=00:09:00",
            f"--chdir={campaign.resolve()}",
            export,
            "--dependency=afterok:7301",
            str(wrappers[1].resolve()),
        ],
    ]
    assert "submitted job_id=7301 dependency=none" in completed.stdout
    assert "submitted job_id=7302 dependency=afterok:7301" in completed.stdout


def test_chained_submit_dry_run_validates_without_invoking_sbatch(tmp_path: Path) -> None:
    campaign, wrappers = _campaign_with_two_blocks(tmp_path)
    preflight = _preflight_report(campaign)

    completed = subprocess.run(
        _module_command(
            "--dry-run",
            "--throttle",
            "2",
            "--time",
            "00:09:00",
            "--preflight-report",
            str(preflight),
            *(str(path) for path in wrappers),
        ),
        cwd=Path.cwd(),
        env={**os.environ, "PATH": ""},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "DRY-RUN[0] sbatch --parsable --array=0-0%2 --time=00:09:00" in completed.stdout
    assert "'--dependency=afterok:<job-id-0>'" in completed.stdout


def test_chained_submit_rejects_out_of_order_blocks_before_sbatch(tmp_path: Path) -> None:
    campaign, wrappers = _campaign_with_two_blocks(tmp_path)
    fake_bin, call_log = _fake_sbatch(tmp_path)
    preflight = _preflight_report(campaign)

    completed = subprocess.run(
        _module_command(
            "--throttle",
            "2",
            "--time",
            "00:09:00",
            "--preflight-report",
            str(preflight),
            str(wrappers[1]),
            str(wrappers[0]),
        ),
        cwd=Path.cwd(),
        env=_environment(tmp_path, fake_bin),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "strictly consecutive order" in completed.stderr
    assert not call_log.exists()


def test_chained_submit_rejects_noninitial_throttle_and_excess_walltime(tmp_path: Path) -> None:
    campaign, wrappers = _campaign_with_two_blocks(tmp_path)
    preflight = _preflight_report(campaign)

    wrong_throttle = subprocess.run(
        _module_command(
            "--dry-run",
            "--throttle",
            "64",
            "--time",
            "00:09:00",
            "--preflight-report",
            str(preflight),
            str(wrappers[0]),
        ),
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    excess_time = subprocess.run(
        _module_command(
            "--dry-run",
            "--throttle",
            "2",
            "--time",
            "00:10:01",
            "--preflight-report",
            str(preflight),
            str(wrappers[0]),
        ),
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert wrong_throttle.returncode == 1
    assert "pre-registered initial concurrency (2)" in wrong_throttle.stderr
    assert excess_time.returncode == 1
    assert "exceeds the profile cap" in excess_time.stderr


def test_chained_submit_rejects_shorter_time_for_fixed_scientific_profile(
    tmp_path: Path,
) -> None:
    campaign, wrappers = _campaign_with_two_blocks(tmp_path)
    resources_path = campaign / "profiles" / "resources.json"
    resources = json.loads(resources_path.read_text(encoding="utf-8"))
    resources["resources"][0]["fixed_walltime"] = True
    resources_path.write_text(json.dumps(resources), encoding="utf-8")
    preflight = _preflight_report(campaign)

    completed = subprocess.run(
        _module_command(
            "--dry-run",
            "--throttle",
            "2",
            "--time",
            "00:09:00",
            "--preflight-report",
            str(preflight),
            str(wrappers[0]),
        ),
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "--time must equal the fixed profile walltime (00:10:00)" in completed.stderr


def test_chained_submit_rejects_invalid_duration_before_sbatch(tmp_path: Path) -> None:
    campaign, wrappers = _campaign_with_two_blocks(tmp_path)
    fake_bin, call_log = _fake_sbatch(tmp_path)
    preflight = _preflight_report(campaign)

    completed = subprocess.run(
        _module_command(
            "--throttle",
            "2",
            "--time",
            "00:60:00",
            "--preflight-report",
            str(preflight),
            str(wrappers[0]),
        ),
        cwd=Path.cwd(),
        env=_environment(tmp_path, fake_bin),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "minutes and seconds must be < 60" in completed.stderr
    assert not call_log.exists()


def test_chained_submit_rejects_tampered_index_before_sbatch(tmp_path: Path) -> None:
    campaign, wrappers = _campaign_with_two_blocks(tmp_path)
    fake_bin, call_log = _fake_sbatch(tmp_path)
    preflight = _preflight_report(campaign)
    index_path = campaign / "profiles" / "local.cpu_test.block000.indices"
    index_path.write_text("99\n", encoding="utf-8")

    completed = subprocess.run(
        _module_command(
            "--throttle",
            "2",
            "--time",
            "00:09:00",
            "--preflight-report",
            str(preflight),
            str(wrappers[0]),
        ),
        cwd=Path.cwd(),
        env=_environment(tmp_path, fake_bin),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "array index SHA-256 mismatch" in completed.stderr
    assert not call_log.exists()


def test_chained_submit_requires_preflight_before_sbatch(tmp_path: Path) -> None:
    _, wrappers = _campaign_with_two_blocks(tmp_path)
    fake_bin, call_log = _fake_sbatch(tmp_path)

    completed = subprocess.run(
        _module_command(
            "--throttle",
            "2",
            "--time",
            "00:09:00",
            str(wrappers[0]),
        ),
        cwd=Path.cwd(),
        env=_environment(tmp_path, fake_bin),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--preflight-report" in completed.stderr
    assert not call_log.exists()


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("expired", "preflight report has expired"),
        ("wrong_architecture", "preflight architecture differs; expected CPU"),
        ("wrong_manifest", "preflight manifest digest differs from the wrappers"),
    ],
)
def test_chained_submit_rejects_invalid_preflight_before_sbatch(
    tmp_path: Path, failure: str, expected_error: str
) -> None:
    campaign, wrappers = _campaign_with_two_blocks(tmp_path)
    fake_bin, call_log = _fake_sbatch(tmp_path)
    now = datetime.now(UTC)
    overrides: dict[str, Any]
    if failure == "expired":
        overrides = {
            "created_at": (now - timedelta(hours=2)).isoformat(),
            "expires_at": (now - timedelta(hours=1)).isoformat(),
        }
    elif failure == "wrong_architecture":
        overrides = {"required_architecture": "A100"}
    else:
        overrides = {"manifest_sha256": "0" * 64}
    preflight = _preflight_report(campaign, overrides=overrides)

    completed = subprocess.run(
        _module_command(
            "--throttle",
            "2",
            "--time",
            "00:09:00",
            "--preflight-report",
            str(preflight),
            str(wrappers[0]),
        ),
        cwd=Path.cwd(),
        env=_environment(tmp_path, fake_bin),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert expected_error in completed.stderr
    assert not call_log.exists()


def test_chained_submit_rejects_nonparsable_sbatch_id_and_stops_chain(tmp_path: Path) -> None:
    campaign, wrappers = _campaign_with_two_blocks(tmp_path)
    fake_bin, call_log = _fake_sbatch(tmp_path, output="Submitted batch job 7301")
    preflight = _preflight_report(campaign)

    completed = subprocess.run(
        _module_command(
            "--throttle",
            "2",
            "--time",
            "00:09:00",
            "--preflight-report",
            str(preflight),
            *(str(path) for path in wrappers),
        ),
        cwd=Path.cwd(),
        env=_environment(tmp_path, fake_bin),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "invalid parsable job id" in completed.stderr
    assert len(call_log.read_text(encoding="utf-8").splitlines()) == 1
