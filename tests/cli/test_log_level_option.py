from __future__ import annotations

from typer.testing import CliRunner

from modssc.cli import app as app_mod
from modssc.cli.inductive import app as inductive_app

runner = CliRunner()


def test_modssc_log_level_basic(monkeypatch) -> None:
    monkeypatch.setattr(
        app_mod,
        "_BRICK_STATUS",
        {
            "datasets": app_mod.BrickStatus(
                module="modssc.cli.datasets",
                name="datasets",
                help_text="Datasets",
                available=True,
            )
        },
    )
    res = runner.invoke(app_mod.app, ["--log-level", "basic", "doctor"])
    assert res.exit_code == 0


def test_modssc_inductive_log_level_detailed() -> None:
    res = runner.invoke(inductive_app, ["--log-level", "detailed", "methods", "list"])
    assert res.exit_code == 0


def test_modssc_log_level_invalid_value() -> None:
    res = runner.invoke(app_mod.app, ["--log-level", "nope", "doctor"])
    assert res.exit_code != 0
    output = getattr(res, "stderr", "") + res.stdout
    assert "Unknown log level" in output
