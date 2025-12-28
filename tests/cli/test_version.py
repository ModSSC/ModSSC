from __future__ import annotations

from typer.testing import CliRunner

from modssc.__about__ import __version__
from modssc.cli.app import app


def test_cli_version() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert f"modssc {__version__}" in res.stdout
