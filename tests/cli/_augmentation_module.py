from __future__ import annotations

from typer.testing import CliRunner

from modssc.cli.augmentation import app


def test_cli_list_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "tabular.gaussian_noise" in result.stdout
