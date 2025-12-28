from __future__ import annotations

import json

from typer.testing import CliRunner

from modssc.cli.preprocess import app

runner = CliRunner()


def test_cli_steps_list_json() -> None:
    res = runner.invoke(app, ["steps", "list", "--json"])
    assert res.exit_code == 0
    items = json.loads(res.stdout)
    assert "core.ensure_2d" in items


def test_cli_models_list_json() -> None:
    res = runner.invoke(app, ["models", "list", "--json"])
    assert res.exit_code == 0
    items = json.loads(res.stdout)
    assert "stub:text" in items


def test_cli_steps_info() -> None:
    res = runner.invoke(app, ["steps", "info", "core.ensure_2d"])
    assert res.exit_code == 0
    info = json.loads(res.stdout)
    assert info["id"] == "core.ensure_2d"
