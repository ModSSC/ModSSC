from __future__ import annotations

from typer.testing import CliRunner

from modssc.cli.app import app

runner = CliRunner()


def test_cli_list_and_info() -> None:
    res = runner.invoke(app, ["datasets", "list"])
    assert res.exit_code == 0
    assert "toy" in res.stdout

    res2 = runner.invoke(app, ["datasets", "info", "--dataset", "toy"])
    assert res2.exit_code == 0
    assert '"key": "toy"' in res2.stdout


def test_cli_download_toy(tmp_path) -> None:
    res = runner.invoke(
        app,
        [
            "datasets",
            "download",
            "--dataset",
            "toy",
            "--cache-dir",
            str(tmp_path),
            "--force",
        ],
    )
    assert res.exit_code == 0
    assert "Downloaded: toy" in res.stdout

    res2 = runner.invoke(app, ["datasets", "cache", "ls", "--cache-dir", str(tmp_path)])
    assert res2.exit_code == 0
