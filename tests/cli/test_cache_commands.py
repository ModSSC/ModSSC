from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from modssc.cli.app import app
from modssc.data_loader.cache import CacheLayout, ensure_layout, index_list


def test_cli_cache_ls_purge_and_gc(tmp_path: Path) -> None:
    runner = CliRunner()

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

    res = runner.invoke(app, ["datasets", "cache", "ls", "--cache-dir", str(tmp_path)])
    assert res.exit_code == 0

    res = runner.invoke(
        app, ["datasets", "cache", "purge", "toy:default", "--cache-dir", str(tmp_path)]
    )
    assert res.exit_code == 0

    runner.invoke(
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

    layout = CacheLayout(root=tmp_path)
    ensure_layout(layout)
    fp = index_list(layout)[0]["fingerprint"]

    res = runner.invoke(
        app,
        ["datasets", "cache", "purge", fp, "--fingerprint", "--cache-dir", str(tmp_path)],
    )
    assert res.exit_code == 0

    res = runner.invoke(
        app, ["datasets", "cache", "gc", "--keep-latest", "--cache-dir", str(tmp_path)]
    )
    assert res.exit_code == 0

    res = runner.invoke(
        app, ["datasets", "cache", "gc", "--no-keep-latest", "--cache-dir", str(tmp_path)]
    )
    assert res.exit_code == 0
