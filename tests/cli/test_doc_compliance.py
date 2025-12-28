from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from modssc.cli.graph import app as graph_app


def test_graph_docs_match_cli_signature() -> None:
    root = Path(__file__).resolve().parents[2]
    doc = (root / "docs/cli/graph.md").read_text(encoding="utf-8")
    assert "--dataset" in doc

    runner = CliRunner()
    res = runner.invoke(graph_app, ["build", "--help"])
    assert res.exit_code == 0
    assert "--dataset" in res.stdout
