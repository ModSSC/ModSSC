from __future__ import annotations

import json

from typer.testing import CliRunner

from modssc.cli.transductive import app

runner = CliRunner()


def test_transductive_methods_list() -> None:
    res = runner.invoke(app, ["methods", "list"])
    assert res.exit_code == 0
    assert "label_propagation" in res.stdout


def test_transductive_methods_info() -> None:
    res = runner.invoke(app, ["methods", "info", "label_propagation"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["method_id"] == "label_propagation"
    assert payload["import_path"].endswith("label_propagation:LabelPropagationMethod")
    assert payload["info"]["method_id"] == "label_propagation"


def test_transductive_methods_list_with_log_level() -> None:
    res = runner.invoke(app, ["methods", "list", "--log-level", "basic"])
    assert res.exit_code == 0


def test_transductive_methods_info_unknown() -> None:
    res = runner.invoke(app, ["methods", "info", "nope", "--log-level", "basic"])
    assert res.exit_code == 2
    assert "Unknown method id" in res.output


def test_transductive_methods_info_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "modssc.cli.transductive.transductive_registry._debug_registry",
        lambda: {"dummy": "modssc.transductive.methods.fake:Fake"},
    )
    monkeypatch.setattr(
        "modssc.cli.transductive.transductive_registry.get_method_info",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    res = runner.invoke(app, ["methods", "info", "dummy", "--log-level", "basic"])
    assert res.exit_code == 2
    assert '"error": "boom"' in res.stdout
