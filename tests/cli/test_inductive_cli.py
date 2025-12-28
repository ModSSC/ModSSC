from __future__ import annotations

from typer.testing import CliRunner

from modssc.cli.inductive import app


def test_inductive_methods_list_with_log_level() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["methods", "list", "--log-level", "basic"])
    assert result.exit_code == 0


def test_inductive_methods_info_unknown() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["methods", "info", "nope", "--log-level", "basic"])
    assert result.exit_code == 2
    assert "Unknown method id" in result.output


def test_inductive_methods_info_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "modssc.cli.inductive.inductive_registry._debug_registry",
        lambda: {"dummy": "modssc.inductive.methods.fake:Fake"},
    )
    monkeypatch.setattr(
        "modssc.cli.inductive.inductive_registry.get_method_info",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["methods", "info", "dummy", "--log-level", "basic"])
    assert result.exit_code == 2
    assert '"error": "boom"' in result.stdout


def test_inductive_methods_info_success(monkeypatch) -> None:
    from modssc.inductive.base import MethodInfo

    monkeypatch.setattr(
        "modssc.cli.inductive.inductive_registry._debug_registry",
        lambda: {"dummy": "modssc.inductive.methods.fake:Fake"},
    )
    monkeypatch.setattr(
        "modssc.cli.inductive.inductive_registry.get_method_info",
        lambda *_: MethodInfo(method_id="dummy", name="Dummy", year=2020),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["methods", "info", "dummy"])
    assert result.exit_code == 0
    assert '"method_id": "dummy"' in result.stdout
