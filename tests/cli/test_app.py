from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from modssc.cli import app as app_mod


def test_try_add_typer_import_error(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'missing_dep'", name="missing_dep")

    monkeypatch.setattr(app_mod.importlib, "import_module", boom)
    app_mod._try_add_typer("nope", "x", "y")
    status = app_mod._BRICK_STATUS["x"]
    assert status.available is False
    assert status.missing_module == "missing_dep"


def test_try_add_typer_external_import_error_parses_message(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise ImportError("No module named 'numpy'")

    monkeypatch.setattr(app_mod.importlib, "import_module", boom)
    monkeypatch.setattr(app_mod, "_BRICK_STATUS", {})
    app_mod._try_add_typer("nope", "x", "y")
    status = app_mod._BRICK_STATUS["x"]
    assert status.missing_module == "numpy"


def test_try_add_typer_internal_import_error(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'modssc.boom'", name="modssc.boom")

    monkeypatch.setattr(app_mod.importlib, "import_module", boom)
    monkeypatch.setattr(app_mod, "_BRICK_STATUS", {})
    with pytest.raises(ModuleNotFoundError):
        app_mod._try_add_typer("modssc.cli.graph", "graph", "Graph")


def test_is_internal_import_error_without_module_name() -> None:
    assert app_mod._is_internal_import_error(ImportError("boom"), "modssc.cli.x") is True


def test_try_add_typer_no_app_attribute(monkeypatch) -> None:
    monkeypatch.setattr(app_mod.importlib, "import_module", lambda *_: SimpleNamespace())
    with pytest.raises(RuntimeError):
        app_mod._try_add_typer("no_app", "x", "y")


def test_doctor_json_reports_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        app_mod,
        "_BRICK_STATUS",
        {
            "graph": app_mod.BrickStatus(
                module="modssc.cli.graph",
                name="graph",
                help_text="Graph",
                available=False,
                error="No module named 'torch'",
                missing_module="torch",
                extra_hint="graph",
            )
        },
    )
    runner = CliRunner()
    res = runner.invoke(app_mod.app, ["doctor", "--json"])
    assert res.exit_code == 2
    data = json.loads(res.stdout)
    assert data[0]["name"] == "graph"
    assert data[0]["extra_hint"] == "graph"


def test_doctor_text_output(monkeypatch) -> None:
    monkeypatch.setattr(
        app_mod,
        "_BRICK_STATUS",
        {
            "datasets": app_mod.BrickStatus(
                module="modssc.cli.datasets",
                name="datasets",
                help_text="Datasets",
                available=True,
            ),
            "graph": app_mod.BrickStatus(
                module="modssc.cli.graph",
                name="graph",
                help_text="Graph",
                available=False,
                error="No module named 'torch'",
                missing_module="torch",
                extra_hint="graph",
            ),
        },
    )
    runner = CliRunner()
    res = runner.invoke(app_mod.app, ["doctor"])
    assert res.exit_code == 2
    assert "datasets: ok" in res.stdout
    assert "graph: unavailable" in res.stdout


def test_extra_hint_for_missing_module() -> None:
    assert app_mod._extra_hint_for("datasets", "torch") == "graph"


def test_is_internal_import_error_for_module_name() -> None:
    exc = ModuleNotFoundError("No module named 'foo'", name="foo")
    assert app_mod._is_internal_import_error(exc, "foo") is True


def test_invalid_env_log_level(monkeypatch) -> None:
    monkeypatch.setenv("MODSSC_LOG_LEVEL", "nope")
    runner = CliRunner()
    res = runner.invoke(app_mod.app, ["doctor"])
    assert res.exit_code != 0
    assert "Unknown log level" in res.output


def test_doctor_all_available(monkeypatch) -> None:
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
    runner = CliRunner()
    res = runner.invoke(app_mod.app, ["doctor"])
    assert res.exit_code == 0
    assert "datasets: ok" in res.stdout


def test_doctor_unavailable_without_hints(monkeypatch) -> None:
    monkeypatch.setattr(
        app_mod,
        "_BRICK_STATUS",
        {
            "graph": app_mod.BrickStatus(
                module="modssc.cli.graph",
                name="graph",
                help_text="Graph",
                available=False,
                error=None,
                missing_module=None,
                extra_hint=None,
            )
        },
    )
    runner = CliRunner()
    res = runner.invoke(app_mod.app, ["doctor"])
    assert res.exit_code == 2
    assert "graph: unavailable" in res.stdout
