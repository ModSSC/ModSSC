from __future__ import annotations

import logging

import pytest
import typer
from typer.testing import CliRunner

from modssc import logging as modlog


@pytest.fixture()
def reset_logging_state() -> None:
    logger_names = ["modssc", "bench"]
    snapshots: dict[str, dict[str, object]] = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        snapshots[name] = {
            "handlers": list(logger.handlers),
            "level": logger.level,
            "propagate": logger.propagate,
        }
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

    saved_handlers = dict(modlog._HANDLERS)
    modlog._HANDLERS.clear()
    yield
    for name, snapshot in snapshots.items():
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        for handler in snapshot["handlers"]:
            logger.addHandler(handler)
        logger.setLevel(snapshot["level"])
        logger.propagate = snapshot["propagate"]
    modlog._HANDLERS.clear()
    modlog._HANDLERS.update(saved_handlers)


def test_resolve_log_level_default(monkeypatch, reset_logging_state) -> None:
    monkeypatch.delenv("MODSSC_LOG_LEVEL", raising=False)
    assert modlog.resolve_log_level(None) == "none"


def test_resolve_log_level_from_env(monkeypatch, reset_logging_state) -> None:
    monkeypatch.setenv("MODSSC_LOG_LEVEL", "basic")
    assert modlog.resolve_log_level(None) == "basic"


def test_configure_logging_levels(reset_logging_state) -> None:
    modlog.configure_logging("none")
    assert logging.getLogger("modssc").level == logging.WARNING

    modlog.configure_logging("basic")
    assert logging.getLogger("modssc").level == logging.INFO

    modlog.configure_logging("detailed")
    assert logging.getLogger("modssc").level == logging.DEBUG


def test_configure_logging_single_handler(reset_logging_state) -> None:
    modlog.configure_logging("basic")
    logger = logging.getLogger("modssc")
    assert len(logger.handlers) == 1

    modlog.configure_logging("detailed")
    assert len(logger.handlers) == 1


def test_configure_logging_reuses_existing_handler(reset_logging_state) -> None:
    modlog.configure_logging("basic")
    logger = logging.getLogger("modssc")
    handler = logger.handlers[0]
    logger.removeHandler(handler)

    modlog.configure_logging("basic")
    assert handler in logger.handlers


def test_configure_logging_skips_when_handlers_present(reset_logging_state) -> None:
    logger = logging.getLogger("modssc")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    modlog._HANDLERS.clear()

    modlog.configure_logging("basic")
    assert handler in logger.handlers


def test_configure_logging_invalid(reset_logging_state) -> None:
    with pytest.raises(ValueError):
        modlog.configure_logging("nope")


def test_add_log_level_callback_invalid_env(monkeypatch, reset_logging_state) -> None:
    monkeypatch.setenv("MODSSC_LOG_LEVEL", "nope")
    app = typer.Typer()
    modlog.add_log_level_callback(app)

    @app.command()
    def dummy():
        return None

    runner = CliRunner()
    result = runner.invoke(app, ["dummy"])
    assert result.exit_code != 0
    assert "Unknown log level" in result.output


def test_cli_logging_module_imports() -> None:
    import modssc.cli.logging as cli_logging

    assert "configure_logging" in cli_logging.__all__
