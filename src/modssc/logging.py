from __future__ import annotations

import logging
import os
from typing import Annotated

import typer

_LOG_LEVEL_ALIASES = {
    "quiet": "none",
    "full": "detailed",
}

_LOG_LEVELS = {
    "none": logging.WARNING,
    "basic": logging.INFO,
    "detailed": logging.DEBUG,
}

_BASIC_FMT = "%(levelname)s %(name)s: %(message)s"
_DETAILED_FMT = "%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s"

_HANDLERS: dict[str, logging.Handler] = {}


def _allowed_levels() -> str:
    base = ", ".join(sorted(_LOG_LEVELS.keys()))
    aliases = ", ".join(sorted(_LOG_LEVEL_ALIASES.keys()))
    return f"{base} (aliases: {aliases})"


def normalize_log_level(level: str) -> str:
    value = str(level).strip().lower()
    value = _LOG_LEVEL_ALIASES.get(value, value)
    if value not in _LOG_LEVELS:
        raise ValueError(f"Unknown log level {level!r}. Use one of: {_allowed_levels()}.")
    return value


def resolve_log_level(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        env_value = os.environ.get("MODSSC_LOG_LEVEL")
        if env_value:
            return normalize_log_level(env_value)
        return "none"
    return normalize_log_level(value)


def _validate_log_level(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return normalize_log_level(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


LogLevelOption = Annotated[
    str | None,
    typer.Option(
        "--log-level",
        "--log",
        case_sensitive=False,
        help="Logging level: none, basic, detailed (aliases: quiet, full).",
        callback=_validate_log_level,
    ),
]


def configure_logging(level: str, *, fmt: str | None = None) -> None:
    normalized = normalize_log_level(level)
    target_level = _LOG_LEVELS[normalized]
    format_str = fmt or (_DETAILED_FMT if normalized == "detailed" else _BASIC_FMT)

    for logger_name in ("modssc", "bench"):
        _configure_logger(logger_name, level=target_level, fmt=format_str)


def _configure_logger(name: str, *, level: int, fmt: str) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = _HANDLERS.get(name)
    if handler is not None:
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt))
        if handler not in logger.handlers:
            logger.addHandler(handler)
        return

    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    _HANDLERS[name] = handler


def add_log_level_callback(app: typer.Typer) -> None:
    @app.callback()
    def _log_level_callback(log_level: LogLevelOption = None) -> None:
        try:
            resolved = resolve_log_level(log_level)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        configure_logging(resolved)
