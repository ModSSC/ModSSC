from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any

import typer

from modssc.__about__ import __version__
from modssc.logging import LogLevelOption, configure_logging, resolve_log_level

app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass(frozen=True)
class BrickStatus:
    module: str
    name: str
    help_text: str
    available: bool
    error: str | None = None
    missing_module: str | None = None
    extra_hint: str | None = None


_BRICK_STATUS: dict[str, BrickStatus] = {}

_BRICK_EXTRA_HINTS: dict[str, str] = {
    "datasets": "datasets",
    "graph": "graph",
    "inductive": "inductive-torch or inductive-tf",
    "supervised": "sklearn or supervised-torch",
}

_MISSING_MODULE_HINTS: dict[str, str] = {
    "datasets": "datasets",
    "sklearn": "sklearn",
    "torch": "graph",
    "torch_geometric": "graph",
    "tensorflow": "tfds",
    "tensorflow_datasets": "tfds",
}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"modssc {__version__}")
        raise typer.Exit(code=0)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the package version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    log_level: LogLevelOption = None,
) -> None:
    """ModSSC command line interface."""
    try:
        resolved = resolve_log_level(log_level)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    configure_logging(resolved)
    _ = version


def _extra_hint_for(name: str, missing_module: str | None) -> str | None:
    if missing_module and missing_module in _MISSING_MODULE_HINTS:
        return _MISSING_MODULE_HINTS[missing_module]
    return _BRICK_EXTRA_HINTS.get(name)


def _missing_module_from_error(exc: ImportError) -> str | None:
    missing = getattr(exc, "name", None)
    if missing:
        return str(missing)
    msg = str(exc)
    if "No module named" in msg:
        return msg.split("No module named", 1)[-1].strip().strip("'\"")
    return None


def _is_internal_import_error(exc: ImportError, module: str) -> bool:
    missing = _missing_module_from_error(exc)
    if missing is None:
        return True
    if missing == module:
        return True
    return missing.startswith("modssc")


def _try_add_typer(module: str, name: str, help_text: str) -> None:
    try:
        mod = importlib.import_module(module)
    except (ModuleNotFoundError, ImportError) as exc:
        if _is_internal_import_error(exc, module):
            raise
        missing = _missing_module_from_error(exc)
        extra_hint = _extra_hint_for(name, missing)
        _BRICK_STATUS[name] = BrickStatus(
            module=module,
            name=name,
            help_text=help_text,
            available=False,
            error=str(exc),
            missing_module=missing,
            extra_hint=extra_hint,
        )
        return
    subapp = getattr(mod, "app", None)
    if subapp is None:
        raise RuntimeError(f"CLI module {module!r} does not define 'app'.")
    app.add_typer(subapp, name=name, help=help_text)
    _BRICK_STATUS[name] = BrickStatus(module=module, name=name, help_text=help_text, available=True)


@app.command("doctor")
def doctor(json_output: bool = typer.Option(False, "--json", help="Output JSON.")) -> None:
    """Report which optional CLI bricks are available and why others are missing."""
    statuses = list(_BRICK_STATUS.values())
    if json_output:
        payload: list[dict[str, Any]] = []
        for s in statuses:
            payload.append(
                {
                    "name": s.name,
                    "module": s.module,
                    "available": s.available,
                    "error": s.error,
                    "missing_module": s.missing_module,
                    "extra_hint": s.extra_hint,
                }
            )
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for s in statuses:
            if s.available:
                typer.echo(f"{s.name}: ok")
                continue
            detail = f"unavailable ({s.error})" if s.error else "unavailable"
            typer.echo(f"{s.name}: {detail}")
            if s.missing_module:
                typer.echo(f"  missing module: {s.missing_module}")
            if s.extra_hint:
                typer.echo(f'  install: pip install "modssc[{s.extra_hint}]"')
    if any(not s.available for s in statuses):
        raise typer.Exit(code=2)


# Register subcommands if their bricks are present.
_try_add_typer("modssc.cli.datasets", "datasets", "Dataset management (data_loader brick).")
_try_add_typer("modssc.cli.sampling", "sampling", "Deterministic sampling and splits.")
_try_add_typer("modssc.cli.preprocess", "preprocess", "Preprocessing pipelines.")
_try_add_typer("modssc.cli.graph", "graph", "Graph construction and graph-derived views.")
_try_add_typer("modssc.cli.augmentation", "augmentation", "Data augmentation registry tools.")
_try_add_typer("modssc.cli.evaluation", "evaluation", "Metrics and evaluation tools.")
_try_add_typer("modssc.cli.inductive", "inductive", "Inductive SSL methods.")
_try_add_typer("modssc.cli.transductive", "transductive", "Transductive SSL methods.")
_try_add_typer("modssc.cli.supervised", "supervised", "Supervised baselines.")
