"""CLI for the transductive SSL brick."""

from __future__ import annotations

import json
from dataclasses import asdict

import typer

from modssc.cli._utils import exit_with_error
from modssc.logging import LogLevelOption, add_log_level_callback, configure_logging
from modssc.transductive import registry as transductive_registry

app = typer.Typer(help="Transductive SSL methods.")
add_log_level_callback(app)

methods_app = typer.Typer(help="List and inspect transductive methods.")
app.add_typer(methods_app, name="methods")


@methods_app.command("list")
def methods_list(
    include_planned: bool = typer.Option(
        True,
        "--all/--available-only",
        help="Include planned/unimplemented methods.",
    ),
    log_level: LogLevelOption = None,
) -> None:
    """List registered transductive methods."""
    if log_level is not None:
        configure_logging(log_level)
    methods = transductive_registry.available_methods(available_only=not include_planned)
    for k in methods:
        typer.echo(k)


@methods_app.command("info")
def methods_info(
    method_id: str = typer.Argument(..., help="Method id"),
    log_level: LogLevelOption = None,
) -> None:
    """Show method metadata as JSON (when resolvable)."""
    if log_level is not None:
        configure_logging(log_level)
    registry = transductive_registry._debug_registry()
    import_path = registry.get(method_id)
    if import_path is None:
        exit_with_error(f"Unknown method id: {method_id!r}")

    payload = {"method_id": method_id, "import_path": import_path}
    try:
        info = transductive_registry.get_method_info(method_id)
    except Exception as exc:  # optional deps or missing class
        payload["error"] = str(exc)
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        raise typer.Exit(code=2) from exc

    payload["info"] = asdict(info)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
