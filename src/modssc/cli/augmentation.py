from __future__ import annotations

import json

import typer

from modssc.data_augmentation.registry import available_ops, op_info
from modssc.logging import LogLevelOption, add_log_level_callback, configure_logging

app = typer.Typer(no_args_is_help=True, help="Data augmentation: list ops and inspect defaults.")
add_log_level_callback(app)


@app.command("list")
def list_ops(
    modality: str | None = typer.Option(
        None, "--modality", help="Filter by modality: vision/text/tabular/audio/graph"
    ),
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    ops = available_ops(modality=modality) if modality else available_ops()
    for op_id in ops:
        typer.echo(op_id)


@app.command("info")
def info(
    op_id: str = typer.Argument(..., help="Operation id"),
    as_json: bool = False,
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    info = op_info(op_id)
    if as_json:
        typer.echo(json.dumps(info, indent=2))
        return
    typer.echo(f"op_id: {info['op_id']}")
    typer.echo(f"modality: {info['modality']}")
    if info.get("doc"):
        typer.echo("\n" + info["doc"])
    typer.echo("\nDefaults:")
    typer.echo(json.dumps(info.get("defaults", {}), indent=2))
