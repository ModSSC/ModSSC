from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from modssc.cli._utils import exit_with_error
from modssc.evaluation import evaluate, list_metrics
from modssc.logging import LogLevelOption, add_log_level_callback, configure_logging

app = typer.Typer(help="Metrics and evaluation tools.")
add_log_level_callback(app)

DEFAULT_METRICS = ("accuracy", "macro_f1", "balanced_accuracy")


def _load_npy(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        raise typer.BadParameter("Only .npy is supported")
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"Failed to load {path}: {exc}") from exc


@app.command("list")
def list_available(
    json_output: Annotated[bool, typer.Option("--json", help="Output JSON.")] = False,
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    metrics = list_metrics()
    if json_output:
        typer.echo(json.dumps({"metrics": metrics}, indent=2, sort_keys=True))
        return
    for metric in metrics:
        typer.echo(metric)


@app.command("compute")
def compute(
    y_true_path: Annotated[
        Path,
        typer.Option(..., "--y-true", exists=True, dir_okay=False, help="Path to a .npy file."),
    ],
    y_pred_path: Annotated[
        Path,
        typer.Option(..., "--y-pred", exists=True, dir_okay=False, help="Path to a .npy file."),
    ],
    metrics: Annotated[
        list[str] | None,
        typer.Option("--metric", "-m", help="Metric name (repeatable)."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output JSON.")] = False,
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    y_true = _load_npy(y_true_path)
    y_pred = _load_npy(y_pred_path)
    metrics_list = list(DEFAULT_METRICS) if metrics is None else metrics
    try:
        results = evaluate(y_true, y_pred, metrics_list)
    except ValueError as exc:
        exit_with_error(str(exc))
    if json_output:
        typer.echo(json.dumps(results, indent=2, sort_keys=True))
        return
    for name, value in results.items():
        typer.echo(f"{name}: {value}")
