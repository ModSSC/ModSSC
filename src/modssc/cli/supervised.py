from __future__ import annotations

import json

import typer

from modssc.logging import LogLevelOption, add_log_level_callback, configure_logging
from modssc.supervised import available_classifiers, classifier_info

app = typer.Typer(help="Supervised baselines (kNN, SVM, logistic regression).")
add_log_level_callback(app)


@app.command("list")
def list_classifiers(
    available_only: bool = typer.Option(  # noqa: B008
        False, "--available-only", help="Show only backends that are importable."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print as JSON"),  # noqa: B008
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    items = available_classifiers(available_only=bool(available_only))
    if as_json:
        typer.echo(json.dumps(items, indent=2))
        raise typer.Exit(0)

    for spec in items:
        typer.echo(f"{spec['key']}: {spec['description']}")
        backends = spec.get("backends", {})
        for b, bs in backends.items():
            extra = bs.get("required_extra")
            extra_txt = f" (extra={extra})" if extra else ""
            typer.echo(f"  {b}{extra_txt}")


@app.command("info")
def info(
    classifier_id: str = typer.Argument(..., help="Classifier id, ex: knn, svm_rbf, logreg"),
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    typer.echo(json.dumps(classifier_info(classifier_id), indent=2))
