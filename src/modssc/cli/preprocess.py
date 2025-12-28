from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import yaml

from modssc.cli._utils import DatasetCacheOption, exit_with_error
from modssc.data_loader import load_dataset
from modssc.data_loader.errors import DataLoaderError
from modssc.logging import LogLevelOption, add_log_level_callback, configure_logging
from modssc.preprocess import available_models, available_steps, model_info, preprocess, step_info
from modssc.preprocess.errors import PreprocessError
from modssc.preprocess.plan import load_plan

app = typer.Typer(help="Preprocessing pipelines (deterministic, cacheable).")
add_log_level_callback(app)
steps_app = typer.Typer(help="Step registry utilities.")
models_app = typer.Typer(help="Pretrained encoder registry utilities.")

app.add_typer(steps_app, name="steps")
app.add_typer(models_app, name="models")

logger = logging.getLogger(__name__)


@steps_app.command("list")
def steps_list(
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    items = available_steps()
    if json_output:
        typer.echo(json.dumps(items))
        return
    for s in items:
        typer.echo(s)


@steps_app.command("info")
def steps_info(step_id: str, log_level: LogLevelOption = None) -> None:
    if log_level is not None:
        configure_logging(log_level)
    typer.echo(json.dumps(step_info(step_id), indent=2))


@models_app.command("list")
def models_list(
    modality: str | None = typer.Option(None, "--modality", help="Filter by modality."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    items = available_models(modality=modality)
    if json_output:
        typer.echo(json.dumps(items))
        return
    for m in items:
        typer.echo(m)


@models_app.command("info")
def models_info(model_id: str, log_level: LogLevelOption = None) -> None:
    if log_level is not None:
        configure_logging(log_level)
    typer.echo(json.dumps(model_info(model_id), indent=2))


@app.command("run")
def run(
    plan_path: Annotated[
        Path, typer.Option(..., "--plan", exists=True, dir_okay=False, help="YAML plan file.")
    ],
    dataset: Annotated[
        str, typer.Option("--dataset", help="Dataset key from modssc.data_loader.")
    ] = "toy",
    dataset_cache_dir: DatasetCacheOption = None,
    seed: Annotated[int, typer.Option("--seed", help="Master seed.")] = 0,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Disable cache.")] = False,
    purge_unused: Annotated[
        bool,
        typer.Option(
            "--purge-unused",
            help="Drop artifacts not needed by downstream steps to reduce RAM.",
        ),
    ] = False,
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    try:
        ds = load_dataset(dataset, cache_dir=dataset_cache_dir)
    except DataLoaderError as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Preprocess run failed while loading dataset %s", dataset)
        exit_with_error(str(exc))
    try:
        plan = load_plan(plan_path)
    except (yaml.YAMLError, ValueError) as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Preprocess run failed while parsing plan %s", plan_path)
        exit_with_error(f"Invalid plan file: {exc}")
    fit_idx = np.arange(ds.train.y.shape[0], dtype=np.int64)
    try:
        result = preprocess(
            ds,
            plan,
            seed=seed,
            fit_indices=fit_idx,
            cache=not no_cache,
            purge_unused_artifacts=purge_unused,
        )
    except PreprocessError as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Preprocess run failed for dataset %s", dataset)
        exit_with_error(str(exc))
    out = result.dataset
    xshape = getattr(out.train.X, "shape", None)
    typer.echo(
        json.dumps(
            {
                "dataset": dataset,
                "output_key": plan.output_key,
                "train_X_shape": xshape,
                "preprocess_fingerprint": result.preprocess_fingerprint,
                "cache_dir": result.cache_dir,
            },
            indent=2,
        )
    )
