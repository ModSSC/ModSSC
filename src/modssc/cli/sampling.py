from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from modssc.cli._utils import DatasetCacheOption, ensure_mapping, exit_with_error, load_yaml_or_json
from modssc.data_loader import load_dataset
from modssc.data_loader.errors import DataLoaderError
from modssc.logging import LogLevelOption, add_log_level_callback, configure_logging
from modssc.sampling.api import load_split, sample, save_split
from modssc.sampling.plan import (
    SamplingPlan,
)

app = typer.Typer(help="Sampling commands (split protocols, labeled/unlabeled).")
add_log_level_callback(app)

logger = logging.getLogger(__name__)


def _plan_from_path(path: Path) -> SamplingPlan:
    obj = load_yaml_or_json(path, label="plan")
    obj = ensure_mapping(obj, message="Plan file must contain a mapping at the root")
    try:
        return SamplingPlan.from_dict(obj)
    except ValueError as exc:
        exit_with_error(str(exc))


@app.command("create")
def create(
    dataset_id: str = typer.Option(..., "--dataset"),  # noqa: B008
    plan_path: Path = typer.Option(  # noqa: B008
        ..., "--plan", exists=True, dir_okay=False
    ),
    seed: int = typer.Option(0, "--seed"),  # noqa: B008
    out_dir: Path = typer.Option(..., "--out"),  # noqa: B008
    overwrite: bool = typer.Option(False, "--overwrite"),  # noqa: B008
    dataset_cache_dir: DatasetCacheOption = None,  # noqa: B008
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    try:
        ds = load_dataset(dataset_id, cache_dir=dataset_cache_dir, download=True)
    except DataLoaderError as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Sampling create failed while loading dataset %s", dataset_id)
        exit_with_error(str(exc))
    ds_fp = ds.meta.get("dataset_fingerprint") if isinstance(ds.meta, dict) else None
    if ds_fp is None:
        exit_with_error(
            "Dataset fingerprint is missing in dataset.meta. Pass it explicitly in code."
        )
    plan = _plan_from_path(plan_path)
    result, _ = sample(ds, plan=plan, seed=seed, dataset_fingerprint=str(ds_fp), save=False)
    save_split(result, out_dir, overwrite=overwrite)
    payload = {
        "output_dir": str(out_dir),
        "dataset_fingerprint": result.dataset_fingerprint,
        "split_fingerprint": result.split_fingerprint,
        "stats": result.stats,
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("show")
def show(split_dir: Path, log_level: LogLevelOption = None) -> None:
    if log_level is not None:
        configure_logging(log_level)
    res = load_split(split_dir)
    typer.echo(json.dumps({"plan": res.plan, "stats": res.stats}, indent=2, sort_keys=True))


@app.command("validate")
def validate(
    split_dir: Path,
    dataset_id: str = typer.Option(..., "--dataset"),  # noqa: B008
    dataset_cache_dir: DatasetCacheOption = None,  # noqa: B008
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    try:
        ds = load_dataset(dataset_id, cache_dir=dataset_cache_dir, download=False)
    except DataLoaderError as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Sampling validate failed while loading dataset %s", dataset_id)
        exit_with_error(str(exc))
    res = load_split(split_dir)
    y_train = ds.train.y
    n_train = len(y_train)
    n_test = len(ds.test.y) if ds.test is not None else None
    n_nodes = (
        len(ds.train.y)
        if (
            getattr(ds.train, "edges", None) is not None
            or getattr(ds.train, "masks", None) is not None
        )
        else None
    )
    res.validate(n_train=n_train, n_test=n_test, n_nodes=n_nodes)
    typer.echo("OK")
