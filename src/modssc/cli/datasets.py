from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from modssc.cli._utils import exit_with_error
from modssc.data_loader import api
from modssc.data_loader.errors import DataLoaderError
from modssc.logging import LogLevelOption, add_log_level_callback, configure_logging

app = typer.Typer(help="Dataset commands (download, cache, info).")
add_log_level_callback(app)

logger = logging.getLogger(__name__)


@app.command("providers")
def providers(log_level: LogLevelOption = None) -> None:
    if log_level is not None:
        configure_logging(log_level)
    names = api.provider_names()
    for n in names:
        typer.echo(n)


@app.command("list")
def list_datasets(
    modalities: list[str] | None = typer.Option(None, "--modalities"),  # noqa: B008
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    keys = api.available_datasets()
    if modalities:
        wanted = set(modalities)
        try:
            keys = [k for k in keys if api.dataset_info(k).modality in wanted]
        except DataLoaderError as exc:
            if logger.isEnabledFor(logging.DEBUG):
                logger.exception("Failed to resolve dataset info for modalities filter")
            exit_with_error(str(exc))
    for k in keys:
        typer.echo(k)


@app.command("info")
def info(
    dataset_id: str = typer.Option(..., "--dataset"), log_level: LogLevelOption = None
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    try:
        spec = api.dataset_info(dataset_id)
    except DataLoaderError as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Dataset info failed for %s", dataset_id)
        exit_with_error(str(exc))
    typer.echo(json.dumps(spec.as_dict(), indent=2, sort_keys=True))


@app.command("download")
def download(
    dataset_id: str | None = typer.Option(None, "--dataset"),
    all: bool = typer.Option(False, "--all"),  # noqa: B008
    force: bool = typer.Option(False, "--force"),  # noqa: B008
    cache_dir: Path | None = typer.Option(None, "--cache-dir", "--dataset-cache-dir"),  # noqa: B008
    ignore_missing_extras: bool = typer.Option(True, "--ignore-missing-extras"),  # noqa: B008
    skip_cached: bool = typer.Option(False, "--skip-cached"),  # noqa: B008
    modalities: list[str] | None = typer.Option(None, "--modalities"),  # noqa: B008
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    try:
        if all:
            report = api.download_all_datasets(
                cache_dir=cache_dir,
                force=force,
                ignore_missing_extras=ignore_missing_extras,
                skip_cached=skip_cached,
                modalities=modalities,
            )
            typer.echo(report.summary())
            if report.missing_extras:
                typer.echo("")
                typer.echo("Missing extras:")
                for extra, keys in sorted(report.missing_extras.items()):
                    typer.echo(f"{extra}: {', '.join(keys)}")
            if report.failed:
                typer.echo("")
                typer.echo("Failed:")
                for k, err in sorted(report.failed.items()):
                    typer.echo(f"{k}: {err}")
                raise typer.Exit(code=2)
            return

        if dataset_id is None:
            exit_with_error("Provide --dataset or use --all")

        api.download_dataset(dataset_id, cache_dir=cache_dir, force=force)
        typer.echo(f"Downloaded: {dataset_id}")

    except DataLoaderError as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Dataset download failed")
        exit_with_error(str(exc))


cache_app = typer.Typer(help="Cache inspection and maintenance.")
app.add_typer(cache_app, name="cache")


@cache_app.command("ls")
def cache_ls(
    cache_dir: Path | None = typer.Option(None, "--cache-dir", "--dataset-cache-dir"),  # noqa: B008
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    layout = api._layout(cache_dir)  # internal helper is stable enough for CLI
    rows = api.cache.index_list(layout)
    for r in rows:
        typer.echo(f"{r['fingerprint']} {r['canonical_uri']} {r.get('size_bytes', 0)}")


@cache_app.command("purge")
def cache_purge(
    dataset_or_fp: str,
    cache_dir: Path | None = typer.Option(None, "--cache-dir", "--dataset-cache-dir"),  # noqa: B008
    fingerprint: bool = typer.Option(False, "--fingerprint"),  # noqa: B008
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    layout = api._layout(cache_dir)
    if fingerprint:
        api.cache.purge_fingerprint(layout, dataset_or_fp)
        typer.echo(f"Purged fingerprint: {dataset_or_fp}")
    else:
        fps = api.cache.purge_dataset(layout, dataset_or_fp)
        typer.echo(f"Purged variants: {len(fps)}")


@cache_app.command("gc")
def cache_gc(
    keep_latest: bool = typer.Option(True, "--keep-latest/--no-keep-latest"),  # noqa: B008
    cache_dir: Path | None = typer.Option(None, "--cache-dir", "--dataset-cache-dir"),  # noqa: B008
    log_level: LogLevelOption = None,
) -> None:
    if log_level is not None:
        configure_logging(log_level)
    layout = api._layout(cache_dir)
    if keep_latest:
        removed = api.cache.gc_keep_latest(layout)
        typer.echo(f"Removed: {len(removed)}")
    else:
        typer.echo("Nothing to do.")
