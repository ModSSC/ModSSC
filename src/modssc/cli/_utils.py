from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
import yaml

DatasetCacheOption = Annotated[
    Path | None,
    typer.Option(
        "--dataset-cache-dir",
        "--dataset-cache",
        help="Dataset cache directory.",
    ),
]


def exit_with_error(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=2)


def load_yaml_or_json(path: Path, *, label: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        exit_with_error(f"Failed to read {label} file {path}: {exc}")
    try:
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return yaml.safe_load(text)
    except (ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        exit_with_error(f"Invalid {label} file: {exc}")


def ensure_mapping(obj: Any, *, message: str) -> dict[str, Any]:
    if obj is None:
        return {}
    if not isinstance(obj, Mapping):
        exit_with_error(message)
    return dict(obj)
