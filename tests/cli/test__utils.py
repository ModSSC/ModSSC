from __future__ import annotations

from pathlib import Path

import pytest
import typer

from modssc.cli._utils import ensure_mapping, load_yaml_or_json


def test_load_yaml_or_json_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{")
    with pytest.raises(typer.Exit) as exc:
        load_yaml_or_json(path, label="plan")
    assert exc.value.exit_code == 2


def test_load_yaml_or_json_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("a: [1, 2")
    with pytest.raises(typer.Exit) as exc:
        load_yaml_or_json(path, label="plan")
    assert exc.value.exit_code == 2


def test_load_yaml_or_json_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    with pytest.raises(typer.Exit) as exc:
        load_yaml_or_json(path, label="plan")
    assert exc.value.exit_code == 2


def test_ensure_mapping_none_returns_empty() -> None:
    assert ensure_mapping(None, message="boom") == {}
