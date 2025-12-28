from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def dump_yaml(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
