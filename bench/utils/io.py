from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

_PLACEHOLDER_RE = re.compile(
    r"(?<![$\\])\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))"
)


def _iter_placeholder_names(text: str) -> set[str]:
    names: set[str] = set()
    for match in _PLACEHOLDER_RE.finditer(text):
        name = match.group(1) or match.group(2)
        if name:
            names.add(name)
    return names


def _format_path(path: tuple[str | int, ...]) -> str:
    if not path:
        return "<root>"
    out: list[str] = []
    for part in path:
        if isinstance(part, int):
            if out:
                out[-1] = f"{out[-1]}[{part}]"
            else:
                out.append(f"[{part}]")
            continue
        out.append(part)
    return ".".join(out)


def _expand_env_vars(value: Any, *, path: tuple[str | int, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_vars(item, path=path + (str(key),)) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item, path=path + (i,)) for i, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(_expand_env_vars(item, path=path + (i,)) for i, item in enumerate(value))
    if isinstance(value, str):
        # Allow ${VAR} or $VAR placeholders in YAML configs, but fail fast when missing.
        missing = sorted(name for name in _iter_placeholder_names(value) if name not in os.environ)
        if missing:
            where = _format_path(path)
            names = ", ".join(missing)
            raise ValueError(
                f"Unresolved environment variable(s) at {where}: {names}. "
                "Export them or remove the placeholders."
            )
        return os.path.expandvars(value)
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return _expand_env_vars(data)


def dump_yaml(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
