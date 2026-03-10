from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from modssc.paths import default_local_cache_root

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


def _default_modssc_env(config_path: Path) -> dict[str, str]:
    root_override = os.environ.get("MODSSC_CACHE_ROOT")
    if root_override:
        cache_root = Path(root_override).expanduser().resolve()
    else:
        cache_root = default_local_cache_root(start=config_path.parent)
        if cache_root is None:
            return {}

    dataset_override = os.environ.get("MODSSC_CACHE_DIR")
    dataset_dir = (
        Path(dataset_override).expanduser().resolve()
        if dataset_override
        else cache_root / "datasets"
    )
    return {
        "MODSSC_CACHE_ROOT": str(cache_root),
        "MODSSC_OUTPUT_DIR": str(cache_root / "output"),
        "MODSSC_DATASET_CACHE_DIR": str(dataset_dir),
        "MODSSC_CACHE_DIR": str(dataset_dir),
        "MODSSC_PREPROCESS_CACHE_DIR": str(cache_root / "preprocess"),
        "MODSSC_SPLIT_CACHE_DIR": str(cache_root / "splits"),
        "MODSSC_GRAPH_CACHE_DIR": str(cache_root / "graphs"),
        "MODSSC_GRAPH_VIEWS_CACHE_DIR": str(cache_root / "graph_views"),
    }


def _resolve_env_var(name: str, defaults: Mapping[str, str]) -> str | None:
    if name in os.environ:
        return os.environ[name]
    return defaults.get(name)


def _expand_placeholders(text: str, *, defaults: Mapping[str, str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if not name:
            return match.group(0)
        resolved = _resolve_env_var(name, defaults)
        return resolved if resolved is not None else match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, text)


def _expand_env_vars(
    value: Any,
    *,
    path: tuple[str | int, ...] = (),
    defaults: Mapping[str, str],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _expand_env_vars(item, path=path + (str(key),), defaults=defaults)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_env_vars(item, path=path + (i,), defaults=defaults)
            for i, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _expand_env_vars(item, path=path + (i,), defaults=defaults)
            for i, item in enumerate(value)
        )
    if isinstance(value, str):
        # Allow ${VAR} or $VAR placeholders in YAML configs, but fail fast when missing.
        missing = sorted(
            name
            for name in _iter_placeholder_names(value)
            if _resolve_env_var(name, defaults) is None
        )
        if missing:
            where = _format_path(path)
            names = ", ".join(missing)
            raise ValueError(
                f"Unresolved environment variable(s) at {where}: {names}. "
                "Export them or remove the placeholders."
            )
        return _expand_placeholders(value, defaults=defaults)
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    defaults = _default_modssc_env(p)
    return _expand_env_vars(data, defaults=defaults)


def dump_yaml(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
