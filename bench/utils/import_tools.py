from __future__ import annotations

import importlib
import importlib.util
import re
import tomllib
from pathlib import Path
from typing import Any

from modssc.utils.imports import load_object as _load_object

_PACKAGE_ALIASES = {
    "scikit-learn": "sklearn",
    "torch-geometric": "torch_geometric",
    "tensorflow-datasets": "tensorflow_datasets",
    "sentence-transformers": "sentence_transformers",
    "open-clip-torch": "open_clip",
    "Pillow": "PIL",
    "faiss-cpu": "faiss",
}


def load_object(path: str) -> Any:
    return _load_object(path, error_prefix="Invalid import path")


def _find_pyproject(start: Path | None = None) -> Path | None:
    current = start or Path.cwd()
    for parent in [current] + list(current.parents):
        cand = parent / "pyproject.toml"
        if cand.exists():
            return cand
    return None


def _read_optional_deps(pyproject_path: Path | None) -> dict[str, list[str]]:
    if pyproject_path is None:
        return {}
    raw = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = raw.get("project", {})
    opt = project.get("optional-dependencies", {})
    out: dict[str, list[str]] = {}
    for extra, items in opt.items():
        if not isinstance(items, list):
            continue
        out[str(extra)] = [str(item) for item in items]
    return out


def _pkg_name(spec: str) -> str:
    # Strip extras and version specifiers.
    name = spec.split("[")[0]
    match = re.match(r"^[A-Za-z0-9_.-]+", name)
    return match.group(0) if match else name


def _pkg_to_import(pkg: str) -> str:
    if pkg in _PACKAGE_ALIASES:
        return _PACKAGE_ALIASES[pkg]
    return pkg.replace("-", "_")


def check_extra_installed(extra: str, *, pyproject_path: Path | None = None) -> list[str]:
    pyproject = pyproject_path or _find_pyproject()
    optional_deps = _read_optional_deps(pyproject)
    if extra not in optional_deps:
        raise ValueError(f"Unknown optional dependency extra: {extra!r}")
    packages = optional_deps.get(extra, [])
    missing: list[str] = []
    for spec in packages:
        pkg = _pkg_name(spec)
        if not pkg:
            continue
        mod = _pkg_to_import(pkg)
        if importlib.util.find_spec(mod) is None:
            missing.append(mod)
    return missing
