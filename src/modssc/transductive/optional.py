from __future__ import annotations

from importlib import import_module
from types import ModuleType

from .errors import OptionalDependencyError


def optional_import(module: str, *, extra: str, package_hint: str | None = None) -> ModuleType:
    """Import a module, raising a friendly OptionalDependencyError if missing."""
    try:
        return import_module(module)
    except Exception as e:  # pragma: no cover
        pkg = package_hint or module.split(".")[0]
        raise OptionalDependencyError(pkg, extra, message=str(e)) from e
