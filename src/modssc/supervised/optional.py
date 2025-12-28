from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from modssc.supervised.errors import OptionalDependencyError


def optional_import(module: str, *, extra: str, feature: str) -> ModuleType:
    """Import a module that may be missing.

    Parameters
    ----------
    module:
        Import path, for example "sklearn".
    extra:
        The name of the optional dependency group in pyproject.toml.
    feature:
        Human-readable feature name used in error messages.
    """
    try:
        return importlib.import_module(module)
    except Exception as e:
        raise OptionalDependencyError(extra=extra, feature=feature) from e


def has_module(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)
