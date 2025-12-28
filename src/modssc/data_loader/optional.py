from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from modssc.data_loader.errors import OptionalDependencyError


def optional_import(module: str, *, extra: str, purpose: str | None = None) -> ModuleType:
    """Import an optional dependency with an actionable error message."""
    try:
        return importlib.import_module(module)
    except (ModuleNotFoundError, ImportError) as e:
        raise OptionalDependencyError(extra=extra, purpose=purpose) from e


def optional_import_attr(module: str, attr: str, *, extra: str, purpose: str | None = None) -> Any:
    mod = optional_import(module, extra=extra, purpose=purpose)
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise OptionalDependencyError(extra=extra, purpose=purpose) from e
