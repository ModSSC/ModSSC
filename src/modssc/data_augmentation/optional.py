from __future__ import annotations

from importlib import import_module
from typing import Any

from .errors import OptionalDependencyError


def optional_import(module: str, *, extra: str) -> Any:
    """Import *module* or raise :class:`OptionalDependencyError` with the given extra name."""
    try:
        return import_module(module)
    except Exception as e:  # pragma: no cover (import failures vary by env)
        raise OptionalDependencyError(extra=extra) from e
