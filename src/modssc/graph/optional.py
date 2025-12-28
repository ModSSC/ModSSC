from __future__ import annotations

import importlib
from typing import Any

from .errors import OptionalDependencyError


def optional_import(module: str, *, extra: str, purpose: str | None = None) -> Any:
    """Import an optional dependency.

    Parameters
    ----------
    module:
        Module name, e.g. "scipy".
    extra:
        The extra name to suggest in error messages.
    purpose:
        Optional short description of why the dependency is needed (for error messages).

    Returns
    -------
    Any
        Imported module.

    Raises
    ------
    OptionalDependencyError
        If the module cannot be imported.
    """
    try:
        return importlib.import_module(module)
    except (ModuleNotFoundError, ImportError) as e:
        raise OptionalDependencyError(extra=extra, purpose=purpose, message=str(e)) from e


def optional_import_attr(module: str, attr: str, *, extra: str, purpose: str | None = None) -> Any:
    """Import an attribute from an optional dependency."""
    mod = optional_import(module, extra=extra, purpose=purpose)
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise OptionalDependencyError(extra=extra, purpose=purpose, message=str(e)) from e
