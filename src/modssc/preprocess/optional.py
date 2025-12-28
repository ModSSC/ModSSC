from __future__ import annotations

import importlib
from typing import Any

from modssc.preprocess.errors import OptionalDependencyError


def is_available(module: str) -> bool:
    """Return True if `module` can be imported."""
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def require(
    *,
    module: str,
    extra: str,
    purpose: str | None = None,
) -> Any:
    """Import `module` or raise OptionalDependencyError pointing to `extra`."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise OptionalDependencyError(extra=extra, purpose=purpose) from e
