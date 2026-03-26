from __future__ import annotations

import importlib as _importlib

from modssc.dependencies.optional import make_optional_import, make_optional_import_attr

from .errors import OptionalDependencyError

# Re-export `importlib` so existing tests and callers can monkeypatch the import
# hook through this module path.
importlib = _importlib


def _error_factory(**kwargs) -> Exception:
    exc = kwargs["exc"]
    return OptionalDependencyError(
        extra=kwargs["extra"],
        purpose=kwargs.get("purpose"),
        message=str(exc),
    )


optional_import = make_optional_import(error_factory=_error_factory)
optional_import_attr = make_optional_import_attr(
    optional_import=optional_import,
    error_factory=_error_factory,
)
