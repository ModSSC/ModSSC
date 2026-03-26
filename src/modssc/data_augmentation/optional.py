from __future__ import annotations

from modssc.dependencies.optional import make_optional_import

from .errors import OptionalDependencyError


def _error_factory(**kwargs) -> Exception:
    return OptionalDependencyError(extra=kwargs["extra"])


optional_import = make_optional_import(error_factory=_error_factory)
