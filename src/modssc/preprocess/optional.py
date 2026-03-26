from __future__ import annotations

from typing import Any

from modssc.dependencies.optional import has_module, make_optional_import
from modssc.preprocess.errors import OptionalDependencyError


def _error_factory(**kwargs) -> Exception:
    return OptionalDependencyError(extra=kwargs["extra"], purpose=kwargs.get("purpose"))


is_available = has_module
_require_impl = make_optional_import(error_factory=_error_factory)


def require(*, module: str, extra: str, purpose: str | None = None) -> Any:
    """Import `module` or raise OptionalDependencyError pointing to `extra`."""
    return _require_impl(module, extra=extra, purpose=purpose)
