from __future__ import annotations

from modssc.data_loader.errors import OptionalDependencyError
from modssc.dependencies.optional import make_optional_import, make_optional_import_attr


def _error_factory(**kwargs) -> Exception:
    return OptionalDependencyError(extra=kwargs["extra"], purpose=kwargs.get("purpose"))


optional_import = make_optional_import(error_factory=_error_factory)
optional_import_attr = make_optional_import_attr(
    optional_import=optional_import,
    error_factory=_error_factory,
)
