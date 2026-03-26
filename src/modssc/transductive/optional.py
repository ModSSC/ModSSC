from __future__ import annotations

from modssc.dependencies.optional import make_optional_import

from .errors import OptionalDependencyError


def _error_factory(**kwargs) -> Exception:
    exc = kwargs["exc"]
    package = kwargs.get("package_hint") or kwargs["module"].split(".")[0]
    return OptionalDependencyError(package=package, extra=kwargs["extra"], message=str(exc))


optional_import = make_optional_import(error_factory=_error_factory)
