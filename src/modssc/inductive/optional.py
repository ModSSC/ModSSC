from __future__ import annotations

from types import ModuleType

from modssc.optional_import_utils import make_optional_import

from .errors import OptionalDependencyError

optional_import = make_optional_import(error_cls=OptionalDependencyError)


def import_torch(*, extra: str = "inductive-torch") -> ModuleType:
    return optional_import("torch", extra=extra, package_hint="torch")
