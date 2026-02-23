from __future__ import annotations

from modssc.optional_import_utils import make_optional_import

from .errors import OptionalDependencyError

optional_import = make_optional_import(error_cls=OptionalDependencyError)
