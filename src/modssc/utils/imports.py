from __future__ import annotations

import importlib
from typing import Any


def load_object(import_path: str, *, error_prefix: str = "Invalid import path") -> Any:
    """Load an object from ``module:qualname``."""
    if ":" not in import_path:
        raise ValueError(f"{error_prefix}: {import_path!r}")
    module_name, qualname = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj
