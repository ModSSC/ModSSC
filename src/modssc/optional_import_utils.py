from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType


def make_optional_import(*, error_cls: type[Exception]) -> Callable[..., ModuleType]:
    def optional_import(module: str, *, extra: str, package_hint: str | None = None) -> ModuleType:
        try:
            return import_module(module)
        except Exception as e:  # pragma: no cover
            pkg = package_hint or module.split(".")[0]
            raise error_cls(pkg, extra, message=str(e)) from e

    return optional_import
