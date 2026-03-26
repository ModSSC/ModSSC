from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType


def has_module(module: str) -> bool:
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def make_optional_import(
    *,
    error_factory: Callable[..., Exception],
) -> Callable[..., ModuleType]:
    def optional_import(
        module: str,
        *,
        extra: str,
        package_hint: str | None = None,
        purpose: str | None = None,
        feature: str | None = None,
    ) -> ModuleType:
        try:
            return importlib.import_module(module)
        except (ModuleNotFoundError, ImportError) as exc:
            raise error_factory(
                module=module,
                extra=extra,
                package_hint=package_hint,
                purpose=purpose,
                feature=feature,
                exc=exc,
            ) from exc

    return optional_import


def make_optional_import_attr(
    *,
    optional_import: Callable[..., ModuleType],
    error_factory: Callable[..., Exception],
) -> Callable[..., object]:
    def optional_import_attr(
        module: str,
        attr: str,
        *,
        extra: str,
        package_hint: str | None = None,
        purpose: str | None = None,
        feature: str | None = None,
    ) -> object:
        mod = optional_import(
            module,
            extra=extra,
            package_hint=package_hint,
            purpose=purpose,
            feature=feature,
        )
        try:
            return getattr(mod, attr)
        except AttributeError as exc:
            raise error_factory(
                module=module,
                extra=extra,
                package_hint=package_hint,
                purpose=purpose,
                feature=feature,
                exc=exc,
            ) from exc

    return optional_import_attr
