"""Native construction of method dataclass specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from typing import Any, Literal

MethodSpecErrorKind = Literal["method_introspection", "method_spec"]


class MethodSpecError(ValueError):
    """Raised when a method's native dataclass spec cannot be constructed."""

    def __init__(self, kind: MethodSpecErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def method_spec_has_field(
    method_cls: type[Any],
    field_name: str,
    *,
    strict: bool = False,
) -> bool:
    """Return whether a method exposes a field on its native specification.

    Method instantiation and specification introspection are runtime behavior,
    so callers such as the YAML runner do not need to inspect method objects.
    """

    if not isinstance(field_name, str) or not field_name.strip():
        raise ValueError("field_name must be a non-empty string")
    try:
        instance = method_cls()
    except (TypeError, ValueError, RuntimeError, ImportError, ModuleNotFoundError) as exc:
        if strict:
            raise MethodSpecError(
                "method_introspection",
                f"failed to instantiate method for spec introspection: {exc}",
            ) from exc
        return False
    return hasattr(getattr(instance, "spec", None), field_name)


def build_method_spec(
    method_cls: type[Any],
    params: Mapping[str, Any] | None = None,
    *,
    require_spec: bool = False,
    strict: bool = False,
) -> Any | None:
    """Build a method's dataclass spec from native defaults and overrides.

    This helper is regime-independent: inductive and transductive runners can
    both use it without importing one another. ``require_spec`` materializes
    the default spec even when no user overrides are present.
    """

    overrides = dict(params or {})
    if not overrides and not require_spec:
        return None

    try:
        instance = method_cls()
    except (TypeError, ValueError, RuntimeError, ImportError, ModuleNotFoundError) as exc:
        if strict:
            raise MethodSpecError(
                "method_introspection",
                f"failed to instantiate method for spec introspection: {exc}",
            ) from exc
        instance = None

    spec = getattr(instance, "spec", None) if instance is not None else None
    if spec is None or not is_dataclass(spec):
        raise MethodSpecError(
            "method_spec",
            "method configuration requires a dataclass spec, but none is available",
        )

    try:
        return replace(spec, **overrides) if overrides else spec
    except TypeError as exc:
        raise MethodSpecError("method_spec", f"invalid method params: {exc}") from exc


__all__ = [
    "MethodSpecError",
    "MethodSpecErrorKind",
    "build_method_spec",
    "method_spec_has_field",
]
