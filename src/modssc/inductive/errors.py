from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionalDependencyError(ImportError):
    """Raised when an optional dependency (extra) is required but missing."""

    package: str
    extra: str
    message: str | None = None

    def __str__(self) -> str:
        base = self.message or f"Optional dependency {self.package!r} is required."
        return f'{base} Install with: pip install "modssc[{self.extra}]"'


class InductiveValidationError(ValueError):
    """Raised when inputs are invalid for inductive methods."""


class InductiveNotImplementedError(NotImplementedError):
    """Raised when a method is registered but not implemented yet."""

    def __init__(self, method_id: str, hint: str | None = None) -> None:
        msg = f"Inductive method {method_id!r} is registered but not implemented yet."
        if hint:
            msg = f"{msg} {hint}"
        super().__init__(msg)
        self.method_id = method_id
        self.hint = hint
