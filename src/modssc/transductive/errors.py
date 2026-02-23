from __future__ import annotations

from dataclasses import dataclass

from modssc.dependency_errors import PackageOptionalDependencyMessageMixin


@dataclass(frozen=True)
class OptionalDependencyError(PackageOptionalDependencyMessageMixin, ImportError):
    """Raised when an optional dependency (extra) is required but missing."""

    package: str
    extra: str
    message: str | None = None


class TransductiveValidationError(ValueError):
    """Raised when inputs are invalid for transductive methods."""


class TransductiveNotImplementedError(NotImplementedError):
    """Raised when a transductive method is registered but not implemented yet."""

    def __init__(self, method_id: str, hint: str | None = None) -> None:
        msg = f"Transductive method {method_id!r} is registered but not implemented yet."
        if hint:
            msg = f"{msg} {hint}"
        super().__init__(msg)
        self.method_id = method_id
        self.hint = hint
