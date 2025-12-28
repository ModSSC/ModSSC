from __future__ import annotations


class DataAugmentationError(Exception):
    """Base error for the data augmentation brick."""


class DataAugmentationValidationError(DataAugmentationError, ValueError):
    """Raised when a plan/op configuration is invalid."""


class OptionalDependencyError(DataAugmentationError, ImportError):
    """Raised when an optional dependency is required but not installed."""

    def __init__(self, extra: str, message: str | None = None) -> None:
        self.extra = extra
        super().__init__(message or f"Missing optional dependency. Install extras: {extra!r}.")
