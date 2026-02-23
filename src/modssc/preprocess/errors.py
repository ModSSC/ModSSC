from __future__ import annotations

from dataclasses import dataclass

from modssc.dependency_errors import MissingExtraErrorMessageMixin


class PreprocessError(RuntimeError):
    """Base error for the preprocess module."""


class PreprocessValidationError(PreprocessError):
    """Raised when invariants are violated (alignment, shapes, plan constraints, etc.)."""


class PreprocessCacheError(PreprocessError):
    """Raised when cached artifacts are missing, corrupted, or inconsistent."""


@dataclass(frozen=True)
class OptionalDependencyError(MissingExtraErrorMessageMixin, PreprocessError):
    """Raised when an optional dependency required by a step/model is missing."""

    extra: str
    purpose: str | None = None
