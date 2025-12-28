from __future__ import annotations

from dataclasses import dataclass


class PreprocessError(RuntimeError):
    """Base error for the preprocess module."""


class PreprocessValidationError(PreprocessError):
    """Raised when invariants are violated (alignment, shapes, plan constraints, etc.)."""


class PreprocessCacheError(PreprocessError):
    """Raised when cached artifacts are missing, corrupted, or inconsistent."""


@dataclass(frozen=True)
class OptionalDependencyError(PreprocessError):
    """Raised when an optional dependency required by a step/model is missing."""

    extra: str
    purpose: str | None = None

    def __str__(self) -> str:
        msg = f"Missing optional dependency extra: {self.extra!r}."
        if self.purpose:
            msg += f" Required for: {self.purpose}."
        msg += f' Install with: pip install "modssc[{self.extra}]"'
        return msg
