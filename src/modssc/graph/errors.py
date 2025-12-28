from __future__ import annotations


class GraphError(Exception):
    """Base exception for modssc.graph."""


class GraphValidationError(GraphError):
    """Raised when graph inputs or specs are invalid."""


class GraphCacheError(GraphError):
    """Raised when a cached graph entry is missing or corrupted."""


class OptionalDependencyError(GraphError):
    """Raised when an optional dependency is required but not installed."""

    def __init__(
        self,
        *,
        extra: str,
        purpose: str | None = None,
        message: str | None = None,
    ):
        self.extra = extra
        self.purpose = purpose

        msg = f"Missing optional dependency extra: {extra!r}."
        if purpose:
            msg += f" Required for: {purpose}."
        msg += f' Install with: pip install "modssc[{extra}]"'
        if message:
            msg += f" (Import error: {message})"
        super().__init__(msg)
