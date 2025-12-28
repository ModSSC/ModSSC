from __future__ import annotations


class ViewsError(Exception):
    """Base exception for the `modssc.views` brick."""


class ViewsValidationError(ViewsError, ValueError):
    """Raised when a ViewsPlan / ViewSpec is invalid."""
