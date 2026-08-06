"""Backward-compatible imports for the public continuation runtime."""

from modssc.runtime.continuation import (
    PlannedContinuation,
    continuation_requested,
    raise_planned_continuation,
    request_continuation,
)

__all__ = [
    "PlannedContinuation",
    "continuation_requested",
    "raise_planned_continuation",
    "request_continuation",
]
