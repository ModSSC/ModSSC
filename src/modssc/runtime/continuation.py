from __future__ import annotations

import os
import signal
from types import FrameType, TracebackType
from typing import Final

PLANNED_CONTINUATION_EXIT_CODE: Final = 75
# Stable process status for a checkpointed, retryable continuation (EX_TEMPFAIL).

_REQUESTED_ENV: Final = "MODSSC_CONTINUATION_REQUESTED"
_SIGNAL_ENV: Final = "MODSSC_CONTINUATION_SIGNAL"


class PlannedContinuation(BaseException):
    """Cooperative, non-failure exit after a durable training checkpoint."""

    exit_code = PLANNED_CONTINUATION_EXIT_CODE

    def __init__(self, signum: int | None = None) -> None:
        resolved = signum
        if resolved is None:
            raw = os.environ.get(_SIGNAL_ENV)
            resolved = int(raw) if raw and raw.isdigit() else 0
        super().__init__(f"planned continuation requested by signal {resolved}")
        self.signum = int(resolved)


def continuation_requested() -> bool:
    """Return whether the current process requested a planned continuation."""

    return os.environ.get(_REQUESTED_ENV) == "1"


def clear_continuation_request() -> None:
    """Clear process-local continuation state before or after one execution."""

    os.environ.pop(_REQUESTED_ENV, None)
    os.environ.pop(_SIGNAL_ENV, None)


def request_continuation(signum: int) -> None:
    """Record a signal-safe request consumed at an iteration boundary."""

    os.environ[_REQUESTED_ENV] = "1"
    os.environ[_SIGNAL_ENV] = str(int(signum))


def _request_continuation_from_signal(signum: int, _frame: FrameType | None) -> None:
    request_continuation(signum)


class _ContinuationSignalHandler:
    """Context manager that never rewrites the exception it propagates."""

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self._signal_number: int | None = None
        self._previous_handler = signal.SIG_DFL

    def __enter__(self) -> None:
        clear_continuation_request()
        if not self._enabled:
            return

        continuation_signal = getattr(signal, "SIGUSR1", None)
        if continuation_signal is None:  # pragma: no cover - supported HPC targets are POSIX
            raise RuntimeError("cooperative continuation requires SIGUSR1 support")
        previous_handler = signal.getsignal(continuation_signal)
        signal.signal(continuation_signal, _request_continuation_from_signal)
        self._signal_number = int(continuation_signal)
        self._previous_handler = previous_handler

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        try:
            if self._signal_number is not None:
                signal.signal(self._signal_number, self._previous_handler)
        finally:
            clear_continuation_request()
        return False


def continuation_signal_handler(*, enabled: bool) -> _ContinuationSignalHandler:
    """Scope a cooperative ``SIGUSR1`` handler to one resumable execution.

    The handler only records a request. Training code must observe that request
    at a safe boundary, durably commit its checkpoint, and then raise
    :class:`PlannedContinuation`. The caller may translate that exception to
    :data:`PLANNED_CONTINUATION_EXIT_CODE` only when this scope is enabled.
    """

    return _ContinuationSignalHandler(enabled=enabled)


def raise_planned_continuation() -> None:
    """Exit cooperatively after the caller persisted a complete checkpoint."""

    if not continuation_requested():
        raise RuntimeError("planned continuation was not requested")
    raise PlannedContinuation()


__all__ = [
    "PLANNED_CONTINUATION_EXIT_CODE",
    "PlannedContinuation",
    "clear_continuation_request",
    "continuation_signal_handler",
    "continuation_requested",
    "raise_planned_continuation",
    "request_continuation",
]
