from __future__ import annotations

import os


class PlannedContinuation(BaseException):
    """Cooperative, non-failure exit after a durable training checkpoint."""

    def __init__(self, signum: int | None = None) -> None:
        resolved = signum
        if resolved is None:
            raw = os.environ.get("MODSSC_CONTINUATION_SIGNAL")
            resolved = int(raw) if raw and raw.isdigit() else 0
        super().__init__(f"planned continuation requested by signal {resolved}")
        self.signum = int(resolved)


def continuation_requested() -> bool:
    """Return whether the current process requested a planned continuation."""

    return os.environ.get("MODSSC_CONTINUATION_REQUESTED") == "1"


def request_continuation(signum: int) -> None:
    """Record a signal-safe request consumed at an iteration boundary."""

    os.environ["MODSSC_CONTINUATION_REQUESTED"] = "1"
    os.environ["MODSSC_CONTINUATION_SIGNAL"] = str(int(signum))


def raise_planned_continuation() -> None:
    """Exit cooperatively after the caller persisted a complete checkpoint."""

    if not continuation_requested():
        raise RuntimeError("planned continuation was not requested")
    raise PlannedContinuation()


__all__ = [
    "PlannedContinuation",
    "continuation_requested",
    "raise_planned_continuation",
    "request_continuation",
]
