from __future__ import annotations

import importlib.util
import signal
from pathlib import Path
from types import ModuleType

import pytest

from modssc.runtime import continuation as continuation_module
from modssc.runtime.continuation import (
    PlannedContinuation,
    continuation_requested,
    raise_planned_continuation,
    request_continuation,
)


def test_request_and_raise_planned_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODSSC_CONTINUATION_REQUESTED", raising=False)
    monkeypatch.delenv("MODSSC_CONTINUATION_SIGNAL", raising=False)

    assert not continuation_requested()
    with pytest.raises(RuntimeError, match="was not requested"):
        raise_planned_continuation()

    request_continuation(signal.SIGUSR1)

    assert continuation_requested()
    with pytest.raises(PlannedContinuation) as raised:
        raise_planned_continuation()
    assert raised.value.signum == signal.SIGUSR1
    assert str(raised.value) == f"planned continuation requested by signal {signal.SIGUSR1}"


@pytest.mark.parametrize(("raw", "expected"), [("invalid", 0), (None, 0), ("12", 12)])
def test_planned_continuation_resolves_environment_signal(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
    expected: int,
) -> None:
    if raw is None:
        monkeypatch.delenv("MODSSC_CONTINUATION_SIGNAL", raising=False)
    else:
        monkeypatch.setenv("MODSSC_CONTINUATION_SIGNAL", raw)

    assert PlannedContinuation().signum == expected
    assert PlannedContinuation(9).signum == 9


def test_legacy_campaign_module_is_a_compatibility_shim() -> None:
    path = Path(__file__).resolve().parents[2] / "bench" / "campaign" / "continuation.py"
    spec = importlib.util.spec_from_file_location("legacy_continuation", path)
    assert spec is not None and spec.loader is not None
    legacy_continuation = ModuleType(spec.name)
    spec.loader.exec_module(legacy_continuation)

    assert legacy_continuation.PlannedContinuation is PlannedContinuation
    assert legacy_continuation.continuation_requested is continuation_requested
    assert legacy_continuation.raise_planned_continuation is raise_planned_continuation
    assert legacy_continuation.request_continuation is request_continuation
    assert continuation_module.PlannedContinuation is PlannedContinuation
