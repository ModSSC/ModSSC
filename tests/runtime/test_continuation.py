from __future__ import annotations

import signal
from dataclasses import dataclass

import pytest

from modssc.runtime import continuation as continuation_module
from modssc.runtime.continuation import (
    PLANNED_CONTINUATION_EXIT_CODE,
    PlannedContinuation,
    clear_continuation_request,
    continuation_requested,
    continuation_signal_handler,
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
    assert raised.value.exit_code == 75
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


def test_runtime_package_exports_the_canonical_continuation_types() -> None:
    assert continuation_module.PlannedContinuation is PlannedContinuation
    assert PLANNED_CONTINUATION_EXIT_CODE == 75


def test_disabled_signal_scope_clears_state_without_installing_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_continuation(signal.SIGUSR1)

    def _unexpected_signal_installation(*_args: object) -> None:
        raise AssertionError("disabled continuation scope must not install a signal handler")

    monkeypatch.setattr(continuation_module.signal, "signal", _unexpected_signal_installation)
    with continuation_signal_handler(enabled=False):
        assert not continuation_requested()
        request_continuation(signal.SIGUSR1)

    assert not continuation_requested()
    assert "MODSSC_CONTINUATION_SIGNAL" not in continuation_module.os.environ


def test_enabled_signal_scope_records_request_and_restores_previous_handler() -> None:
    previous_handler = signal.getsignal(signal.SIGUSR1)

    def raise_after_signal() -> None:
        with continuation_signal_handler(enabled=True):
            installed_handler = signal.getsignal(signal.SIGUSR1)
            assert callable(installed_handler)
            installed_handler(signal.SIGUSR1, None)
            assert continuation_requested()
            assert PlannedContinuation().signum == signal.SIGUSR1
            raise RuntimeError("leave scope")

    try:
        with pytest.raises(RuntimeError, match="leave scope"):
            raise_after_signal()

        assert signal.getsignal(signal.SIGUSR1) is previous_handler
        assert not continuation_requested()
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)
        clear_continuation_request()


def test_signal_scope_propagates_frozen_exception_without_rewriting_traceback() -> None:
    @dataclass(frozen=True)
    class FrozenRuntimeError(RuntimeError):
        message: str

        def __post_init__(self) -> None:
            RuntimeError.__init__(self, self.message)

    error = FrozenRuntimeError("frozen failure")

    def raise_frozen_error() -> None:
        with continuation_signal_handler(enabled=False):
            raise error

    with pytest.raises(FrozenRuntimeError) as raised:
        raise_frozen_error()

    assert raised.value is error
