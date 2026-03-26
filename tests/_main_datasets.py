from __future__ import annotations

import runpy
import sys

import pytest


def test_run_package_as_module(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["modssc", "datasets", "list"])
    monkeypatch.delitem(sys.modules, "modssc.__main__", raising=False)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("modssc", run_name="__main__")
    assert exc.value.code == 0

    out = capsys.readouterr().out.lower()
    assert "toy" in out
