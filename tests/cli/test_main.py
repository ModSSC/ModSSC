from __future__ import annotations

import runpy
import sys

import pytest

from modssc.__about__ import __version__


def test_python_m_modssc_calls_main(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["modssc", "--version"])

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("modssc.__main__", run_name="__main__")

    assert exc.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out == __version__ or out.endswith(__version__)


def test_import_main_module_does_not_execute() -> None:
    import importlib

    mod = importlib.import_module("modssc.__main__")
    assert hasattr(mod, "app")
