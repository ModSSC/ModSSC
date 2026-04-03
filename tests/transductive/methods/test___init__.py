from __future__ import annotations

import importlib

import pytest

import modssc.transductive.methods as methods
from modssc.transductive.methods.classic.label_propagation import label_propagation


def _assert_module_importable(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing.startswith("modssc"):
            raise
        pytest.skip(f"Optional dependency missing while importing {module_name}: {missing}")
    except Exception as exc:
        if exc.__class__.__name__ == "OptionalDependencyError" or 'pip install "modssc[' in str(
            exc
        ):
            pytest.skip(f"Optional dependency missing while importing {module_name}: {exc}")
        raise


def test_module_importable() -> None:
    _assert_module_importable("modssc.transductive.methods")


def test_lazy_method_exports_resolve_known_symbols() -> None:
    assert methods.label_propagation is label_propagation
