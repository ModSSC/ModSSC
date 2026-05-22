from __future__ import annotations

import importlib

import pytest


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
    _assert_module_importable("modssc.transductive.methods.gnn.sgc")


def test_spec_defaults() -> None:
    module = _assert_module_importable("modssc.transductive.methods.gnn.sgc")
    spec = module.SGCSpec()
    assert spec.k == 2
    assert spec.lr == pytest.approx(0.2)
    assert spec.weight_decay == pytest.approx(5e-6)
    assert spec.max_epochs == 100
    assert spec.patience == 100
    assert spec.weight_decay_scope == "all"
