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
    _assert_module_importable("modssc.inductive.methods.free_match")


def test_spec_defaults() -> None:
    module = _assert_module_importable("modssc.inductive.methods.free_match")
    spec = module.FreeMatchSpec()
    assert spec.temperature == pytest.approx(0.5)
    assert spec.mu == 7
    assert spec.lambda_u == pytest.approx(1.0)
    assert spec.lambda_e == pytest.approx(0.001)
    assert spec.ema_p == pytest.approx(0.999)
    assert spec.use_quantile is True
    assert spec.batch_size == 64
