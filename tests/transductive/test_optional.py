from __future__ import annotations

import importlib

import pytest

from modssc.transductive.errors import OptionalDependencyError
from modssc.transductive.optional import optional_import


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
    _assert_module_importable("modssc.transductive.optional")


def test_optional_import_error_factory_uses_package_and_extra() -> None:
    with pytest.raises(OptionalDependencyError) as exc:
        optional_import("definitely_missing_transductive_module_xyz", extra="transductive-torch")
    assert "transductive-torch" in str(exc.value)
    assert "definitely_missing_transductive_module_xyz" in str(exc.value)
