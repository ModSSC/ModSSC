from __future__ import annotations

import importlib

import pytest

from modssc.data_loader.catalog.text import TEXT_CATALOG


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
    _assert_module_importable("modssc.data_loader.catalog.text")


def test_webkb_course_cotraining_is_pinned_to_the_stdlib_provider() -> None:
    webkb = TEXT_CATALOG["webkb_course_cotraining"]

    assert webkb.provider == "webkb1998"
    assert webkb.uri == "webkb1998:course"
    assert webkb.modality == "text"
    assert webkb.task == "classification"
    assert webkb.required_extra is None
    assert webkb.source_kwargs == {}
    assert webkb.license is None
    assert "1,051" in webkb.description
    assert "10.1145/279943.279962" in (webkb.citation or "")
