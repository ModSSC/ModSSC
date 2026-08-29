from __future__ import annotations

import importlib

import pytest

from modssc.data_loader.catalog.graph import GRAPH_CATALOG


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
    _assert_module_importable("modssc.data_loader.catalog.graph")


def test_cora_catalog_pins_planetoid_public_split() -> None:
    cora = GRAPH_CATALOG["cora"]

    assert cora.uri == "pyg:Planetoid/Cora"
    assert cora.source_kwargs == {"split": "public"}


def test_other_planetoid_catalog_entries_keep_default_constructor_options() -> None:
    assert GRAPH_CATALOG["citeseer"].source_kwargs == {}
    assert GRAPH_CATALOG["pubmed"].source_kwargs == {}
