from __future__ import annotations

import importlib

import pytest

from modssc.data_loader.catalog.tabular import TABULAR_CATALOG


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
    _assert_module_importable("modssc.data_loader.catalog.tabular")


def test_wdbc_is_a_distinct_pinned_openml_dataset() -> None:
    wdbc = TABULAR_CATALOG["wdbc"]
    breast_cancer = TABULAR_CATALOG["breast_cancer"]

    assert wdbc.uri == "openml:1510"
    assert wdbc.source_kwargs == {"data_id": 1510}
    assert wdbc.modality == "tabular"
    assert wdbc.task == "classification"
    assert wdbc.required_extra == "openml"
    assert wdbc.license == "CC BY 4.0"
    assert wdbc.homepage == (
        "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"
    )
    assert "10.24432/C5DW2B" in (wdbc.citation or "")

    assert breast_cancer.uri == "openml:15"
    assert breast_cancer.source_kwargs == {"data_id": 15}
    assert wdbc.fingerprint(schema_version=1) != breast_cancer.fingerprint(schema_version=1)


def test_vote_is_the_pinned_congressional_voting_records_dataset() -> None:
    vote = TABULAR_CATALOG["vote"]

    assert vote.uri == "openml:56"
    assert vote.source_kwargs == {"data_id": 56}
    assert vote.modality == "tabular"
    assert vote.task == "classification"
    assert vote.required_extra == "openml"
    assert vote.license == "CC BY 4.0"
    assert vote.homepage == ("https://archive.ics.uci.edu/dataset/105/congressional+voting+records")
    assert "10.24432/C5C01P" in (vote.citation or "")
    assert "16 nominal" in vote.description

    assert vote.fingerprint(schema_version=1) != TABULAR_CATALOG["adult"].fingerprint(
        schema_version=1
    )


def test_wine_is_the_pinned_uci_openml_dataset() -> None:
    wine = TABULAR_CATALOG["wine"]

    assert wine.uri == "openml:187"
    assert wine.source_kwargs == {"data_id": 187}
    assert wine.modality == "tabular"
    assert wine.task == "classification"
    assert wine.required_extra == "openml"
    assert wine.license == "CC BY 4.0"
    assert wine.homepage == "https://archive.ics.uci.edu/dataset/109/wine"
    assert "10.24432/C5PC7J" in (wine.citation or "")
    assert "178 rows" in wine.description
