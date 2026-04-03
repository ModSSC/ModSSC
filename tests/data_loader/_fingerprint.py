from __future__ import annotations

from modssc.data_loader.types import DatasetSpec


def test_fingerprint_ignores_doc_fields() -> None:
    a = DatasetSpec(
        key="toy",
        provider="toy",
        uri="toy:default",
        modality="tabular",
        task="classification",
        description="a",
        homepage="x",
        license="MIT",
        citation="c1",
        source_kwargs={"seed": 0},
    )
    b = DatasetSpec(
        key="toy",
        provider="toy",
        uri="toy:default",
        modality="tabular",
        task="classification",
        description="b",
        homepage="y",
        license="Apache",
        citation="c2",
        source_kwargs={"seed": 0},
    )
    assert a.fingerprint(schema_version=1) == b.fingerprint(schema_version=1)


def test_fingerprint_changes_with_kwargs() -> None:
    a = DatasetSpec(
        key="toy",
        provider="toy",
        uri="toy:default",
        modality="tabular",
        task="classification",
        description="a",
        source_kwargs={"seed": 0},
    )
    b = DatasetSpec(
        key="toy",
        provider="toy",
        uri="toy:default",
        modality="tabular",
        task="classification",
        description="a",
        source_kwargs={"seed": 1},
    )
    assert a.fingerprint(schema_version=1) != b.fingerprint(schema_version=1)
