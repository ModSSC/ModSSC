from __future__ import annotations

import json

import pytest

from modssc.data_loader.types import DatasetIdentity, DatasetSpec, DownloadReport


def test_dataset_identity_fingerprint_is_hex() -> None:
    identity = DatasetIdentity(
        provider="toy",
        canonical_uri="toy:default",
        dataset_id="default",
        version=None,
        modality="tabular",
        task="classification",
        required_extra=None,
        resolved_kwargs={"seed": 0, "n_samples": 10},
    )
    fp = identity.fingerprint(schema_version=1)
    assert isinstance(fp, str)
    assert len(fp) == 64
    int(fp, 16)

    bad = DatasetIdentity(
        provider="toy",
        canonical_uri="toy:default",
        dataset_id="default",
        version=None,
        modality="tabular",
        task="classification",
        required_extra=None,
        resolved_kwargs={"bad": object()},
    )
    with pytest.raises(
        ValueError, match="DatasetIdentity.resolved_kwargs must be JSON serializable"
    ):
        bad.fingerprint(schema_version=1)


def test_dataset_spec_fingerprint_and_as_dict() -> None:
    spec = DatasetSpec(
        key="toy",
        provider="toy",
        uri="toy:default",
        modality="tabular",
        task="classification",
        description="x",
        required_extra=None,
        source_kwargs={"seed": 0},
    )
    d = spec.as_dict()
    json.dumps(d)
    fp = spec.fingerprint(schema_version=1)
    assert len(fp) == 64


def test_dataset_spec_fingerprint_not_serializable():
    spec = DatasetSpec(
        key="test",
        provider="test",
        uri="test:test",
        modality="test",
        task="test",
        description="test",
        source_kwargs={"obj": object()},
    )
    with pytest.raises(ValueError, match="DatasetSpec.source_kwargs must be JSON serializable"):
        spec.fingerprint(schema_version=1)


def test_download_report_summary_text() -> None:
    rep = DownloadReport(
        downloaded=["a", "b"],
        skipped_already_cached=["c"],
        skipped_missing_extras=["d"],
        missing_extras={"vision": ["d"]},
        failed={"e": "boom"},
    )
    txt = rep.summary()
    assert "Downloaded" in txt
    assert "Skipped (missing extras)" in txt
    assert "Failed" in txt


def test_download_report_has_failures():
    report_ok = DownloadReport()
    assert not report_ok.has_failures()

    report_fail = DownloadReport(failed={"foo": "bar"})
    assert report_fail.has_failures()


def test_dataset_identity_as_dict():
    identity = DatasetIdentity(
        provider="test",
        canonical_uri="test:test",
        dataset_id="test",
        version=None,
        modality="test",
        task="test",
    )
    d = identity.as_dict()
    assert d["provider"] == "test"
