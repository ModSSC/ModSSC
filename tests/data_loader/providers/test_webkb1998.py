from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from modssc.data_loader.errors import DataLoaderError
from modssc.data_loader.providers import webkb1998
from modssc.data_loader.providers.webkb1998 import (
    WebKB1998Provider,
    _download_archive,
    _read_webkb_archive,
)
from modssc.data_loader.types import DatasetIdentity
from modssc.data_loader.uri import ParsedURI


def _archive_bytes(
    files: dict[str, bytes],
    *,
    extra_members: list[tarfile.TarInfo] | None = None,
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        directories = [
            "course-cotrain-data",
            "course-cotrain-data/fulltext",
            "course-cotrain-data/fulltext/course",
            "course-cotrain-data/fulltext/non-course",
            "course-cotrain-data/inlinks",
            "course-cotrain-data/inlinks/course",
            "course-cotrain-data/inlinks/non-course",
        ]
        for name in directories:
            member = tarfile.TarInfo(name)
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        for member in extra_members or []:
            archive.addfile(member)
    return output.getvalue()


def _paired_files() -> dict[str, bytes]:
    root = "course-cotrain-data"
    return {
        f"{root}/fulltext/course/course-a": b"<html>Course A</html>\xa3",
        f"{root}/inlinks/course/course-a": b"course a",
        f"{root}/fulltext/non-course/other-b": b"<html>Other B</html>",
        f"{root}/inlinks/non-course/other-b": b"other b",
    }


def _write_archive(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def test_resolve_pins_the_historical_artifact() -> None:
    identity = WebKB1998Provider().resolve(
        ParsedURI(provider="webkb1998", reference=" course "), options={}
    )

    assert identity.canonical_uri == "webkb1998:course"
    assert identity.dataset_id == "course"
    assert identity.modality == "text"
    assert identity.required_extra is None
    assert identity.resolved_kwargs["source_url"] == webkb1998.SOURCE_URL
    assert identity.resolved_kwargs["archive_sha256"] == webkb1998.ARCHIVE_SHA256
    assert identity.resolved_kwargs["expected_pairs"] == 1051


def test_resolve_rejects_unknown_reference_and_source_overrides() -> None:
    provider = WebKB1998Provider()
    with pytest.raises(DataLoaderError, match="Unknown WebKB"):
        provider.resolve(ParsedURI(provider="webkb1998", reference="other"), options={})
    with pytest.raises(DataLoaderError, match="does not accept source overrides"):
        provider.resolve(
            ParsedURI(provider="webkb1998", reference="course"),
            options={"source_url": "file:///tmp/not-official"},
        )


def test_provider_downloads_parses_and_reuses_a_mocked_miniature_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _archive_bytes(_paired_files())
    digest = hashlib.sha256(payload).hexdigest()
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        assert request.full_url == webkb1998.SOURCE_URL
        assert timeout == 60
        return _Response(payload)

    monkeypatch.setattr(webkb1998, "ARCHIVE_SHA256", digest)
    monkeypatch.setattr(webkb1998, "EXPECTED_PAIRS", 2)
    monkeypatch.setattr(webkb1998, "EXPECTED_CLASS_COUNTS", {"course": 1, "non-course": 1})
    monkeypatch.setattr(webkb1998.urllib.request, "urlopen", fake_urlopen)

    provider = WebKB1998Provider()
    identity = provider.resolve(ParsedURI(provider="webkb1998", reference="course"), options={})
    dataset = provider.load_canonical(identity, raw_dir=tmp_path)

    assert calls == 1
    assert dataset.test is None
    assert dataset.train.X.shape == (2, 2)
    assert dataset.train.X.dtype == object
    assert dataset.train.y.dtype == np.int64
    assert dataset.train.y.tolist() == [0, 1]
    assert dataset.train.X[1, 0].endswith("\xa3")
    assert dataset.meta["archive_sha256"] == digest
    assert dataset.meta["archive_size_bytes"] == len(payload)
    assert dataset.meta["class_counts"] == {"course": 1, "non-course": 1}
    assert dataset.meta["label_mapping"] == {"non-course": 0, "course": 1}
    assert dataset.meta["sample_ids"] == ["non-course/other-b", "course/course-a"]
    assert len(dataset.meta["fulltext_sha256"]) == 64
    assert len(dataset.meta["inlinks_sha256"]) == 64
    assert len(dataset.meta["pair_manifest_sha256"]) == 64
    assert len(dataset.meta["sample_ids_sha256"]) == 64
    assert dataset.meta["license"] is None

    monkeypatch.setattr(
        webkb1998.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("cached archive should not be downloaded again"),
    )
    cached = provider.load_canonical(identity, raw_dir=tmp_path)
    assert np.array_equal(cached.train.X, dataset.train.X)


def test_load_rejects_noncanonical_identity(tmp_path: Path) -> None:
    identity = DatasetIdentity(
        provider="webkb1998",
        canonical_uri="webkb1998:course",
        dataset_id="course",
        version="test",
        modality="text",
        task="classification",
        resolved_kwargs={"source_url": "file:///tmp/override"},
    )
    with pytest.raises(DataLoaderError, match="does not match"):
        WebKB1998Provider().load_canonical(identity, raw_dir=tmp_path)


def test_download_rejects_corrupt_cached_and_downloaded_archives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "cached.tar.gz"
    destination.write_bytes(b"corrupt")
    with pytest.raises(DataLoaderError, match="SHA-256 mismatch"):
        _download_archive("https://example.invalid/archive", destination, expected_sha256="0" * 64)

    destination.unlink()
    monkeypatch.setattr(
        webkb1998.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(b"wrong download"),
    )
    with pytest.raises(DataLoaderError, match="downloaded archive SHA-256 mismatch"):
        _download_archive("https://example.invalid/archive", destination, expected_sha256="0" * 64)
    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_download_rejects_nonfile_destination_and_network_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "archive"
    destination.mkdir()
    with pytest.raises(DataLoaderError, match="not a file"):
        _download_archive("https://example.invalid/archive", destination, expected_sha256="0" * 64)

    destination.rmdir()

    def fail_urlopen(*_args, **_kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(webkb1998.urllib.request, "urlopen", fail_urlopen)
    with pytest.raises(DataLoaderError, match="Unable to download"):
        _download_archive("https://example.invalid/archive", destination, expected_sha256="0" * 64)
    assert list(tmp_path.glob("*.part")) == []


def test_download_wraps_temporary_file_creation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_tempfile(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(webkb1998.tempfile, "NamedTemporaryFile", fail_tempfile)
    with pytest.raises(DataLoaderError, match="Unable to download"):
        _download_archive(
            "https://example.invalid/archive",
            tmp_path / "archive.tar.gz",
            expected_sha256="0" * 64,
        )


def test_download_preserves_provider_error_before_temporary_path_assignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingContext:
        def __enter__(self):
            raise DataLoaderError("intentional provider failure")

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        webkb1998.tempfile, "NamedTemporaryFile", lambda *_args, **_kwargs: FailingContext()
    )
    with pytest.raises(DataLoaderError, match="intentional provider failure"):
        _download_archive(
            "https://example.invalid/archive",
            tmp_path / "archive.tar.gz",
            expected_sha256="0" * 64,
        )


def test_download_enforces_size_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(webkb1998, "_MAX_DOWNLOAD_BYTES", 2)
    monkeypatch.setattr(
        webkb1998.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(b"too large"),
    )
    with pytest.raises(DataLoaderError, match="maximum allowed download size"):
        _download_archive(
            "https://example.invalid/archive",
            tmp_path / "archive.tar.gz",
            expected_sha256="0" * 64,
        )


def test_read_rejects_view_misalignment(tmp_path: Path) -> None:
    files = _paired_files()
    del files["course-cotrain-data/inlinks/course/course-a"]
    archive_path = tmp_path / "misaligned.tar.gz"
    _write_archive(archive_path, _archive_bytes(files))

    with pytest.raises(DataLoaderError, match="not exactly aligned"):
        _read_webkb_archive(
            archive_path,
            expected_pairs=2,
            expected_class_counts={"course": 1, "non-course": 1},
        )


def test_read_rejects_wrong_pair_and_class_counts(tmp_path: Path) -> None:
    archive_path = tmp_path / "mini.tar.gz"
    _write_archive(archive_path, _archive_bytes(_paired_files()))

    with pytest.raises(DataLoaderError, match="Unexpected number"):
        _read_webkb_archive(
            archive_path,
            expected_pairs=3,
            expected_class_counts={"course": 1, "non-course": 2},
        )
    with pytest.raises(DataLoaderError, match="Unexpected WebKB 1998 class counts"):
        _read_webkb_archive(
            archive_path,
            expected_pairs=2,
            expected_class_counts={"course": 2, "non-course": 0},
        )


@pytest.mark.parametrize(
    "member",
    [
        tarfile.TarInfo("../escape"),
        tarfile.TarInfo("/absolute"),
        tarfile.TarInfo("course-cotrain-data\\evil"),
        tarfile.TarInfo("course-cotrain-data/./evil"),
    ],
)
def test_read_rejects_unsafe_member_paths(tmp_path: Path, member: tarfile.TarInfo) -> None:
    member.size = 0
    archive_path = tmp_path / "unsafe.tar.gz"
    _write_archive(archive_path, _archive_bytes(_paired_files(), extra_members=[member]))
    with pytest.raises(DataLoaderError, match="Unsafe path"):
        _read_webkb_archive(
            archive_path,
            expected_pairs=2,
            expected_class_counts={"course": 1, "non-course": 1},
        )


def test_read_rejects_links_unexpected_members_and_duplicates(tmp_path: Path) -> None:
    link = tarfile.TarInfo("course-cotrain-data/fulltext/course/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../outside"
    link_archive = tmp_path / "link.tar.gz"
    _write_archive(link_archive, _archive_bytes(_paired_files(), extra_members=[link]))
    with pytest.raises(DataLoaderError, match="Unsupported non-regular"):
        _read_webkb_archive(
            link_archive,
            expected_pairs=2,
            expected_class_counts={"course": 1, "non-course": 1},
        )

    unexpected = tarfile.TarInfo("course-cotrain-data/README")
    unexpected.size = 0
    unexpected_archive = tmp_path / "unexpected.tar.gz"
    _write_archive(unexpected_archive, _archive_bytes(_paired_files(), extra_members=[unexpected]))
    with pytest.raises(DataLoaderError, match="Unexpected file"):
        _read_webkb_archive(
            unexpected_archive,
            expected_pairs=2,
            expected_class_counts={"course": 1, "non-course": 1},
        )

    duplicate_archive = tmp_path / "duplicate.tar.gz"
    duplicate = tarfile.TarInfo("course-cotrain-data/fulltext/course/course-a")
    duplicate.size = 0
    _write_archive(duplicate_archive, _archive_bytes(_paired_files(), extra_members=[duplicate]))
    with pytest.raises(DataLoaderError, match="Duplicate WebKB"):
        _read_webkb_archive(
            duplicate_archive,
            expected_pairs=2,
            expected_class_counts={"course": 1, "non-course": 1},
        )


def test_read_rejects_unexpected_directory(tmp_path: Path) -> None:
    directory = tarfile.TarInfo("course-cotrain-data/unexpected")
    directory.type = tarfile.DIRTYPE
    archive_path = tmp_path / "directory.tar.gz"
    _write_archive(archive_path, _archive_bytes(_paired_files(), extra_members=[directory]))
    with pytest.raises(DataLoaderError, match="Unexpected directory"):
        _read_webkb_archive(
            archive_path,
            expected_pairs=2,
            expected_class_counts={"course": 1, "non-course": 1},
        )


def test_read_rejects_unreadable_and_truncated_members(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_path = tmp_path / "mini.tar.gz"
    _write_archive(archive_path, _archive_bytes(_paired_files()))

    monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda *_args, **_kwargs: None)
    with pytest.raises(DataLoaderError, match="Unable to read"):
        _read_webkb_archive(
            archive_path,
            expected_pairs=2,
            expected_class_counts={"course": 1, "non-course": 1},
        )

    monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda *_args, **_kwargs: io.BytesIO(b""))
    with pytest.raises(DataLoaderError, match="Truncated"):
        _read_webkb_archive(
            archive_path,
            expected_pairs=2,
            expected_class_counts={"course": 1, "non-course": 1},
        )


def test_read_rejects_invalid_tar_and_archive_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    invalid = tmp_path / "invalid.tar.gz"
    invalid.write_bytes(b"not a tar")
    with pytest.raises(DataLoaderError, match="Invalid WebKB"):
        _read_webkb_archive(
            invalid,
            expected_pairs=2,
            expected_class_counts={"course": 1, "non-course": 1},
        )

    archive_path = tmp_path / "mini.tar.gz"
    _write_archive(archive_path, _archive_bytes(_paired_files()))
    monkeypatch.setattr(webkb1998, "_MAX_TAR_MEMBERS", 1)
    with pytest.raises(DataLoaderError, match="too many members"):
        _read_webkb_archive(
            archive_path,
            expected_pairs=2,
            expected_class_counts={"course": 1, "non-course": 1},
        )


def test_read_rejects_member_and_total_size_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_path = tmp_path / "mini.tar.gz"
    _write_archive(archive_path, _archive_bytes(_paired_files()))

    monkeypatch.setattr(webkb1998, "_MAX_MEMBER_BYTES", 1)
    with pytest.raises(DataLoaderError, match="Invalid member size"):
        _read_webkb_archive(
            archive_path,
            expected_pairs=2,
            expected_class_counts={"course": 1, "non-course": 1},
        )

    monkeypatch.setattr(webkb1998, "_MAX_MEMBER_BYTES", 1024)
    monkeypatch.setattr(webkb1998, "_MAX_TOTAL_MEMBER_BYTES", 1)
    with pytest.raises(DataLoaderError, match="maximum uncompressed size"):
        _read_webkb_archive(
            archive_path,
            expected_pairs=2,
            expected_class_counts={"course": 1, "non-course": 1},
        )
