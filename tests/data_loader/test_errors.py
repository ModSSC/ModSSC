from modssc.data_loader.errors import (
    DataLoaderError,
    DatasetNotCachedError,
    OptionalDependencyError,
    ProviderNotFoundError,
    UnknownDatasetError,
)


def test_error_types_and_messages() -> None:
    err = UnknownDatasetError("does_not_exist")
    assert isinstance(err, DataLoaderError)
    assert "does_not_exist" in str(err)

    err2 = DatasetNotCachedError("toy")
    assert isinstance(err2, DataLoaderError)
    assert "toy" in str(err2)

    err3 = ProviderNotFoundError("nope")
    assert "nope" in str(err3)


def test_optional_dependency_error_with_purpose():
    err = OptionalDependencyError(extra="vision", purpose="loading images")
    msg = str(err)
    assert "Missing optional dependency extra: 'vision'." in msg
    assert "Required for: loading images." in msg
    assert 'pip install "modssc[vision]"' in msg


def test_optional_dependency_error_without_purpose():
    err = OptionalDependencyError(extra="audio")
    msg = str(err)
    assert "Missing optional dependency extra: 'audio'." in msg
    assert "Required for" not in msg
    assert 'pip install "modssc[audio]"' in msg
