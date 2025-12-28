from modssc.preprocess.errors import OptionalDependencyError


def test_optional_dependency_error_with_purpose():
    err = OptionalDependencyError(extra="test-extra", purpose="testing stuff")
    msg = str(err)
    assert "Missing optional dependency extra: 'test-extra'." in msg
    assert "Required for: testing stuff." in msg
    assert 'Install with: pip install "modssc[test-extra]"' in msg


def test_optional_dependency_error_without_purpose():
    err = OptionalDependencyError(extra="test-extra")
    msg = str(err)
    assert "Missing optional dependency extra: 'test-extra'." in msg
    assert "Required for:" not in msg
    assert 'Install with: pip install "modssc[test-extra]"' in msg
