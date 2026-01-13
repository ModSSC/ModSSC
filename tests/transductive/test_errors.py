from modssc.transductive.errors import (
    OptionalDependencyError,
    TransductiveNotImplementedError,
    TransductiveValidationError,
)


def test_transductive_optional_dependency_error_str():
    err = OptionalDependencyError(package="torch", extra="transductive-torch", message="boom")
    msg = str(err)
    assert "boom" in msg
    assert "modssc[transductive-torch]" in msg


def test_transductive_not_implemented_error_no_hint():
    err = TransductiveNotImplementedError("mystery")
    msg = str(err)
    assert "mystery" in msg
    assert "not implemented" in msg
    assert err.method_id == "mystery"
    assert err.hint is None


def test_transductive_not_implemented_error_with_hint():
    err = TransductiveNotImplementedError("mystery", hint="use numpy")
    msg = str(err)
    assert "use numpy" in msg
    assert err.method_id == "mystery"
    assert err.hint == "use numpy"


def test_transductive_validation_error_is_value_error():
    err = TransductiveValidationError("bad")
    assert isinstance(err, ValueError)
