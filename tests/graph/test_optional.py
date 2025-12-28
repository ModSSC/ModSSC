import pytest

from modssc.graph.errors import OptionalDependencyError
from modssc.graph.optional import optional_import, optional_import_attr


def test_optional_import_failure():
    with pytest.raises(OptionalDependencyError) as exc:
        optional_import("non_existent_module_xyz", extra="test")
    assert "Missing optional dependency extra: 'test'" in str(exc.value)


def test_optional_dependency_error_message():
    err = OptionalDependencyError(extra="foo", purpose="bar")
    assert "Required for: bar" in str(err)

    err = OptionalDependencyError(extra="foo", message="Module not found")
    assert "(Import error: Module not found)" in str(err)


def test_optional_import_attr_missing_attribute():
    with pytest.raises(OptionalDependencyError) as exc:
        optional_import_attr("os", "non_existent_attr", extra="test")
    assert "Missing optional dependency extra: 'test'" in str(exc.value)
    assert "Import error" in str(exc.value)
