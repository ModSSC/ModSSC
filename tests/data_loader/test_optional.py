import pytest

from modssc.data_loader.errors import OptionalDependencyError
from modssc.data_loader.optional import optional_import_attr


def test_optional_import_attr_success():
    path_cls = optional_import_attr("pathlib", "Path", extra="test")
    from pathlib import Path

    assert path_cls is Path


def test_optional_import_attr_missing_attribute():
    with pytest.raises(OptionalDependencyError) as excinfo:
        optional_import_attr("os", "non_existent_attr", extra="test_extra", purpose="testing")

    assert excinfo.value.extra == "test_extra"
    assert excinfo.value.purpose == "testing"
    assert isinstance(excinfo.value.__cause__, AttributeError)


def test_optional_import_attr_missing_module():
    with pytest.raises(OptionalDependencyError):
        optional_import_attr("non_existent_module", "attr", extra="test")


def test_optional_import_success_and_failure() -> None:
    from modssc.data_loader.optional import optional_import

    mod = optional_import("json", extra="core", purpose="json")
    assert mod.dumps({"a": 1})

    with pytest.raises(OptionalDependencyError) as exc:
        optional_import("this_module_does_not_exist_123", extra="openml", purpose="openml")
    assert exc.value.extra == "openml"
