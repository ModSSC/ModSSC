import pytest

from modssc.supervised.errors import OptionalDependencyError
from modssc.supervised.optional import get_attr, has_module, optional_import


def test_optional_import_missing():
    with pytest.raises(OptionalDependencyError) as exc:
        optional_import("non_existent_module_xyz_123", extra="test", feature="foo")
    assert (
        "Optional dependency missing for 'foo'. Install with: pip install \"modssc[test]\""
        in str(exc.value)
    )


def test_has_module_missing():
    assert not has_module("non_existent_module_xyz_123")


def test_has_module_present():
    assert has_module("os")


def test_get_attr():
    class Obj:
        x = 1

    assert get_attr(Obj(), "x") == 1
    assert get_attr(Obj(), "y", 2) == 2
