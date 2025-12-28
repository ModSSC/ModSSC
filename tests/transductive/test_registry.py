import sys
import types
from unittest.mock import patch

import pytest

from modssc.transductive.registry import (
    _REGISTRY,
    _debug_registry,
    available_methods,
    get_method_class,
    get_method_info,
    register_method,
)


def test_registry_duplicate_registration():
    with patch.dict(_REGISTRY, {}, clear=True):
        register_method("test_method", "pkg.module:Class")

        register_method("test_method", "pkg.module:Class")

        with pytest.raises(ValueError, match="already registered"):
            register_method("test_method", "pkg.module:OtherClass")


def test_available_methods():
    with patch.dict(_REGISTRY, {}, clear=True):
        register_method("b_method", "pkg:B")
        register_method("a_method", "pkg:A")
        register_method("planned_method", "pkg:C", status="planned")

        methods = available_methods()
        assert methods == sorted(methods)
        assert "a_method" in methods
        assert "b_method" in methods
        assert "planned_method" not in methods

        methods_all = available_methods(available_only=False)
        assert "planned_method" in methods_all


def test_get_method_class_success():
    with patch.dict(_REGISTRY, {}, clear=True):
        register_method("json_decoder", "json.decoder:JSONDecoder")

        cls = get_method_class("json_decoder")
        from json.decoder import JSONDecoder

        assert cls is JSONDecoder


def test_get_method_class_unknown():
    with (
        patch.dict(_REGISTRY, {}, clear=True),
        pytest.raises(KeyError, match="Unknown method_id: 'unknown'"),
    ):
        get_method_class("unknown")


def test_register_method_invalid_inputs():
    with patch.dict(_REGISTRY, {}, clear=True):
        with pytest.raises(ValueError, match="method_id must be a non-empty string"):
            register_method("", "pkg.module:Class")

        with pytest.raises(ValueError, match="import_path must be of the form"):
            register_method("ok", "pkg.module.Class")


def test_register_method_invalid_status():
    with patch.dict(_REGISTRY, {}, clear=True), pytest.raises(ValueError, match="status must be"):
        register_method("bad_status", "pkg.module:Class", status="invalid")


def test_get_method_info_requires_methodinfo():
    fake_mod = types.ModuleType("fake_mod")

    class DummyMethod:
        info = "not a MethodInfo"

    fake_mod.DummyMethod = DummyMethod

    with (
        patch.dict(sys.modules, {"fake_mod": fake_mod}),
        patch.dict(_REGISTRY, {}, clear=True),
    ):
        register_method("dummy", "fake_mod:DummyMethod")
        with pytest.raises(TypeError, match="must expose a class attribute"):
            get_method_info("dummy")


def test_debug_registry_contains_builtins():
    out = _debug_registry()
    assert "label_propagation" in out
    assert "label_spreading" in out
