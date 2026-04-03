from __future__ import annotations

from modssc.supervised.registry_data import BUILTIN_CLASSIFIERS


def test_builtin_classifier_specs_are_non_empty_and_unique() -> None:
    assert BUILTIN_CLASSIFIERS
    keys = [entry["key"] for entry in BUILTIN_CLASSIFIERS]
    assert len(keys) == len(set(keys))


def test_builtin_classifier_specs_have_backend_entries() -> None:
    for entry in BUILTIN_CLASSIFIERS:
        assert entry["description"]
        assert entry["preferred_backends"]
        assert entry["backends"]
        for backend in entry["backends"]:
            assert "backend" in backend
            assert "factory" in backend
            assert "supports_gpu" in backend
